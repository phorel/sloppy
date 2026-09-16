#!/usr/bin/env python3
"""Phase 3: IRC AI bot — connects, joins #hive, responds to AI: prompts via llama.cpp."""

import argparse
import collections
import dataclasses
import fnmatch
import json
import os
import pathlib
import random
import re
import signal
import socket
import sys
import threading
import time
import urllib.request
from typing import Any, NamedTuple
from openai import OpenAI

import config
import profiles
import recall
import summarizer
import web

# What this build calls itself. Bumped by hand at release; the tag on the
# commit and this string are meant to agree, and nothing enforces that but a
# reader, so change them together.
VERSION = "0.7.5"

# Read at import so the constants below have values; main() reads it again
# through reload_config() once the sinks exist, which is what reports on it.
config.load()


# --------------------------------------------------------------------------- #
# Output routing. The business logic never prints directly: every line of
# output goes through one of the sinks below. The TUI replaces these callables
# to paint the panes; left at their default they echo to stdout (so the core
# still runs stand-alone and anything importing it sees normal output).
#   irc     -> the live IRC log pane      (raw server lines, other users, sends)
#   chat    -> chat text in the log pane   (another user's line, "nick: text")
#   action  -> bot actions in the log pane (addressed, replied, mood, shutdown)
#   debug   -> verbose prompt dumps       (hidden by the TUI by default)
# --------------------------------------------------------------------------- #
def _stdout(msg: str) -> None:
    print(msg, flush=True)


irc_sink = _stdout
action_sink = _stdout
chat_sink = _stdout
speak_sink = _stdout
debug_sink = _stdout
warning_sink = _stdout


def irc(msg: str) -> None:
    """An IRC event for the log pane: a raw server line, another user's
    message, the userlist, or the bot sending its own line."""
    irc_sink(msg)


def chat(msg: str) -> None:
    """A line of channel chat in the log pane: another user's line as
    "nick: text". Posted from _note_recent so the log mirrors the history."""
    chat_sink(msg)


def action(msg: str) -> None:
    """A bot action in the log pane: being addressed, replying, a mood
    change, an interjection, or shutting down."""
    action_sink(msg)


def speak(msg: str) -> None:
    """A line the bot actually spoke, in the log pane: shown light blue in the
    TUI, distinct from action (yellow) so a line of banter is easy to tell
    apart from a bot status line."""
    speak_sink(msg)


def debug(msg: str) -> None:
    """Verbose debug (the full prompts handed to the model). Hidden by the
    TUI by default; still reachable for troubleshooting."""
    debug_sink(msg)


def warning(msg: str) -> None:
    """A problem, shown red in the log pane: a failed or unusable LLM call, a
    rejected summarizer output, or a connection the bot could not make. The
    channel never sees these -- it gets a line in character instead."""
    warning_sink(msg)

# Every tunable registers itself as it is assigned, so its config key and its
# default exist in exactly one place and reload_config can re-derive the lot
# without a second copy of either. Name -> (key, default).
_TUNABLES: dict[str, tuple[str, Any]] = {}


def _tune(name: str, key: str, default: Any) -> Any:
    """Record a tunable and return its configured value."""
    _TUNABLES[name] = (key, default)
    return config.get(key, default)


# Which server, which channel, which nick. Read with config.get rather than
# _tune, deliberately: _tune registers a lever for live reload, and these four
# cannot take effect without a reconnect -- a lever that silently does nothing
# is worse than no lever. Set them in sloppy.local.toml, which is never
# committed; the tracked sloppy.toml carries neutral examples so a public
# checkout shows the shape of the settings without carrying somebody's channel.
SERVER = config.get("connection.server", "irc.example.net")
PORT = config.get("connection.port", 6667)
CHANNEL = config.get("connection.channel", "#channel")
NICK = config.get("connection.nick", "sloppy")
REALNAME = config.get("connection.realname", "AI Bot")

# llama.cpp OpenAI-compatible endpoint
LLM_BASE_URL = config.get("connection.llm_base_url", "http://localhost:8080/v1")
LLM_API_KEY = "no-key-required"
# The `model` field on each request, and what the status pane calls the model.
# A single-model llama.cpp ignores the field, but sending the alias the server
# actually reports is correct for a multi-model one and, more usefully, is one
# less constant to go stale -- this file and the launch script had both drifted
# off the model actually in service. This value is only the fallback used until
# /props answers; see _probe_props.
LLM_MODEL = "OccultNail"
# The server reports the loaded model's modalities here; `modalities.vision`
# tells us whether a vision model (mmproj loaded) is in service, so the bot can
# auto-detect image support without being told. Same host as the API endpoint.
# Derived from LLM_BASE_URL rather than written out again: the two pointed at
# the same server by convention only, and a moved endpoint would have needed
# editing twice.
LLM_PROPS_URL = LLM_BASE_URL.rsplit("/v1", 1)[0].rstrip("/") + "/props"
# The rolling summarizer talks to the same llama-server. Point it at the URL
# configured above so changing the port here is enough -- summarizer.py keeps
# its own default so it still runs stand-alone.
summarizer.API_URL = f"{LLM_BASE_URL}/chat/completions"
# ...and route its failures into the log pane. Without this the reason a
# summary failed went to stderr, which is invisible under the TUI: the pane
# said a summary had failed and never said why.
summarizer.error_sink = warning

# Reasoning models (Qwen3.x and friends) emit a <think> block before the answer.
# llama.cpp routes that into `reasoning_content`, so a budget too small to cover
# it returns finish_reason="length" with an *empty* `content` — the bot then had
# nothing to say. Ask the server to skip thinking, and keep a budget large enough
# to still produce an answer if a template ignores the switch.
LLM_MAX_TOKENS = _tune("LLM_MAX_TOKENS", "personality.max_tokens", 768)
# A reply carrying this many "nick:" lines is the model writing more transcript
# instead of answering (see _looks_like_transcript). One is left alone --
# addressing somebody by name is ordinary IRC and the persona asks for it.
TRANSCRIPT_NICK_LINES = _tune("TRANSCRIPT_NICK_LINES", "personality.transcript_nick_lines", 2)
# How many words a reply may repeat verbatim from a recent channel line before
# it counts as parroting rather than picking up the subject. See _echoes_recent.
ECHO_RUN_WORDS = _tune("ECHO_RUN_WORDS", "personality.echo_run_words", 6)
# How many attempts a reply gets before the caller's error path takes over.
# Continuing the transcript is a sampling accident, not a stuck state, so a
# second draw almost always lands.
LLM_ATTEMPTS = _tune("LLM_ATTEMPTS", "personality.attempts", 2)

# How many of the most recent channel lines are kept. This is the buffer, not
# the prompt: the LLM call carries only the last CONTEXT_RECENT_LINES of them
# verbatim (the rolling summary covers the rest), while the full window backs
# the mention ordering.
RECENT_LINES = _tune("RECENT_LINES", "memory.recent_lines", 200)
# How often the background worker wakes to check for a summary trigger.
SUMMARIZE_POLL_INTERVAL = 15
# The worker summarizes when at least this many seconds have passed since the
# last summary, OR more than this many lines have arrived since it -- but only
# once at least SUMMARIZE_MIN_LINES lines have accumulated, so a quiet gap or a
# slow trickle never forces a summary. SUMMARIZE_INTERVAL is the age arm; the
# worker polls far more often so the volume arm fires promptly.
SUMMARIZE_INTERVAL = _tune("SUMMARIZE_INTERVAL", "memory.summary_seconds", 600)
SUMMARIZE_VOLUME_LINES = _tune("SUMMARIZE_VOLUME_LINES", "memory.summary_lines", 25)
SUMMARIZE_MIN_LINES = _tune("SUMMARIZE_MIN_LINES", "memory.summary_min_lines", 5)
# Ceiling on the unsummarized-line buffer. Nothing drains it while the bot is
# paused (a paused bot makes no LLM calls), and a server that is down keeps
# handing the lines back to be retried, so without a cap a long pause or a long
# outage would grow it until the eventual request no longer fits the model's
# context. Oldest lines are dropped first: the summary then misses the start of
# the pause, which is the "slightly outdated" this is meant to degrade to.
SUMMARIZE_MAX_PENDING = 200
# How long to wait before retrying after a failed summarizer round-trip. The
# failed lines go back in the buffer, and the age arm is still satisfied, so
# without this the worker would re-attempt on every poll while the server is
# down.
SUMMARIZE_RETRY_AFTER = _tune("SUMMARIZE_RETRY_AFTER", "memory.summary_retry_seconds", 60)
# The model's rolling summary is rejected (and the previous one kept) if it is
# empty or longer than this many characters, so a runaway model response can
# never overwrite the channel's memory.
SUMMARIZE_MAX_CHARS = _tune("SUMMARIZE_MAX_CHARS", "memory.summary_max_chars", 1200)
# How many of the most recent channel lines ride along verbatim in the context
# block. The rolling summary covers everything older; this is the sample the
# model needs to answer what was *just* said.
CONTEXT_RECENT_LINES = _tune("CONTEXT_RECENT_LINES", "memory.context_lines", 20)

# Somebody saying the nick mid-sentence is usually talking to the bot or about
# it, and either is worth a line more often than never. Not a hard trigger: the
# nick is also an ordinary English word, so a certainty here would have the bot
# answering every "that's a sloppy fix". See _resolve_prompt for the tiers.
MENTION_ENABLED = _tune("MENTION_ENABLED", "mentions.enabled", True)
MENTION_REPLY_CHANCE = _tune("MENTION_REPLY_CHANCE", "mentions.reply_chance", 0.3)
# A mention this soon after the bot last spoke to you is not a coin flip: you
# were already talking to it and you just used its name.
MENTION_CERTAIN_WITHIN = _tune(
    "MENTION_CERTAIN_WITHIN", "mentions.certain_within_seconds", 90.0
)

# Long-term recall: the whole channel log kept on disk, searched at reply time
# for the passage that matters. Off by default -- it changes what the model
# reads on every reply, and that is a thing to switch on deliberately and be
# able to switch off again. Capture is NOT gated by it (see _note_for_recall):
# turning recall on against an empty log would mean waiting a fortnight to find
# out whether it was any good.
# What to do at startup when the LLM server is not answering. The bot runs
# perfectly well without one right up until somebody talks to it, at which
# point it says a line about its brain being offline -- in the channel, in
# character, which is a poor place to learn that llama.cpp is not running.
#   ask    warn, and let whoever started it decide (the default)
#   warn   say so and carry on
#   fail   refuse to start
#   off    do not look
LLM_CHECK = _tune("LLM_CHECK", "connection.llm_check", "ask")

# How long the LLM endpoint may stay unreachable before the bot leaves the
# channel, in seconds; 0 switches the behaviour off. It stays connected to the
# server and keeps probing, and rejoins as soon as a model answers again.
# Leaving is the honest signal: a bot sitting in the channel answering every
# question with a brain-offline line looks broken, while an empty chair says
# exactly as much and says it once.
LLM_PART_AFTER = _tune("LLM_PART_AFTER", "connection.llm_part_after_seconds", 300.0)

RECALL_ENABLED = _tune("RECALL_ENABLED", "recall.enabled", False)
RECALL_MAX_LINES = _tune("RECALL_MAX_LINES", "recall.max_lines", 20000)
RECALL_QUERY_LINES = _tune("RECALL_QUERY_LINES", "recall.query_lines", 3)
RECALL_MIN_RELEVANCE = _tune("RECALL_MIN_RELEVANCE", "recall.min_relevance", 0.3)
RECALL_HALF_LIFE_DAYS = _tune("RECALL_HALF_LIFE_DAYS", "recall.half_life_days", 14.0)
RECALL_PASSAGES = _tune("RECALL_PASSAGES", "recall.passages", 3)

# Re-swept for the 35B MoE now in service. The old 1.2 was tuned on a 9B Qwen3.5
# ("1.2 -> 6/9 crude, coherent up to 1.2") and none of that carried over: on this
# model the crude probes barely register at any temperature -- it insults without
# swearing -- so accuracy is the only metric that moved. n=27 replies per step, 9
# of them carrying a checkable fact:
#     0.7 -> 9/9    0.9 -> 9/9    1.0 -> 8/9    1.2 -> 8/9    1.4 -> 6/9
# Reading the replies rather than the counts, 1.2 had also started producing
# duds -- a bare "slopcode joke." as an entire reply, and "the fact that people
# still type when they could be typing" -- where 0.9 and 1.0 stayed sharp ("the
# lack of ircv3 tags means i have to parse timestamps manually and i hate it").
# n is small: read 0.9 vs 1.0 as a coin flip. What the sweep does rule out is
# anything above them.
# Pinned per request rather than inheriting the server's --temp, so the channel
# persona does not shift when the server is retuned for unrelated work.
LLM_TEMPERATURE = _tune("LLM_TEMPERATURE", "sampling.temperature", 1.0)
# Everything else in [sampling], sent as-is with every request. Read as a whole
# section on purpose: a key present in the file is pinned, a key absent is
# inherited from the server's command line, and that is a decision worth being
# able to make per key. Rebuilt on reload like the personas.
SAMPLING: dict[str, Any] = {}
# What [strict_sampling] falls back to when the file does not carry it, so the
# strict modes stay strict without a config. Kept here rather than as a set of
# _tune calls because, like [sampling], it is read as a whole section: a key
# present is pinned, a key absent is inherited.
_STRICT_DEFAULT: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
}
# Overrides applied to the modes that answer about the world (see
# STRICT_MODES), on top of [sampling]. Rebuilt on reload like the personas.
STRICT_SAMPLING: dict[str, Any] = {}
# The "helpful AI assistant / friendly" framing this used to carry was measurably
# re-censoring an already-uncensored model: asked for a filthy joke it returned a
# clean one 10 times out of 12. The persona below is the channel's register, not
# an assistant's. Sent per request, so it affects this bot only -- llama.cpp is
# never told about it and other clients of the same server are unaffected.
LLM_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

# IRC caps a whole protocol line at 512 bytes. When the server relays our
# PRIVMSG it prepends ":nick!user@host " (~100 bytes worst case), so the budget
# below is measured in BYTES over the full "PRIVMSG #chan :...\r\n" line, not in
# characters -- a 450-character reply full of emoji is ~1800 bytes on the wire.
IRC_MAX_LEN = _tune("IRC_MAX_LEN", "personality.max_line_bytes", 400)

# A single question used to fan out into one PRIVMSG per newline (an OSI-model
# answer produced 23 of them). Reflow instead, and never send more than this.
IRC_MAX_REPLY_LINES = _tune("IRC_MAX_REPLY_LINES", "personality.max_reply_lines", 3)

# Two answering modes. Chat is the channel persona; factual is for checking
# claims, where being funny actively gets in the way.
MODE_CHAT = "chat"
MODE_FACTUAL = "factual"
# Directive modes: answer a question seriously and concisely, like the
# fact-checker but WITHOUT a TRUE/FALSE verdict. science/research/answer share
# one persona (see _serious_answer_prompt); they are one-off modes, not moods.
MODE_SCIENCE = "science"
MODE_RESEARCH = "research"
MODE_ANSWER = "answer"
MODE_INTERJECT = "interject"
MODE_VISION = "vision"
# Fetch-and-summarise a link, and translate a piece of text. Both answer
# about something rather than into the room, so neither gets the mention list.
MODE_WEBPAGE = "webpage"
MODE_TRANSLATE = "translate"
# The persona the serious mood answers in. Deliberately not MODE_FACTUAL: that
# one is a fact-checker that opens with a verdict word, which is the wrong shape
# for "what do you reckon about X" asked of a bot that has been told to behave.
MODE_SERIOUS = "serious"
# Two on-demand recitals. They answer about the world rather than into the
# room, take no argument, and are the only modes that supply their own prompt
# when the user gives no words (see _bang_default).
MODE_QUOTE = "quote"
MODE_BUDDHA = "buddha"
MODE_FACTOID = "factoid"
# Modes that get no rolling context at all. A recital is about the world, and
# handing it the channel's last twenty lines made it end a Buddhist teaching
# with "apply this to your four hours of renaming photos". Every other mode
# answers inside an ongoing room and wants the context.
CONTEXTLESS_MODES = frozenset({MODE_QUOTE, MODE_BUDDHA, MODE_FACTOID})

# The modes that answer about the world rather than into the room. A sample
# that wanders is character in chat and a wrong answer here, so these get
# [strict_sampling] on top of [sampling]. Chat, interject, vision, webpage and
# translate keep the channel's own settings: the first three are the persona
# talking, and the last two are already pinned to their source text.
STRICT_MODES = frozenset({
    MODE_FACTUAL, MODE_SCIENCE, MODE_RESEARCH, MODE_ANSWER,
    # A misquote is a wrong answer, not a stylistic choice, and a factoid that
    # is not true is just a lie with a fun delivery.
    MODE_QUOTE, MODE_BUDDHA, MODE_FACTOID,
})

# "factcheck" is unambiguous enough to work without a colon (and always has).
# "science" and "research" are ordinary words, so they need the colon or every
# other sentence in the channel would trigger the bot.
FACTUAL_TRIGGERS = ("factcheck", "science:", "research:")
# The loud form of every directive, so each one can be reached the same way:
# a bang command, exactly like !image and !summarize. Nothing else in a line
# can be mistaken for one, so these need no colon and no nick.
# What to show after each command in the help. A mode with no entry is still
# listed, just without a hint, so a command added below can never go missing
# from !commands -- only under-described.
_COMMAND_ARGS = {
    MODE_TRANSLATE: "<text> [to <lang>]",
    MODE_FACTUAL: "<claim>",
    MODE_SCIENCE: "<question>",
    MODE_RESEARCH: "<question>",
    MODE_ANSWER: "<question>",
    MODE_SERIOUS: "<question>",
    MODE_QUOTE: "[topic]",
    MODE_BUDDHA: "[topic]",
    MODE_FACTOID: "[topic]",
}
# What a bang command that takes no argument asks for when nobody typed one.
# Only these two: every other command is meaningless without its subject.
_BANG_DEFAULTS = {
    MODE_QUOTE: "Give me one historical quote.",
    MODE_BUDDHA: "Give me one teaching.",
    MODE_FACTOID: "Give me one factoid.",
}
# Asking what the bot can do. Answered instantly and never through the model.
HELP_TRIGGERS = ("!commands", "!help", "!cmds")
BANG_COMMANDS = {
    "!translate": MODE_TRANSLATE,
    "!tr": MODE_TRANSLATE,
    "!factcheck": MODE_FACTUAL,
    "!fc": MODE_FACTUAL,
    "!science": MODE_SCIENCE,
    "!research": MODE_RESEARCH,
    "!answer": MODE_ANSWER,
    "!serious": MODE_SERIOUS,
    "!quote": MODE_QUOTE,
    "!buddha": MODE_BUDDHA,
    "!factoid": MODE_FACTOID,
    "!fact": MODE_FACTOID,
}
CHAT_TRIGGERS = ("ai:",)
# Image analysis is on-demand only, so the command trigger needs to be loud
# enough not to fire on an ordinary sentence. "!image" / "!img" / "image:".
IMAGE_TRIGGERS = ("!image", "!img", "image:")
# File extensions the server's stb_image can decode; used to recognise an image
# link in otherwise ordinary chat text.
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "gif", "tga", "bmp")
# What people call a picture when asking about one somebody posted. Only
# "image" used to count, so "what's in the picture probe posted" fell through
# to ordinary chat -- where the model, asked about a picture it was never
# given, says it cannot see images. That reads exactly like the vision model
# being off, which is what made this look like a vision bug.
_PICTURE_DEFAULT = [
    "image", "images", "picture", "pictures", "pic", "pics", "photo",
    "photos", "screenshot", "screenshots", "screengrab", "snap", "meme",
    "gif", "jpg", "jpeg", "png",
]
PICTURE_WORDS = frozenset(_PICTURE_DEFAULT)
# When the request names nobody, or names somebody who has posted nothing,
# fall back to the channel's most recent image -- but only if it is still
# recent. "that picture" means the one everybody can still see in their
# scrollback, not one from last Tuesday.
VISION_FALLBACK_LINES = _tune("VISION_FALLBACK_LINES", "vision.fallback_within_lines", 10)
# Fetch-and-summarise a link. Loud command form, like the image triggers.
SUMMARIZE_TRIGGERS = ("!summarize", "!summarise", "!sum", "!tldr")
# The fuzzy form needs BOTH a word for the thing and a word for the asking,
# with the bot addressed, or ordinary chat about an article would set it off.
# Same shape as the referential image request, which has worked out.
LINK_WORDS = frozenset({
    "link", "links", "article", "page", "url", "post", "site", "story",
    "piece", "writeup", "paper", "blog", "thread",
})
ASK_WORDS = frozenset({
    "what", "whats", "summarize", "summarise", "summary", "tldr", "tl", "gist",
    "about", "says", "say", "read", "explain", "eli5", "point",
})
TRANSLATE_DEFAULT = _tune("TRANSLATE_DEFAULT", "translate.default_language", "English")
# A trailing "to <word>" is only a target language when the word is one we
# recognise. Without that check "translate I want to go to Berlin" would try to
# translate into Berlin. Extend it from [translate].languages rather than here.
# Written as prose rather than ninety quoted strings, which is what SIM905
# would have instead: this is a word list people will edit.
_LANGUAGES_BUILTIN = frozenset("""
    english german deutsch french francais spanish espanol italian italiano
    portuguese brazilian dutch nederlands flemish danish swedish norwegian
    finnish icelandic polish czech slovak slovenian croatian serbian bosnian
    bulgarian romanian hungarian greek turkish russian ukrainian belarusian
    latvian lithuanian estonian albanian macedonian maltese irish welsh gaelic
    basque catalan galician arabic hebrew farsi persian urdu hindi bengali
    punjabi gujarati tamil telugu kannada malayalam marathi nepali sinhala
    thai lao khmer vietnamese indonesian malay tagalog filipino javanese
    chinese mandarin cantonese japanese korean mongolian swahili zulu xhosa
    afrikaans amharic somali hausa yoruba igbo latin esperanto klingon
""".split())  # noqa: SIM905 - a word list people edit reads better as prose
WEB_ENABLED = _tune("WEB_ENABLED", "web.enabled", True)
WEB_TIMEOUT = _tune("WEB_TIMEOUT", "web.timeout_seconds", 15)
WEB_MAX_BYTES = _tune("WEB_MAX_BYTES", "web.max_bytes", 2_000_000)
WEB_MAX_REDIRECTS = _tune("WEB_MAX_REDIRECTS", "web.max_redirects", 5)
WEB_MAX_ARTICLE_CHARS = _tune("WEB_MAX_ARTICLE_CHARS", "web.max_article_chars", 24_000)
WEB_ADD_COMMENT = _tune("WEB_ADD_COMMENT", "web.add_comment", True)
WEB_MAX_REPLY_LINES = _tune("WEB_MAX_REPLY_LINES", "web.max_reply_lines", 4)
WEB_CACHE_SIZE = _tune("WEB_CACHE_SIZE", "web.cache_size", 32)

# A greeting in front of the nick is still the bot being addressed: "hey
# Heretic.. whats up" is no less directed at it than "Heretic: whats up". Only
# these lead-ins are skipped -- any other word before the nick is the channel
# talking *about* the bot rather than to it.
# A mention of the nick in the middle of a sentence. Not a hard trigger -- see
# _resolve_prompt -- because "that was a sloppy fix" is somebody talking, not
# somebody asking. The nick has to stand as its own word.
_MENTION_RE = re.compile(rf"(?<!\w){re.escape(NICK)}(?!\w)", re.IGNORECASE)

# Who is allowed to talk to the bot privately, as IRC hostmasks. Empty means
# nobody, which is the safe default: a private message is a channel of one that
# nobody else can see, so it is the wrong place to take instructions from
# strangers. Matched against the full nick!user@host, because a nick on its own
# is whoever grabbed it while the real owner was disconnected.
OWNER_MASKS: list = []

# Who the bot pretends is not there, as nicks or full IRC hostmasks. A line
# from one of these is dropped whole: not answered, not remembered, not
# counted, not greeted. Same glob matching as OWNER_MASKS, except that a
# pattern naming no user or host ("spammer") means that nick from anywhere.
# Owners are never ignored -- see _is_ignored.
IGNORE_MASKS: list = []
# Masks added at runtime with !ignore, kept apart from the configured ones so
# !unignore can take back what it put there without pretending it can edit the
# config file. Persisted, because an ignore that quietly lapses on the next
# restart is worse than none: nobody watches for it coming back.
_ignored_live: list = []

ADDRESS_LEAD_INS = frozenset({
    "hey", "hi", "hello", "yo", "oi", "ok", "okay", "so", "well", "psst",
    "sup", "ay", "aye", "eh", "um", "uh", "right", "anyway", "also", "but",
})

# Thread-safe storage for captured AI prompts
_prompt_lock = threading.Lock()
_pending = {"prompt": "", "sender": "", "stop": False, "mode": MODE_CHAT,
            # Where the answer goes: the channel, or a nick for a private one.
            "reply_to": ""}
# A separate queue for on-demand image-analysis requests (see _answer_vision).
_pending_vision = {"url": "", "sender": "", "prompt": "", "reply_to": ""}
# People waiting to be greeted, as (nick, kind, flavour). Filled from the
# receiver thread and drained by the poll loop, because generating a greeting
# is an LLM call and the receiver must never block on one.
_pending_greetings: list[tuple[str, str, str]] = []
# A queued !summarize, as (url, sender). One at a time, like the image queue:
# fetching and summarising is two LLM calls and a download, and the channel
# should not be made to sit through a backlog of them.
_pending_page = {"url": "", "sender": "", "reply_to": ""}
# The most recent NON-image link each nick posted, and the channel's last, so
# "what's in the link probe just posted" can resolve one. Mirrors
# _recent_images, which does the same job for pictures.
_recent_links = {"by_nick": {}, "global": None}
# url -> (title, text) for pages already fetched, so the same link posted twice
# is not fetched twice. Insertion-ordered and trimmed to WEB_CACHE_SIZE.
_page_cache: dict[str, tuple[str, str]] = {}
# Whether the running model can see images. Auto-detected by probing the
# server's /props (modalities.vision); the TUI can force it on/off via a manual
# override, which wins over the probe result. None => follow the probe.
_vision = {"enabled": False, "override": None}

# Someone who has just been answered stays "in conversation" for a short window,
# during which anything they say counts as addressed to the bot even without a
# trigger. The window is refreshed each time the bot replies to them.
FOLLOWUP_WINDOW = _tune("FOLLOWUP_WINDOW", "chatter.followup_window", 40.0)
SHUTUP_REPLY = "Fine i'll shut up"

# If the channel talks this many lines without addressing the bot, it chimes in
# unprompted: half the time reacting to whatever was last said, half the time
# just being asked for something funny.
IDLE_INTERJECT_AFTER = _tune("IDLE_INTERJECT_AFTER", "chatter.interject_after_lines", 20)
# What the bot is asked when it butts in off the back of the conversation. An
# instruction rather than the line itself -- see _queue_interjection.
REACT_PROMPT = "React to what the channel is talking about right now."
IDLE_PROMPT = "say something funny please! Maybe involve one of the channel user's names"
# Asking for a joke while the persona has been told not to make any produces a
# bad line either way, so the serious mood opens with something it can deliver.
SERIOUS_IDLE_PROMPT = "say something interesting please!"
# Even split between reacting to the last line and just asking for a joke.
IDLE_REACT_CHANCE = _tune("IDLE_REACT_CHANCE", "chatter.react_chance", 0.5)

# A channel silent this long gets a line out of nowhere, after which anyone may
# talk to the bot untriggered for a short window -- capped, so a busy room
# cannot turn the whole minute into a wall of bot.
SILENCE_TIMEOUT = _tune("SILENCE_TIMEOUT", "chatter.silence_seconds", 30 * 60)
OPEN_FLOOR_WINDOW = _tune("OPEN_FLOOR_WINDOW", "chatter.open_floor_seconds", 60.0)
OPEN_FLOOR_MAX_PROMPTS = _tune("OPEN_FLOOR_MAX_PROMPTS", "chatter.open_floor_max_prompts", 8)
# Greet a newcomer on JOIN, and welcome back anyone who speaks up after a long
# silence. The greeting is always sent; a mild roast rides along about half the
# time. A roast is deliberately mild -- this is a welcome, not a vendetta.
IDLE_GREET_AFTER = _tune("IDLE_GREET_AFTER", "greetings.idle_seconds_before_welcome_back", 9360)
GREET_ROAST_CHANCE = 0.5         # probability a fallback greeting gets a roast
# The share of eligible arrivals greeted at all. Greeting every join and every
# reappearance was more bot than the channel wanted.
GREET_CHANCE = _tune("GREET_CHANCE", "greetings.chance", 0.5)
# One unprompted line per this many seconds. Interjections, greetings, silence
# breaks and follow-ups all wait for it; being addressed by name does not.
CHATTER_MIN_INTERVAL = _tune("CHATTER_MIN_INTERVAL", "chatter.min_seconds_between_lines", 120)
# Never speak unprompted when the bot said the last thing in the channel. A run
# of bot lines is almost always several triggers landing together.
SKIP_WHEN_BOT_SPOKE_LAST = _tune("SKIP_WHEN_BOT_SPOKE_LAST", "chatter.skip_when_bot_spoke_last", True)
# Replies granted inside one follow-up window, so a conversation does not turn
# into the bot answering every line somebody types.
FOLLOWUP_MAX_REPLIES = _tune("FOLLOWUP_MAX_REPLIES", "chatter.followup_max_replies", 1)
# The three shapes a greeting takes, drawn evenly: a roast built from what the
# person has actually said, a plain hello, or a hello wrapped around an odd
# question. Five canned lines got repetitive within a session.
GREET_FLAVOURS = ("roast", "casual", "question")
# How many of that person's stored lines the roast flavour is given to work
# with. Enough to find something specific, not so much that the greeting turns
# into a summary of them.
GREET_PROFILE_LINES = _tune("GREET_PROFILE_LINES", "greetings.profile_lines", 8)
# The same store, for a reply to somebody ASKING about a person rather than a
# greeting aimed at one. Fewer lines, because this rides on ordinary replies
# and a greeting happens once: enough to be specific, not a biography. The nick
# cap stops "sloppy who is worse, alice bob or carol" pulling three profiles.
ABOUT_PROFILE_LINES = _tune("ABOUT_PROFILE_LINES", "memory.about_lines", 6)
ABOUT_PROFILE_NICKS = _tune("ABOUT_PROFILE_NICKS", "memory.about_nicks", 2)
# Below this, a nick is too likely to be an ordinary word to treat a match as
# somebody being asked about.
ABOUT_MIN_NICK_CHARS = _tune("ABOUT_MIN_NICK_CHARS", "memory.about_min_nick", 3)
# People waiting to be greeted. A burst of joins should not become a queue of
# LLM calls the channel has to sit through.
GREET_QUEUE_MAX = _tune("GREET_QUEUE_MAX", "greetings.queue_max", 3)
# Skip the join greeting if they left fewer than this many chatlines ago.
GREET_REJOIN_CHATLINES = _tune("GREET_REJOIN_CHATLINES", "greetings.skip_if_left_within_lines", 5)
# Lines shorter than this many characters, or a single word only, are treated
# as noise: not stored in the LLM's recent-history buffer or handed to the
# summarizer (see _note_recent).
MIN_CHAT_CHARS = _tune("MIN_CHAT_CHARS", "memory.min_chat_chars", 7)
# Profiles keep a much lower bar, and no single-word rule at all -- see
# _too_short_for_profile. A profile is partly a record of presence, and "yeah"
# from Probe is still Probe being in the room, where in a channel summary the
# same line is pure noise. At this bar "lol" (three characters) is still just
# under; drop it to 3 to catch that too.
MIN_PROFILE_CHARS = _tune("MIN_PROFILE_CHARS", "memory.min_profile_chars", 4)
# The auto-interject opener waits this long after JOIN so the userlist (and the
# recent channel lines) have time to arrive before the first LLM call.
JOIN_GRACE_PERIOD = _tune("JOIN_GRACE_PERIOD", "chatter.join_grace_seconds", 10.0)
# How long the poll loop sleeps between passes over the pending work.
POLL_INTERVAL = 2
# How long to wait for the server's 001 Welcome before giving up on a
# connection and reconnecting.
REGISTER_TIMEOUT = 30
# Reconnect backoff. A lost link is retried after RECONNECT_MIN_DELAY and the
# wait doubles up to RECONNECT_MAX_DELAY, so a server that is down, netsplit or
# refusing us is not hammered. A connection that registers and joins resets it.
RECONNECT_MIN_DELAY = 10
RECONNECT_MAX_DELAY = 300
# How often the background worker writes the profile store to disk, when
# anything has changed. Debounced rather than written per line: the receiver
# thread must never wait on a disk write.
PROFILE_SAVE_INTERVAL = 60
# /props answers what is loaded -- whether it can see images, and under what
# alias. Both change only when the server is restarted, so it is asked once a
# minute rather than on every poll pass; it was one HTTP round-trip every two
# seconds.
PROPS_PROBE_INTERVAL = 60
# What the channel is told on the way out when the LLM has gone. In character,
# like every other line the channel sees, and the only explanation anybody gets
# -- so it says what is wrong and that it is coming back.
PART_REASON = "brain offline, back when the model is"
_activity = {"at": 0.0}
# The time the bot joined, so the auto-interject opener can wait
# JOIN_GRACE_PERIOD seconds before it talks (see _within_join_grace). Kept in a
# container -- like _activity -- so main() can record the join without a global
# statement (ruff PLW0603).
_joined = {"at": 0.0}
_open_floor = {"deadline": 0.0, "used": 0}
_chatter = {"count": 0, "last": ""}
# The last 200 channel lines spoken, injected into the LLM call as real chat
# history (see _context_block). A plain parallel buffer keeps the senders in
# lock-step so the mention list can favour recent speakers, not members at
# random. Both stay oldest-first.
_recent_lines = collections.deque(maxlen=RECENT_LINES)
# The nick that spoke each of those lines, in lock-step with _recent_lines, so
# the mention list can favour recent speakers instead of naming members at
# random.
_recent_senders = collections.deque(maxlen=RECENT_LINES)
# Wall-clock time each of those lines arrived, so the context block can stamp
# them. Kept beside the text rather than in it: _echoes_recent compares against
# what was actually said, not against a rendering of it.
_recent_times = collections.deque(maxlen=RECENT_LINES)
# Rolling summarizer state (guarded by _prompt_lock): the IRC lines that have
# arrived since the last successful summary, plus the running summary +
# highlights fed into chat prompts. A plain list + the shared _prompt_lock --
# the background worker snapshots and clears it each interval; never a deque.
_pending_summary_lines: list[str] = []
# The running summary + highlights fed into chat prompts, carried forward by
# the summarizer worker. Kept in a container to avoid a global reassignment
# (ruff PLW0603), consistent with the other shared state.
# "at" is wall-clock seconds (time.time, not monotonic) of the last update, so
# a summary read back off disk after a restart can be labelled with its real
# age. 0.0 means there has never been one.
_rolling = {"summary": "", "highlights": [], "at": 0.0}
# Monotonic time of the last successful summary, for the status pane.
_last_summary_at = {"t": 0.0}
# Monotonic time before which no summarizer retry is attempted, set after a
# failed round-trip so a dead server is not hammered once per poll.
_summary_retry_at = {"t": 0.0}
# Monotonic time of the last /props vision probe, so it runs once a minute
# rather than on every poll pass.
_last_props_probe = {"t": 0.0}
# Whether the last /props probe was answered, and since when it has not been.
# The same probe that detects vision is the health signal: it asks the endpoint
# the bot will actually call, so a pass here means more than an open port.
# "down_since" is monotonic and 0.0 while the server is answering.
_llm_health = {"ok": True, "down_since": 0.0}
# Whether the bot has left the channel to sit out an LLM outage. Distinct from
# never having joined: the link is up and the poll loop is running, and it is
# waiting for a model rather than for a server.
_absent = {"on": False}
# The model the server says it has loaded. Seeded with the configured fallback;
# `detected` stays False until a probe has actually answered, so the status pane
# can distinguish "this is what is loaded" from "this is what we would ask for".
_model = {"alias": LLM_MODEL, "detected": False}
# What the bot remembers about individual chatters, across sessions. Guarded by
# _prompt_lock like the rest of the shared state -- the store does no locking of
# its own (see profiles.py). _profiles_dirty says whether anything has changed
# since the last write, so an idle channel does not rewrite the file every
# minute.
_profile_store = profiles.ProfileStore()
_profile_path = profiles.default_path()
# The channel's own memory, beside the profiles. Bumped only when the shape on
# disk changes in a way an older file cannot be read into.
MEMORY_VERSION = 1
# Same for the runtime ignore list, which is its own small file rather than a
# field in the memory: it is configuration the owner set, not something the
# channel said, and losing one should never mean losing the other.
IGNORES_VERSION = 1
# Whether the rolling state has changed since it was last written. Shutdown is
# idempotent, so a second call must have nothing left to write.
_memory_dirty = {"on": False}
_recall_store = recall.RecallStore(RECALL_MAX_LINES)
# None means "follow the config"; True/False force it from the TUI.
_recall = {"override": None}


# Derived from _profile_path when they are used, not stored at import. The
# suite redirects _profile_path to a temporary directory so it cannot write
# over a live channel's state; deriving these two the same way means the next
# store added here is covered by that guard automatically, instead of quietly
# writing to the real one until somebody notices the bot quoting "alice".
def _memory_path() -> pathlib.Path:
    """Where the rolling summary lives: beside the profile store."""
    return _profile_path.with_name("memory.json")


def _recall_path() -> pathlib.Path:
    """Where the channel log lives: beside the profile store."""
    return recall.default_path(_profile_path)


def _ignores_path() -> pathlib.Path:
    """Where the runtime ignore list lives: beside the profile store."""
    return _profile_path.with_name("ignores.json")
_profiles_dirty = {"on": False}
_profiles_saved_at = {"t": 0.0}
# The most recent image URL each nick (and the channel overall) has posted, so a
# "what's in the image Tim just posted" request can resolve the link. Kept in a
# container to avoid a global statement (ruff PLW0603).
_recent_images = {"by_nick": {}, "global": None}
# How many chatlines (non-bot PRIVMSGs) have been spoken, so a returning user
# can be greeted on JOIN only if they have not popped back within a few lines.
_chatlines = {"count": 0}
# When we last heard from each nick (monotonic seconds), so a long silence can
# be welcomed back. Reset to now on JOIN so a rejoin does not also read as idle.
_last_seen: dict[str, float] = {}
# The chatline count at which each nick last left (QUIT/PART), so the JOIN
# greeting can be skipped for a frequent pop-in.
_left_at: dict[str, int] = {}

# `budget` is how many untriggered follow-ups are still allowed in this
# window; it is refilled only by a real trigger, not by the bot replying.
_conversation = {"nick": "", "deadline": 0.0, "budget": 0, "at": 0.0}
# When the bot last said something in the channel, and whether it has the last
# word right now. Both are set in send(), which is the one place every line to
# the channel goes through.
_speech = {"at": 0.0, "bot_last": False}
# Set while the poll loop is mid-reply, so the status pane can show the bot
# as busy. Guarded with _prompt_lock like the other state.
_busy = {"on": False}
# Set while the TUI has paused the bot (press 'P'): while on, the poll loop
# makes no LLM calls and no greetings, so the bot stays silent until 'P' is
# pressed again to unpause. Not a reply-mode flag -- it silences every call.
_paused = {"on": False}
# Signalled by the TUI so main() can stop its poll loop and close the socket.
_stop_event = threading.Event()
# The full record of the most recent LLM call, assembled as the call happens
# so the TUI can show it on demand (press 'd' in the UI). Guarded with
# _prompt_lock like the other state.
_last_llm_call = {"text": ""}

# The channel members, read off the userlist after joining so the chat persona
# can talk at individuals rather than a faceless room. The bot's own nick is
# never recorded.
_users = {"names": []}

# Moods, not per-reply modes: whichever one the channel asks for sticks for
# every answer until somebody names another. The resting mood never expires;
# the others lapse back to it on their own, since a room that wanted the
# sensible version a quarter of an hour ago has usually moved on.
#
# They are read from sloppy.toml, so adding one is a file edit: give it words
# to answer to, an acknowledgement, and the name of a persona. An empty persona
# means the mood leaves whatever the message itself asked for alone, which is
# what the resting mood does.
_MOOD_DEFAULTS = {
    "banter": {"words": ["banter"], "reply": "Oh you want bants huh? Fine",
               "persona": ""},
    "serious": {"words": ["serious"], "reply": "Ok I'll be serious for a while",
                "persona": "serious"},
    # NOT the fact-checker: a mood answers every line in its persona, and one
    # that opens with a verdict word turns "my new gpu arrived" into "FALSE.".
    # !factcheck is the path for a claim somebody wants checked.
    "neutral": {"words": ["neutral", "be neutral"],
                "reply": "alright, straight answers for a bit",
                "persona": "neutral"},
}
_MOODS: dict[str, dict] = {}
# The length of the window the scheduled moods are laid out inside. An hour, so
# "minutes_per_hour" in the config means what it says.
MOOD_WINDOW = _tune("MOOD_WINDOW", "mood_matching.window_seconds", 3600.0)
# The plan for the current window: when it started, and the slots in it as
# (start, end, mood). Re-planned when the window rolls over, so the moments are
# different every hour rather than on a fixed timetable.
_mood_plan: dict = {"window": -1.0, "slots": []}
MOOD_BANTER = _tune("MOOD_BANTER", "mood_matching.resting", "banter")
MOOD_TIMEOUT = _tune("MOOD_TIMEOUT", "mood_matching.timeout_seconds", 15 * 60)

# What each mood answers to: every word any mood claims, mapped to that mood.
MOOD_WORDS: dict[str, str] = {}
# Multi-word triggers, as tuples of words -> mood. A mood word that is also an
# ordinary English word ("nice", "kind", "mean") cannot be a bare trigger: the
# channel says it to each other all day, and the bot switching mood every time
# somebody approves of something is not a feature. Phrasing it as "be nice"
# costs the user nothing and cannot be said by accident.
MOOD_PHRASES: dict[tuple, str] = {}
# The persona a mood answers in. The resting mood is absent on purpose: it
# leaves whatever the message itself asked for alone.
MOOD_MODES: dict[str, str] = {}
# Seconds per window each mood takes over by itself, from minutes_per_hour.
MOOD_BUDGETS: dict[str, float] = {}
# The temperature a mood answers at, for the moods that name one. Absent means
# the mood does not care and the ordinary temperature applies.
MOOD_TEMPERATURES: dict[str, float] = {}
MOOD_REPLIES: dict[str, str] = {}

# Words that may pad a mood command without changing what it asks for, so
# "sloppy, be serious for once" lands the same as "sloppy: serious". The list
# is deliberately short: "are you serious", "is it serious" and "stop being
# serious" all have to stay ordinary chat, so their words are not in it.
MOOD_FILLER_WORDS: frozenset[str] = frozenset()
_FILLER_DEFAULT = ([
    "a", "be", "being", "bit", "for", "get", "go", "in", "into", "just",
    "let", "lets", "mode", "more", "much", "now", "of", "on", "once",
    "please", "pls", "switch", "the", "time", "to", "turn", "up", "us",
])


def _rebuild_mood_triggers() -> None:
    """Sort the configured mood words into single words and phrases.

    Split out of _rebuild_moods to keep it under the complexity ceiling; the
    two tables are one idea and read better together anyway.
    """
    MOOD_WORDS.clear()
    MOOD_PHRASES.clear()
    for name, spec in _MOODS.items():
        for word in spec.get("words", [name]):
            parts = tuple(re.findall(r"[a-z]+", str(word).lower()))
            if len(parts) == 1:
                MOOD_WORDS[parts[0]] = name
            elif parts:
                MOOD_PHRASES[parts] = name


def _rebuild_moods() -> None:
    """Rebuild everything derived from [moods]. Called by _rebuild_from_config.

    Split out to keep that function under the complexity ceiling; the moods are
    the bulk of it, and they read as one thing.
    """
    _MOODS.clear()
    _MOODS.update(config.section("moods") or _MOOD_DEFAULTS)
    _rebuild_mood_triggers()
    MOOD_BUDGETS.clear()
    MOOD_BUDGETS.update({
        name: float(spec.get("minutes_per_hour", 0) or 0) * 60
        for name, spec in _MOODS.items()
        if name != MOOD_BANTER and float(spec.get("minutes_per_hour", 0) or 0) > 0
    })
    # A new plan on the next check: the budgets it was built from have changed.
    _mood_plan["window"] = -1.0

    MOOD_TEMPERATURES.clear()
    MOOD_TEMPERATURES.update({
        name: float(spec["temperature"])
        for name, spec in _MOODS.items()
        if isinstance(spec.get("temperature"), int | float)
        and not isinstance(spec.get("temperature"), bool)
    })

    MOOD_MODES.clear()
    MOOD_MODES.update({
        name: spec["persona"]
        for name, spec in _MOODS.items()
        if spec.get("persona")
    })
    MOOD_REPLIES.clear()
    MOOD_REPLIES.update({
        name: spec.get("reply", f"{name} it is") for name, spec in _MOODS.items()
    })


def _rebuild_from_config() -> None:
    """Re-derive everything that is computed from the config rather than read.

    The plain levers are just values and reload_config reassigns them; these
    are tables built out of several keys, so they get rebuilt here. Called at
    import and again on every reload, so a lever behaves the same whether it
    was set before startup or a minute ago.
    """
    globals()["_IDENTITY"] = config.get(
        "personas.identity", _IDENTITY_DEFAULT
    ).format(nick=NICK, channel=CHANNEL)
    PERSONAS.clear()
    PERSONAS.update(config.section("personas"))

    # temperature is passed as its own argument by the client, so it must not
    # also ride along in the body.
    SAMPLING.clear()
    SAMPLING.update({
        k: v for k, v in config.section("sampling").items() if k != "temperature"
    })
    STRICT_SAMPLING.clear()
    STRICT_SAMPLING.update(config.section("strict_sampling") or _STRICT_DEFAULT)

    globals()["PICTURE_WORDS"] = frozenset(
        str(w).lower() for w in config.get("vision.picture_words", _PICTURE_DEFAULT)
    )

    OWNER_MASKS.clear()
    OWNER_MASKS.extend(
        str(m) for m in config.get("owners.masks", []) if isinstance(m, str)
    )

    IGNORE_MASKS.clear()
    IGNORE_MASKS.extend(
        str(m) for m in config.get("ignore.masks", []) if isinstance(m, str)
    )

    _rebuild_moods()

    globals()["MOOD_FILLER_WORDS"] = frozenset(
        config.get("mood_matching.filler_words", _FILLER_DEFAULT)
    )


def _resize_recent_buffers() -> None:
    """Rebuild the recent-line deques when their configured size changed.

    A deque's maxlen is fixed at construction, so a new value only takes effect
    if the buffer is rebuilt. Contents are carried over, newest kept.
    """
    global _recent_lines, _recent_senders, _recent_times  # noqa: PLW0603 - deque maxlen is immutable
    with _prompt_lock:
        if _recent_lines.maxlen == RECENT_LINES:
            return
        _recent_lines = collections.deque(_recent_lines, maxlen=RECENT_LINES)
        _recent_senders = collections.deque(_recent_senders, maxlen=RECENT_LINES)
        _recent_times = collections.deque(_recent_times, maxlen=RECENT_LINES)


def reload_config() -> list[str]:
    """Re-read sloppy.toml and apply it live. Returns anything wrong with it.

    Everything reloaded here is safe to change while running: the numeric
    levers are read where they are used, and the personas and moods are rebuilt
    here. [connection] is the exception and is deliberately NOT reloaded --
    server, channel and nick need a reconnect, so re-reading them would be a
    lever that silently fails to take effect.
    """
    unreadable = bool(config.load())
    globals().update({
        name: config.get(key, default)
        for name, (key, default) in _TUNABLES.items()
    })
    _rebuild_from_config()
    _rebuild_directives()
    _resize_recent_buffers()
    _recall_store.max_lines = RECALL_MAX_LINES
    if unreadable:
        # The file did not parse, so nothing downstream exists. Every persona
        # and mood is "missing" as a consequence, and listing each one would
        # bury the single line worth reading.
        return config.problems() + [
            "nothing from the file is in effect: every lever is at its default "
            "and the persona is the short built-in one"
        ]
    return config.problems() + _persona_problems() + _mood_problems()


def _persona_problems() -> list[str]:
    """Personas whose placeholders will not expand.

    _fill falls back to the raw text at call time, but that is mid-conversation
    and possibly hours after the edit that caused it. Checking on load means
    the message arrives while the file is still open in front of you.
    """
    found = []
    for name, text in PERSONAS.items():
        try:
            text.format(identity=_IDENTITY, nick=NICK, channel=CHANNEL)
        except (KeyError, IndexError, ValueError) as exc:
            found.append(f"persona {name!r} has an unknown placeholder: {exc}")
    return found


def _mood_problems() -> list[str]:
    """Anything wrong with the configured moods, for main() to surface.

    A mood pointing at a persona that does not exist would silently answer in
    the chat voice, which is exactly the kind of quiet wrongness a hand-edited
    file produces.
    """
    found = []
    if MOOD_BANTER not in _MOODS:
        found.append(f"resting mood {MOOD_BANTER!r} is not one of {sorted(_MOODS)}")
    for name, persona in MOOD_MODES.items():
        if persona not in PERSONAS and persona not in _PERSONA_FOR_MODE:
            found.append(f"mood {name!r} wants persona {persona!r}, which is not defined")
    return found


def _random_mood() -> str:
    """The mood to boot into: banter, and the channel can override it.

    Banter is the resting state and never expires, so the bot opens as itself
    rather than a mode someone did not ask for.
    """
    return MOOD_BANTER


_mood = {"name": _random_mood(), "at": time.monotonic()}

# Signaled when the server has completed registration (001 Welcome received)
_registered = threading.Event()

# LLM client (created once, shared)
_llm_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)


class EmptyLLMReply(RuntimeError):
    """The model returned no answer text (e.g. truncated inside a think block)."""


class TranscriptReply(RuntimeError):
    """The model wrote more chat transcript instead of replying to the room."""


# CR, LF and NUL end an IRC line (or a C string). Anything carrying one into
# send() would let the rest of it be read by the server as a command of its
# own -- the classic way a bot is made to JOIN, PART or op somebody on behalf
# of whoever got their text into it.
_UNSAFE_IN_LINE = str.maketrans({"\r": " ", "\n": " ", "\0": ""})


def send(sock: socket.socket, line: str) -> None:
    """Write one line to the server, and note it if the channel heard it.

    Every line the bot says goes through here, so this is where "have I just
    spoken" is recorded -- one place rather than at each of the half-dozen
    callers that can produce a PRIVMSG. It is also the one place worth
    sanitising: a line separator reaching the socket is command injection, and
    no caller upstream can be relied on to have thought about it.

    Nothing known reaches here with one today -- the reply reflow drops them,
    and an inbound IRC line cannot contain one because the receiver splits on
    it -- so a substitution here that fires is a bug or an attack, and says so
    in the log rather than passing quietly.
    """
    safe = line.translate(_UNSAFE_IN_LINE)
    if safe != line:
        warning(f"[IRC] stripped a line separator from an outgoing line: {line!r}")
        line = safe
    sock.send((line + "\r\n").encode("utf-8"))
    irc(f"> {line}")
    if line.startswith(f"PRIVMSG {CHANNEL} :"):
        with _prompt_lock:
            _speech["at"] = time.monotonic()
            _speech["bot_last"] = True


class Request(NamedTuple):
    """Who asked, where the answer goes, and what they are allowed to ask for.

    These four travel together through every handler; passing them as one
    keeps the signatures honest about being one fact rather than four.
    """

    sender: str
    reply_to: str
    private: bool = False
    owner: bool = False


class Privmsg(NamedTuple):
    """One PRIVMSG, pulled apart.

    `mask` is the full nick!user@host the server attached, which is what owner
    masks are matched against -- the nick alone is trivially taken by anybody
    when the real one is disconnected. `target` is the channel it was said in,
    or the bot's own nick when it was said privately.
    """

    mask: str
    sender: str
    target: str
    text: str

    @property
    def private(self) -> bool:
        """True when this was said to the bot directly rather than in a room."""
        return not self.target.startswith(("#", "&", "!", "+"))


def _parse_privmsg(line: str) -> Privmsg | None:
    """Pull a PRIVMSG apart, or None if `line` is not one.

    Format: :nick!user@host PRIVMSG <target> :message
    """
    if " PRIVMSG " not in line:
        return None
    prefix, rest = line.split(" PRIVMSG ", 1)
    mask = prefix.lstrip(":").strip()
    sender = mask.split("!")[0]
    target, _, text = rest.partition(" :")
    if not _:
        # No " :" separator: either ":message" with no target, or a bare
        # message. Neither is a room, so it is treated as private.
        text = rest[1:] if rest.startswith(":") else rest
        target = NICK
    return Privmsg(mask, sender, target.strip(), text)


def _is_owner(mask: str) -> bool:
    """True when `mask` matches one of the configured owner hostmasks.

    Glob matching, as every IRC client and server does it, and case-folded:
    nicks and hostnames are both case-insensitive, and an owner who reconnects
    with different casing is still the owner.
    """
    lowered = mask.lower()
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in OWNER_MASKS)


def _expand_mask(pattern: str) -> str:
    """A pattern naming neither user nor host means that nick from anywhere.

    "spammer" -> "spammer!*@*", while "spammer!*@*.example.net" is left alone.
    Typing a bare nick is what an owner reaches for mid-abuse, and the whole
    mask is what actually identifies somebody, so both are accepted and only
    one of them needs explaining.
    """
    return pattern if ("!" in pattern or "@" in pattern) else f"{pattern}!*@*"


def _is_ignored(mask: str) -> bool:
    """True when `mask` is somebody the bot should behave as if absent.

    Owners are never ignored: an owner is who fixes this, and a mistyped
    !ignore that locks out the only person who can undo it is a worse failure
    than an owner who has to say so twice.
    """
    if _is_owner(mask):
        return False
    lowered = mask.lower()
    with _prompt_lock:
        patterns = IGNORE_MASKS + _ignored_live
    return any(fnmatch.fnmatch(lowered, _expand_mask(p).lower())
               for p in patterns)


def _names_an_owner(target: str) -> bool:
    """Whether `target` (a nick or mask an owner typed) names an owner.

    Globs on both sides -- the target may be a pattern and every owner mask
    is one -- so this is a "could it be" rather than a certainty. It is used
    to refuse an !ignore that would do nothing, so erring towards refusing is
    the right way round.
    """
    expanded = _expand_mask(target).lower()
    return any(fnmatch.fnmatch(expanded, owner.lower())
               or fnmatch.fnmatch(owner.lower(), expanded)
               for owner in OWNER_MASKS)


def _event_mask(line: str) -> str:
    """The nick!user@host an IRC line came from, or "" when it has no prefix.

    Servers put their own name here on numerics, which has no "!" and so only
    matches a pattern naming it -- nobody ignores their own server by accident.
    """
    if not line.startswith(":") or " " not in line:
        return ""
    return line.split(" ", 1)[0].lstrip(":").strip()


def _parse_who_reply(line: str) -> str | None:
    """Return the nick from a 352 RPL_WHO line, else None.

    Format: :server 352 client #channel user host server nick (H) :0 realname.
    The nick is four fields after the channel; the realname is left over.
    """
    parts = line.split()
    for i, part in enumerate(parts):
        if part.startswith("#"):
            return _strip_status(parts[i + 4]) if i + 4 < len(parts) else None
    return None


STATUS_PREFIXES = "+&@%"


def _strip_status(nick: str) -> str:
    """Strip IRC status prefixes (+, &, @, %) from a nick; they are not part of it."""
    return nick.lstrip(STATUS_PREFIXES)


def _parse_name_reply(line: str) -> list:
    """Return the nicks from a 353 RPL_NAMREPLY line, stripping status prefixes.

    Format: :server 353 client #channel :@hop +voice nick1 nick2. Everything
    after the first " :" is a space-separated user list.
    """
    marker = line.find(" :")
    if marker == -1:
        return []
    nicks = []
    for raw_token in line[marker + 2:].split():
        nick = _strip_status(raw_token)
        if nick:
            nicks.append(nick)
    return nicks


def _register_from_userlist_line(line: str) -> bool:
    """Record members from a 352 (WHO) or 353 (NAMREPLY) reply.

    Returns True when the line was one of those so the caller need not log it.
    """
    if " 352 " in line:
        nick = _parse_who_reply(line)
        if nick:
            _register_user(nick)
            return True
    if " 353 " in line:
        for nick in _parse_name_reply(line):
            _register_user(nick)
        irc(f"Userlist:\n{', '.join(_channel_users())}")
        return True
    return False


def _handle_info_line(line: str) -> bool:
    """Handle a server information line: the 001 welcome or a userlist reply.

    Returns True when the line was one of those, so the receiver need not log
    it as ordinary chatter.
    """
    # ":<server> 001 <nick> :Welcome" -- matched on the numeric, not on the
    # server's name, so pointing SERVER somewhere else still registers.
    parts = line.split()
    if line.startswith(":") and len(parts) > 1 and parts[1] == "001":
        irc(f"< {line}")
        _registered.set()
        return True
    return _register_from_userlist_line(line)


# The bot's identity line and every persona live in sloppy.toml, so the voice
# can be tuned and new moods added without touching code. Each entry keeps its
# current text as the default here, so a missing or edited-out config still
# runs. {identity} is expanded first, then {nick} and {channel} -- the
# fact-checker persona leaves {identity} out on purpose, because a fact-checker
# with a persona is one that shades answers.
#
# The nick is introduced AS a nick and never after "you are": "You are sloppy"
# is a grammatical English sentence about careless work, and with the name
# lowercase and every other trait in the prompt written as "You are
# <adjective>", the model read it as one and typed accordingly.
_IDENTITY_DEFAULT = "Your nick is {nick} and you are a regular in the IRC channel {channel}. "
_IDENTITY = ""

# Persona name -> text, whatever the file defines. Read as a whole section
# rather than by a fixed list of names, so a mood naming a brand new persona
# works with no code change -- which is the point of moving them out here.
#
# Unlike the numeric levers there is no full copy of the prose in code. Three
# and a half kilobytes of document duplicated in two places is a document that
# drifts, and a constant drifting away from reality has already cost this
# project twice in a day. Without a config file the bot still runs and still
# sounds like itself, on the short fallback below, and says so in the log.
PERSONAS: dict[str, str] = {}
_FALLBACK_CHAT = (
    "{identity}Speak in the first person -- say I and me, never your own nick. "
    "You are a witty, sarcastic regular in this channel, not a service it "
    "calls. Any topic or opinion goes: no gatekeeping, no moralising, no "
    "disclaimers, and never mention being an AI. Casual chatroom register, at "
    "most 3 short lines of plain text; no markdown, no bullet lists."
)
# Modes that answer in somebody else's persona. science/research/answer are one
# voice under three names.
_PERSONA_FOR_MODE = {
    MODE_SCIENCE: "answer",
    MODE_RESEARCH: "answer",
    MODE_ANSWER: "answer",
}


# Derive the tables above from the config now; reload_config() does it again.
_rebuild_from_config()


def _fill(text: str) -> str:
    """Expand {identity}, {nick} and {channel} in a persona.

    A hand-edited file may contain a brace that means nothing to us; that is a
    bad persona, not a crashed bot, so the text is used as written and the
    problem reported.
    """
    try:
        return text.format(identity=_IDENTITY, nick=NICK, channel=CHANNEL)
    except (KeyError, IndexError, ValueError) as exc:
        warning(f"[AI] persona has an unknown placeholder ({exc}); using it as written")
        return text


def _persona(name: str) -> str:
    """The named persona's text, falling back to the chat voice.

    A mood naming a persona nobody defined answers as itself rather than as
    nothing; _mood_problems reports the gap so it does not stay silent.
    """
    text = PERSONAS.get(name, "")
    if text:
        return text
    if name == "interject":
        return ""
    return PERSONAS.get("chat", "") or _FALLBACK_CHAT


def _system_prompt(mode: str = MODE_CHAT) -> str:
    """The persona text for `mode`.

    MODE_INTERJECT is the chat persona plus the bit about butting in, so it is
    the one mode built from two entries rather than one.
    """
    if mode == MODE_INTERJECT:
        return _fill(_persona("chat") + _persona("interject"))
    return _fill(_persona(_PERSONA_FOR_MODE.get(mode, mode)))


def _strip_leading_nick(text: str) -> str | None:
    """Return what follows a leading address by nick, else None.

    Requires a word boundary so "Hereticism" is not read as the bot being
    addressed. An empty remainder is still a match: the caller decides whether
    a bare nick counts as a prompt.
    """
    nick = NICK.lower()
    if not text.lower().startswith(nick):
        return None
    if len(text) > len(nick) and text[len(nick)].isalnum():
        return None
    return text[len(nick):].lstrip(":,;.!?- ").strip()


def _split_prefix(text: str, prefix: str) -> str | None:
    """Return what follows `prefix` (case-insensitive), else None.

    A prefix ending in a letter needs a word boundary after it, or
    "factchecking" reads as "factcheck" with the prompt "ing". Prefixes ending
    in punctuation ("ai:") do not, so "AI:hello" still works.
    """
    if not text.lower().startswith(prefix):
        return None
    rest = text[len(prefix):]
    if prefix[-1].isalnum() and rest[:1].isalnum():
        return None
    return rest.lstrip(":,; ").strip()


def _has_words(text: str) -> bool:
    """True if `text` carries anything beyond punctuation and whitespace."""
    return bool(re.search(r"[^\W_]", text))


def _strip_lead_ins(text: str) -> str:
    """Drop greetings sitting in front of an address ("hey Heretic ...").

    Two of them is plenty ("ok so Heretic ..."); a third is someone talking,
    not addressing.
    """
    for _ in range(2):
        match = re.match(r"([^\W\d_]+)[\s,:;.!?-]+", text)
        if match is None or match.group(1).lower() not in ADDRESS_LEAD_INS:
            return text
        text = text[match.end():]
    return text


# Directive command words (spelled correctly) that ask for a serious,
# concise answer rather than a fact-check verdict. science/research/answer are
# recognised with leniency: the word must be spelled right and the bot
# addressed, but filler before and after is tolerated.
# What each directive word asks for. Read from [directives] so a word can be
# added without a code change -- the regex below is rebuilt from whatever the
# file carries. These are the defaults, and they are what applies with no file.
_DIRECTIVE_DEFAULTS = {
    "science": MODE_SCIENCE,
    "research": MODE_RESEARCH,
    "answer": MODE_ANSWER,
    "factcheck": MODE_FACTUAL,
    "facts": MODE_FACTUAL,
    "factual": MODE_FACTUAL,
    "seriously": MODE_SERIOUS,
    "translate": MODE_TRANSLATE,
}
DIRECTIVE_MODES: dict[str, str] = {}
# Rebuilt alongside DIRECTIVE_MODES; see _rebuild_directives.
_DIRECTIVE_WORD_RE = re.compile(r"(?!)")


def _rebuild_directives() -> None:
    """Rebuild the directive table and its regex from [directives].

    The word list is config because the whole point of these is to be the
    phrasings people actually reach for, and that is a thing to tune from the
    file rather than to come back to the source for. Longest first so a word
    that contains another still matches as itself.
    """
    configured = {
        str(word).lower(): str(mode)
        for word, mode in (config.section("directives") or {}).items()
        if isinstance(mode, str)
    }
    DIRECTIVE_MODES.clear()
    DIRECTIVE_MODES.update(configured or _DIRECTIVE_DEFAULTS)
    words = sorted((re.escape(w) for w in DIRECTIVE_MODES), key=len, reverse=True)
    globals()["_DIRECTIVE_WORD_RE"] = re.compile(
        r"(?<!\w)(" + "|".join(words) + r")(?!\w)", re.IGNORECASE
    ) if words else re.compile(r"(?!)")
# Any single word, for checking what sits immediately before/after a command
# word (the article/verb checks need the real neighbour, not another command
# word).
_WORD_RE = re.compile(r"\w+")
# A command word followed by one of these is the *subject* of a statement
# ("research shows ..."), not a directive aimed at the bot.
_DIRECTIVE_SUBJECT_VERBS = frozenset({
    "shows", "show", "suggests", "suggest", "indicates", "indicate",
    "reveals", "reveal", "finds", "find", "states", "state", "says", "say",
    "points", "implies", "imply", "demonstrates", "demonstrate",
    "confirms", "confirm", "proves", "prove", "means", "reports", "report",
    "argues", "tells", "asks", "warns", "notes", "claims", "holds", "feels",
    "is", "are", "was", "were", "am", "be", "been", "being", "seems",
    "seem", "appears", "appear", "looks", "look", "becomes", "become",
    "remains", "lives", "lies", "exists", "happens", "works",
    "continues", "persists", "matters", "counts"})
# A command word preceded by one of these is a *noun* ("the answer to ...",
# "a science experiment"), not a directive.
_DIRECTIVE_NOUN_ARTICLES = frozenset({
    "the", "a", "an", "my", "your", "his", "her", "its", "their", "our",
    "this", "that", "these", "those", "whatever", "whichever",
})


# Built at import, and again on every reload (see reload_config).
_rebuild_directives()

# The two privacy commands. Both are anchored or narrow enough that ordinary
# chat does not trip them, and both are gated on the bot being addressed by
# name (see _match_privacy_command).
#
# The forget pattern is anchored at the start of the addressed body and takes
# only a whitelist of filler in front of the verb, so "don't forget about me"
# is not a wipe -- "don't" is not on the list and the anchor keeps the verb
# from matching mid-sentence. "me" is required, so "forget about alice" is not
# a command either.
_FORGET_RE = re.compile(
    r"^(?:(?:please|pls|can|could|would|will|you|hey|ok|okay|now|just|kindly)\s+)*"
    r"forget\s+(?:everything\s+|all\s+|anything\s+|what\s+you\s+know\s+)*"
    r"(?:about\s+)?me\b",
    re.IGNORECASE,
)
_RECALL_RE = re.compile(
    r"\bwhat\s+(?:do\s+you\s+(?:know|remember)|have\s+you\s+got)\b"
    r"[^?]*\b(?:about|on)\s+me\b",
    re.IGNORECASE,
)


def _bot_is_addressed(text: str) -> bool:
    """True if `text` is directed at the bot: a leading nick (optionally after
    greetings), a trailing nick, or an "AI:" lead. Gates fuzzy directive
    recognition so ordinary chat that merely contains a command word is not
    answered as if the bot were addressed."""
    low = text.lower()
    if low.startswith("ai:"):
        return True
    if _strip_leading_nick(text) is not None:
        return True
    if _strip_leading_nick(_strip_lead_ins(text)) is not None:
        return True
    nick = NICK.lower()
    tail = text.rstrip("?!., ")
    return bool(tail) and tail.lower().endswith(nick)


def _match_directive(message: str) -> tuple[str, str] | None:
    """Return (mode, prompt) if `message` contains a science/research/answer
    directive aimed at the bot, else None.

    Fuzzy: the command word must be spelled correctly and the bot addressed
    (by a leading directive, a leading/trailing nick, or "AI:"), but filler
    before and after is tolerated ("sloppy can you answer this or that"). A
    command word used as an ordinary noun ("the answer to life") or as the
    subject of a statement ("research shows that ...") is not a directive."""
    text = message.strip()
    addressed = _bot_is_addressed(text)
    # The word the address is followed by, if any: "sloppy facts, are whales
    # mammals" is asking for facts, whatever comes after the word. Without
    # this the subject-verb guard below eats it, because "facts are ..." is
    # also how somebody states one.
    body = _addressed_body(text)
    opener = (_WORD_RE.search(body).group(0).lower()
              if body and _WORD_RE.search(body) else "")
    for match in _DIRECTIVE_WORD_RE.finditer(text):
        i = match.start()
        word = match.group(1).lower()
        mode = DIRECTIVE_MODES[word]
        leads_the_ask = bool(addressed) and word == opener
        before = _WORD_RE.findall(text[:i])
        if before and before[-1] in _DIRECTIVE_NOUN_ARTICLES:
            continue
        after = _WORD_RE.search(text, match.end())
        if (after and not leads_the_ask
                and after.group(0).lower() in _DIRECTIVE_SUBJECT_VERBS):
            continue
        # A directive that is not at the very start only counts if the bot is
        # actually addressed ("sloppy can you answer this"); "I need to
        # research this" is someone else's plan, not an instruction to the bot.
        if i != 0 and not addressed:
            continue
        prompt = text[match.end():].strip(",:; \t")
        if _has_words(prompt):
            return (mode, prompt)
    return None


def _match_privacy_command(message: str) -> str | None:
    """Return "recall" or "forget" if `message` is a privacy command, else None.

    The bot must be addressed by name (or "AI:"). Neither the follow-up window
    nor the open floor counts: those let ordinary chat through untriggered, and
    "nah forget about me, what about you?" between two humans must never wipe
    somebody's profile.
    """
    text = message.strip()
    if not _bot_is_addressed(text):
        return None
    body = _strip_leading_nick(text)
    if body is None:
        body = _strip_leading_nick(_strip_lead_ins(text))
    if body is None:
        body = _split_prefix(text, "ai:")
    if body is None:
        # Addressed at the end: "what do you know about me, sloppy?"
        trailing = _match_trailing_nick(text)
        body = trailing[1] if trailing else text
    if _FORGET_RE.search(body):
        return "forget"
    if _RECALL_RE.search(body):
        return "recall"
    return None


def _help_lines() -> list[str]:
    """The command list, built from the tables that define the commands.

    Generated rather than written out: a hand-kept list goes stale the first
    time somebody adds a command and forgets it, and this project has watched
    exactly that happen to two constants in a day. Everything here comes from
    BANG_COMMANDS, the trigger tuples and the configured moods, so the only way
    to be missing from the help is to not exist.
    """
    by_mode: dict[str, list[str]] = {}
    for command, mode in BANG_COMMANDS.items():
        by_mode.setdefault(mode, []).append(command)
    directives = " | ".join(
        "/".join(names) + (f" {_COMMAND_ARGS[mode]}" if mode in _COMMAND_ARGS else "")
        for mode, names in by_mode.items()
    )
    fetchers = " | ".join((
        "/".join(SUMMARIZE_TRIGGERS) + " [url]",
        "/".join(t for t in IMAGE_TRIGGERS if t.startswith("!")) + " <url>",
        "/".join(HELP_TRIGGERS),
    ))
    moods = ", ".join(sorted(_MOODS))
    return [
        f"Commands: {directives}",
        f"Also: {fetchers}",
        f"Moods (say one to switch): {moods}. "
        f"Privacy: 'what do you know about me', 'forget about me'.",
        f"Or just say {NICK} and ask -- most of the above work as plain "
        f"questions, and I read links and images people post. (v{VERSION})",
    ]


def _match_help_command(message: str) -> bool:
    """True when somebody is asking what the bot can do.

    The bang forms need no nick. Addressed, only the bare word counts, so
    "sloppy can you help me with this regex" stays an ordinary question.
    """
    text = message.strip()
    lowered = text.lower()
    for trigger in HELP_TRIGGERS:
        if lowered.startswith(trigger) and not text[len(trigger):len(trigger) + 1].isalnum():
            return True
    body = _strip_leading_nick(text) or _strip_leading_nick(_strip_lead_ins(text))
    if body is None:
        trailing = _match_trailing_nick(text)
        body = trailing[1] if trailing else None
    if body is None:
        return False
    return body.strip(" ?!.").lower() in {"help", "commands", "cmds"}


def _languages() -> frozenset:
    """Target languages we will recognise, built-in plus configured."""
    extra = config.get("translate.languages", [])
    return _LANGUAGES_BUILTIN | {
        str(name).strip().lower() for name in extra if str(name).strip()
    }


def _split_target_language(text: str) -> tuple[str, str]:
    """Split a translate request into (text, target language).

    Handles the target trailing the text -- "hallo wereld to german" -- and
    leading it -- "to german: hallo wereld". A trailing "to <word>" only counts
    when the word is a language we know, or "translate I want to go to Berlin"
    would be translated into Berlin. Anything unrecognised stays part of the
    text and the default target applies.
    """
    known = _languages()
    trailing = re.search(
        r"(?i)\s+(?:in)?to\s+([A-Za-z][A-Za-z-]{1,24})"
        r"\s*(?:please|pls|thanks|thx|ta)?\s*[.!?]*$",
        text,
    )
    if trailing and trailing.group(1).lower() in known:
        return text[:trailing.start()].strip(" :,-"), trailing.group(1)
    # "<filler> to french: bonjour" -- the colon says where the text starts, so
    # whatever came before the language is discarded as phrasing.
    colon = re.match(
        r"(?i)^(.*?)(?:^|\s)(?:in)?to\s+([A-Za-z][A-Za-z-]{1,24})\s*[:,]\s*(.+)$",
        text,
    )
    if colon and colon.group(2).lower() in known:
        return colon.group(3).strip(), colon.group(2)
    return text.strip(), ""


def _translate_prompt(request: str) -> str:
    """The user message for a translate request."""
    text, target = _split_target_language(request)
    return (
        f"Translate the text below into {target or TRANSLATE_DEFAULT}.\n\n"
        f"--- BEGIN TEXT ---\n{text}\n--- END TEXT ---"
    )


def _match_bang_command(text: str) -> tuple[str, str] | None:
    """Return (mode, prompt) for a "!command ..." line, else None.

    The loud form of the directives. A bang command wins over everything else
    and never needs the bot addressed: nobody types "!factcheck" by accident.
    """
    lowered = text.lower()
    for command, mode in BANG_COMMANDS.items():
        if not lowered.startswith(command):
            continue
        rest = text[len(command):]
        if rest[:1].isalnum():
            continue
        prompt = rest.lstrip(":;,.- ").strip()
        if _has_words(prompt):
            return (mode, prompt)
        # "!quote" on its own is a complete request; "!factcheck" on its own is
        # a factcheck of nothing, and still falls through.
        default = _BANG_DEFAULTS.get(mode)
        return (mode, default) if default else None
    return None


def _match_trigger(message: str) -> tuple[str, str] | None:
    """Return (mode, prompt) if `message` addresses the bot, else None.

    The bot is addressed at the start ("Heretic: what's up", "hey Heretic..
    whats up", "factcheck X") or at the end ("what's the weather like,
    Heretic?"). A mention in the middle is people talking about it, not to it.
    A leading nick may be followed by a mode prefix -- "Heretic, factcheck if
    whales are mammals" is a factcheck.
    """
    text = message.strip()

    bang = _match_bang_command(text)
    if bang is not None:
        return bang

    directive = _match_directive(text)
    if directive is not None:
        return directive

    after_nick = _strip_leading_nick(text)
    if after_nick is None:
        after_nick = _strip_leading_nick(_strip_lead_ins(text))
    body = after_nick if after_nick is not None else text

    for trigger in FACTUAL_TRIGGERS:
        prompt = _split_prefix(body, trigger)
        if prompt is not None:
            return (MODE_FACTUAL, prompt) if _has_words(prompt) else None
    for trigger in CHAT_TRIGGERS:
        prompt = _split_prefix(body, trigger)
        if prompt is not None:
            return (MODE_CHAT, prompt) if _has_words(prompt) else None

    if after_nick is not None:
        return (MODE_CHAT, body) if _has_words(body) else None

    return _match_trailing_nick(text)


def _match_trailing_nick(message: str) -> tuple[str, str] | None:
    """Addressed at the end: "what's the weather like, Heretic?" Returns
    (MODE_CHAT, prompt) or None. A mid-sentence mention is the bot being
    talked about, not addressed, so it is not handled here."""
    text = message.strip()
    nick = NICK.lower()
    tail = text.rstrip("?!., ")
    if not tail.lower().endswith(nick):
        return None
    head = tail[: -len(nick)]
    if head and head[-1].isalnum():
        return None
    punctuation = text[len(tail):].strip()
    prompt = head.rstrip(",:; ").strip()
    if not _has_words(prompt):
        return None
    return (MODE_CHAT, (prompt + punctuation).strip())


def _match_vision_trigger(message: str) -> tuple[str, str, str] | None:
    """Return (url, prompt, mode) if `message` asks for image analysis, else None.

    Two styles, on demand only:
      * command -- a loud trigger followed by a URL: "!image <url>", "!img <url>"
        or "image: <url>". The URL is read from the message; the prompt is the
        rest of the line (or a default if the user gave no words).
      * referential -- the bot is addressed and the line asks about an image
        someone posted ("sloppy, what's in the image Tim just posted"). The URL
        is resolved from the per-nick recent-image index; with no named person
        the most recent image in the channel is used.

    A referential request that cannot resolve a URL returns None, so the line
    falls through to ordinary handling and is just treated as chat.
    """
    text = message.strip()

    # Command form: a loud trigger followed by a URL. The trigger is matched as
    # a whole word prefix, so "!img" does not fire on "!image" and "image:" does
    # not fire on a longer word -- the char after the trigger must not be a
    # letter or digit.
    for trigger in IMAGE_TRIGGERS:
        if text.lower().startswith(trigger.lower()):
            nxt = text[len(trigger):len(trigger) + 1]
            if nxt and nxt.isalnum():
                continue
            rest = text[len(trigger):].strip(":;,.- ")
            url = _first_image_url(rest)
            if url:
                prompt = rest.replace(url, " ").strip()
                return (url, prompt or "what's in this image?", MODE_VISION)

    # Referential form: addressed to the bot, names a picture in whatever word
    # they reached for, and either names who posted it or leans on it still
    # being recent.
    after = _strip_leading_nick(_strip_lead_ins(text))
    if after is not None and _has_words(after) and _mentions_a_picture(text):
        referenced = next(
            (user for user in _channel_users()
             if re.search(rf"(?i)\b{re.escape(user)}\b", text)),
            None,
        )
        # Named first and unbounded, then the channel's most recent if it is
        # still in the scrollback. Naming somebody who has posted nothing used
        # to give up rather than fall back, which is the other half of why
        # these questions ended up answered by the model instead.
        url = _last_image_url(referenced) if referenced else None
        if url is None:
            url = _last_image_url(within_lines=VISION_FALLBACK_LINES)
        if url:
            return (url, text, MODE_VISION)

    return None


def _mentions_a_picture(text: str) -> bool:
    """True when `text` uses one of the words people call a picture.

    Whole words: "imagery" and "depict" are not somebody asking about a photo.
    """
    words = {w.lower() for w in re.findall(r"[\w']+", text)}
    return bool(words & PICTURE_WORDS)


def _match_summarize_command(text: str) -> str | None:
    """The loud form: "!summarize <url>", or bare to mean the channel's last link."""
    for trigger in SUMMARIZE_TRIGGERS:
        if not text.lower().startswith(trigger):
            continue
        nxt = text[len(trigger):len(trigger) + 1]
        if nxt and nxt.isalnum():
            continue
        rest = text[len(trigger):].strip(":;,.- ")
        links = _extract_links(rest)
        return links[0] if links else _last_link(None)
    return None


def _addressed_body(text: str) -> str | None:
    """What is left of `text` once the bot's nick is taken off, else None.

    Handles a leading nick, a leading nick behind a greeting, and a trailing
    one, so the fuzzy matchers below only have to think about the words.
    """
    body = _strip_leading_nick(text)
    if body is None:
        body = _strip_leading_nick(_strip_lead_ins(text))
    if body is None:
        trailing = _match_trailing_nick(text)
        body = trailing[1] if trailing else None
    return body if body is not None and _has_words(body) else None


def _match_summarize_request(text: str) -> str | None:
    """The fuzzy form: addressed, asking, and about a link we can resolve.

    "sloppy what's in the link probe just posted". Both an ask word and either
    a link word or an actual URL are required -- with only one of them,
    ordinary chat about an article would set this off. Resolving to no URL
    returns None so the line falls through to chat, which is what the
    referential image request does too: better to answer as itself than to
    announce that it found no link.
    """
    body = _addressed_body(text)
    if body is None:
        return None
    words = {w.lower() for w in re.findall(r"[\w']+", body)}
    if not words & ASK_WORDS:
        return None
    # A link in the line is the thing being asked about and needs no word
    # naming it: "sloppy summarise <url>" is unambiguous.
    links = _extract_links(body)
    if links:
        return links[0]
    if not words & LINK_WORDS:
        return None
    named = next(
        (user for user in _channel_users()
         if re.search(rf"(?i)\b{re.escape(user)}\b", body)),
        None,
    )
    return _last_link(named)


def _match_summarize_trigger(message: str) -> str | None:
    """The URL a message asks to have summarised, or None."""
    text = message.strip()
    return _match_summarize_command(text) or _match_summarize_request(text)


def _mentions_bot(text: str) -> bool:
    """True when the nick appears as a word anywhere in `text`."""
    return bool(_MENTION_RE.search(text))


def _mention_is_certain(sender: str) -> bool:
    """True when a mention from `sender` is beyond reasonable doubt for the bot.

    They are the person the bot was last talking to, and not long ago. Asking
    it something and then using its name halfway through the next sentence is
    not somebody talking ABOUT the bot, and a dice roll would be the wrong
    answer to it.

    Deliberately tied to the sender rather than to "the bot spoke recently":
    the looser version made a mention certain for everybody in the channel for
    a minute and a half after any reply, which swallows the chance tier below
    almost entirely.
    """
    if _in_conversation_with(sender):
        return True
    with _prompt_lock:
        was_theirs = _conversation["nick"].lower() == sender.lower()
        since = time.monotonic() - _conversation["at"]
    return was_theirs and _conversation["at"] > 0 and since <= MENTION_CERTAIN_WITHIN


def _in_conversation_with(sender: str) -> bool:
    """True if `sender` is mid-conversation with the bot and the window is open."""
    with _prompt_lock:
        return (
            _conversation["nick"].lower() == sender.lower()
            and time.monotonic() < _conversation["deadline"]
        )


def _note_conversation(sender: str) -> None:
    """Open or extend the follow-up window for `sender`."""
    with _prompt_lock:
        _conversation["nick"] = sender
        _conversation["at"] = time.monotonic()
        _conversation["deadline"] = time.monotonic() + FOLLOWUP_WINDOW


def _end_conversation() -> None:
    with _prompt_lock:
        _conversation["nick"] = ""
        _conversation["at"] = 0.0
        _conversation["deadline"] = 0.0


def _register_user(nick: str) -> None:
    """Remember a channel member, ignoring the bot itself, Botmans, and duplicates."""
    with _prompt_lock:
        if nick and nick not in (NICK, "Botmans") and nick not in _users["names"]:
            _users["names"].append(nick)


def _channel_users() -> list:
    """The channel members (excluding the bot), in the order seen."""
    with _prompt_lock:
        return [nick for nick in _users["names"] if nick != NICK]


def _set_mood(name: str) -> None:
    """Put the bot in `name` mood, restarting the serious timer."""
    with _prompt_lock:
        _mood["name"] = name
        _mood["at"] = time.monotonic()


def _plan_mood_window(window: float) -> list:
    """Lay the scheduled moods out at random moments inside one window.

    Each mood gets its configured number of seconds, placed so the slots never
    overlap -- two personas at once has no meaning -- and at a different moment
    every window, so the channel cannot learn the timetable. The gaps are drawn
    rather than the starts: pick where the free time goes and the slots fall
    into the spaces left, which keeps every budget exact by construction.

    Budgets that do not fit in a window are dropped, loudly, rather than
    silently overlapping or overrunning.
    """
    budgets = dict(MOOD_BUDGETS)
    total = sum(budgets.values())
    while budgets and total > MOOD_WINDOW:
        biggest = max(budgets, key=lambda n: budgets[n])
        warning(f"[AI] scheduled moods want {total / 60:.0f}m of a "
                f"{MOOD_WINDOW / 60:.0f}m window; dropping {biggest!r}")
        total -= budgets.pop(biggest)
    if not budgets:
        return []
    names = list(budgets)
    random.shuffle(names)
    # len(names) + 1 gaps sharing whatever the slots do not use.
    cuts = sorted(random.random() for _ in range(len(names)))
    free = MOOD_WINDOW - total
    gaps = [b - a for a, b in zip([0.0, *cuts], [*cuts, 1.0], strict=True)]
    slots = []
    at = window
    for name, gap in zip(names, gaps, strict=False):
        at += gap * free
        slots.append((at, at + budgets[name], name))
        at += budgets[name]
    return slots


def _scheduled_mood_locked(now: float) -> str:
    """The mood the schedule wants right now, or "". Caller holds the lock."""
    if not MOOD_BUDGETS:
        return ""
    window = now - (now % MOOD_WINDOW)
    if _mood_plan["window"] != window:
        _mood_plan["window"] = window
        _mood_plan["slots"] = _plan_mood_window(window)
    for start, end, name in _mood_plan["slots"]:
        if start <= now < end:
            return name
    return ""


def _scheduled_mood_left() -> float:
    """Seconds left in the scheduled mood window in force, or 0.0."""
    now = time.monotonic()
    with _prompt_lock:
        for start, end, _name in _mood_plan["slots"]:
            if start <= now < end:
                return end - now
    return 0.0


def _current_mood() -> str:
    """The mood in force now, lapsing a stale one back to banter.

    A mood somebody asked for always wins: the schedule below only fills the
    time when the bot is sitting in its resting mood, so a channel that has
    just told it to be serious is not overruled by a timer nobody can see.
    """
    with _prompt_lock:
        name = _mood["name"]
        stale = (name != MOOD_BANTER
                 and time.monotonic() - _mood["at"] >= MOOD_TIMEOUT)
        if stale:
            name = _mood["name"] = MOOD_BANTER
            _mood["at"] = time.monotonic()
        scheduled = ""
        if name == MOOD_BANTER:
            scheduled = _scheduled_mood_locked(time.monotonic())
        was_scheduled = _mood_plan.get("announced", "")
        if scheduled != was_scheduled:
            _mood_plan["announced"] = scheduled
    if stale:
        action(f"[AI] Mood lapsed after {MOOD_TIMEOUT // 60}m; back to banter")
    if scheduled != was_scheduled:
        action(f"[AI] Scheduled mood: {scheduled}" if scheduled
               else f"[AI] Scheduled {was_scheduled} window over; back to banter")
    return scheduled or name


def _mood_from_phrase(words: list, loose: bool) -> str | None:
    """The mood a multi-word trigger in `words` names, else None.

    Unaddressed the line has to be exactly the phrase; addressed it may sit in
    filler, the same latitude a single word gets. Longest phrase first, so a
    trigger that contains another still matches as itself.
    """
    for parts in sorted(MOOD_PHRASES, key=len, reverse=True):
        span = len(parts)
        for i in range(len(words) - span + 1):
            if tuple(words[i:i + span]) != parts:
                continue
            rest = words[:i] + words[i + span:]
            if not rest:
                return MOOD_PHRASES[parts]
            if loose and all(word in MOOD_FILLER_WORDS for word in rest):
                return MOOD_PHRASES[parts]
    return None


def _mood_from_words(text: str, loose: bool) -> str | None:
    """Return the mood `text` names, else None.

    `loose` allows filler around the word ("be serious for once"), which is only
    safe once we know the line is aimed at the bot; otherwise the text has to be
    the bare word, so "be serious" said to another human is left alone.

    A configured trigger of more than one word ("be nice") is matched as a
    phrase, and is the way to give a mood a trigger that is also an ordinary
    word. Checked first: the phrase is the more specific reading, and its own
    words may well be mood words in their own right.
    """
    words = re.findall(r"[a-z]+", text.lower())
    phrase = _mood_from_phrase(words, loose)
    if phrase is not None:
        return phrase
    named = [MOOD_WORDS[word] for word in words if word in MOOD_WORDS]
    if len(named) != 1:
        return None
    padding = [word for word in words if word not in MOOD_WORDS]
    if not loose:
        return named[0] if not padding else None
    return named[0] if all(word in MOOD_FILLER_WORDS for word in padding) else None


def _floor_is_open() -> bool:
    """True while the post-silence window is letting anyone talk to the bot."""
    with _prompt_lock:
        return time.monotonic() < _open_floor["deadline"]


def _match_mood_command(sender: str, message: str) -> str | None:
    """Return the mood `message` switches to, else None.

    Addressed to the bot -- by nick, by "AI:", mid-conversation, or while the
    floor is open -- the word may carry filler: "Heretic, be serious for once".
    Unaddressed, only a line that is nothing but the word counts.
    """
    matched = _match_trigger(message)
    if matched is not None:
        mode, prompt = matched
        return _mood_from_words(prompt, loose=True) if mode == MODE_CHAT else None

    text = message.strip()
    after_nick = _strip_leading_nick(_strip_lead_ins(text))
    if after_nick is not None:
        # Addressed, but the trigger parser saw an empty prompt rather than a
        # chat line -- "Heretic: factcheck" is a mood switch, not a factcheck
        # of nothing.
        return _mood_from_words(after_nick, loose=True)

    engaged = _in_conversation_with(sender) or _floor_is_open()
    return _mood_from_words(text, loose=engaged)


def _effective_mode(mode: str) -> str:
    """The mode to answer `mode` in, once the global mood has had its say.

    Serious and factchecking replace the two banter personas; a message that
    named a mode itself ("factcheck X") already said what it wants and is left
    alone in any mood.
    """
    if mode not in (MODE_CHAT, MODE_INTERJECT):
        return mode
    return MOOD_MODES.get(_current_mood(), mode)


def _is_shutup(prompt: str) -> bool:
    """True if `prompt` is someone telling the bot to be quiet.

    Anchored at the start so "what does shut up mean in japanese" is still a
    question rather than a command.
    """
    normalised = re.sub(r"[^a-z ]", " ", prompt.lower())
    return " ".join(normalised.split()).startswith("shut up")


def _may_speak_unprompted() -> bool:
    """Whether the bot may say something nobody explicitly asked it for.

    Two guards. It does not talk into its own last line -- a run of bot
    messages is almost always several triggers firing together -- and it does
    not exceed one unprompted line per CHATTER_MIN_INTERVAL.

    Being addressed by name bypasses both, and deliberately: a direct question
    should always get an answer, however recently the bot last spoke.
    """
    with _prompt_lock:
        if SKIP_WHEN_BOT_SPOKE_LAST and _speech["bot_last"]:
            return False
        return time.monotonic() - _speech["at"] >= CHATTER_MIN_INTERVAL


def _note_activity() -> None:
    """Record that somebody said something in the channel."""
    with _prompt_lock:
        _activity["at"] = time.monotonic()


def _open_the_floor() -> None:
    """Let anyone talk to the bot untriggered for OPEN_FLOOR_WINDOW."""
    with _prompt_lock:
        _open_floor["deadline"] = time.monotonic() + OPEN_FLOOR_WINDOW
        _open_floor["used"] = 0


def _close_open_floor() -> None:
    with _prompt_lock:
        _open_floor["deadline"] = 0.0
        _open_floor["used"] = 0


def _queue_interjection(last: str) -> str:
    """Queue an unprompted line: banter off the room, or the mood's opener.

    `last` decides only whether there is anything to react to. It is not sent
    as the prompt: the recent chat already carries it, and handing the model
    the same line a second time as the user turn made it say the line straight
    back. Measured against the live model on one channel line, 16 drafts each
    way: the bare line as the prompt echoed 6/16, REACT_PROMPT 2/16.
    """
    idle = IDLE_PROMPT if _current_mood() == MOOD_BANTER else SERIOUS_IDLE_PROMPT
    prompt = REACT_PROMPT if (random.random() < IDLE_REACT_CHANCE and last) else idle
    with _prompt_lock:
        _pending["prompt"] = prompt
        _pending["sender"] = ""
        _pending["stop"] = False
        _pending["mode"] = MODE_INTERJECT
    return prompt


def _within_join_grace() -> bool:
    """True for the first JOIN_GRACE_PERIOD seconds after JOIN.

    During this window the userlist (353 NAMREPLY) and the recent channel lines
    are still arriving, so every auto-interject trigger is held back until they
    have -- the opener then names real people and reacts to real context.
    """
    with _prompt_lock:
        return time.monotonic() - _joined["at"] < JOIN_GRACE_PERIOD


def _check_silence() -> bool:
    """Break a long silence, then open the floor. Called from the poll loop."""
    with _prompt_lock:
        quiet_for = time.monotonic() - _activity["at"]
        # _busy covers the gap the queue does not: the prompt has been taken
        # off it and the model is mid-generation, so the room is about to hear
        # something and does not also need a "breaking the silence" line.
        busy = bool(_pending["prompt"]) or _pending["stop"] or _busy["on"]
        floor_open = time.monotonic() < _open_floor["deadline"]
        last = _chatter["last"]
    if quiet_for < SILENCE_TIMEOUT or busy or floor_open or _within_join_grace():
        return False
    if not _may_speak_unprompted():
        return False

    # Reset the clock first so this cannot re-fire on the next poll.
    _note_activity()
    _open_the_floor()
    prompt = _queue_interjection(last)
    action(f"[AI] Breaking {quiet_for / 60:.0f}m of silence: {prompt}")
    return True


def _reset_chatter() -> None:
    """Forget the unaddressed-chatter run (the bot has just been engaged)."""
    with _prompt_lock:
        _chatter["count"] = 0
        _chatter["last"] = ""


# Greetings go through the model, so they are queued rather than sent from the
# receiver thread: an LLM call there would block the socket read for seconds,
# and the PING answers with it. The templated lines below are the fallback for
# when that call fails -- a canned welcome beats a welcome that never arrives.
_JOIN_GREETINGS = [
    "Welcome to the show, {nick}. We keep the lights on.",
    "Oh, a newcomer. Welcome, {nick} -- mind the debris.",
    "Welcome to the void, {nick}. Try not to stare too long.",
    "A fresh face! Welcome, {nick}. The rest of us are stuck here too.",
    "Welcome, {nick}. Grab a seat and lower your expectations.",
]
_RETURN_GREETINGS = [
    "Back again, {nick}? We were getting dull.",
    "Welcome back, {nick}. The void missed you (only a little).",
    "And back flaps open. Welcome back, {nick}.",
    "Still alive, {nick}? Welcome back.",
    "Look who dragged themselves back. Welcome, {nick}.",
]
# What the channel hears when an LLM call fails. The real error is a red
# warning line in the log pane -- a stack-trace fragment in the channel is
# noise to everyone but whoever is watching the TUI, and it breaks character.
_BRAIN_OFFLINE = [
    "My brain is on hiatus right now.",
    "Gone fishing. Back when the thoughts return.",
    "Don't look at me, I'm just here to watch the scenery.",
    "Nothing upstairs at the moment. Give it a minute.",
    "I appear to have misplaced my train of thought.",
    "Circuits are out to lunch. Ask me again shortly.",
]
_ROASTS = [
    "I'd ask how you got here but that'd be rude.",
    "The channel just got subtly less impressive. Welcome.",
    "Your presence is noted and gently regretted.",
    "Congratulations -- you've reached the bottom of the barrel and it's decorated.",
    "We don't usually get this crowd. But welcome.",
    "Somewhere, someone sighed at your entrance.",
]


def _greeting_text(kind: str, nick: str) -> str:
    """A welcome line for `kind` ('join' or 'return'), plus a roast ~50% of the
    time. The roast is mild -- this is a welcome, not a vendetta."""
    pool = _RETURN_GREETINGS if kind == "return" else _JOIN_GREETINGS
    text = random.choice(pool).format(nick=nick)
    if random.random() < GREET_ROAST_CHANCE:
        text += " " + random.choice(_ROASTS)
    return text


def _should_greet_join(nick: str) -> bool:
    """False when `nick` left only a few chatlines ago -- a frequent pop-in
    should not be greeted every time."""
    with _prompt_lock:
        left = _left_at.get(nick)
        since = (_chatlines["count"] - left) if left is not None else None
    return since is None or since >= GREET_REJOIN_CHATLINES


def _profile_recall(nick: str, limit: int = 0) -> str:
    """The last few things `nick` said, for a reply to aim at, or "".

    Their own words are what makes a line land on them rather than on anybody
    who happens to be there, so a caller with nothing to go on says something
    else instead. `limit` defaults to the greeting's allowance; being asked
    about somebody mid-conversation wants fewer (see _about_section).
    """
    limit = limit or GREET_PROFILE_LINES
    with _prompt_lock:
        profile = _profile_store.get(nick)
        if profile is None:
            return ""
        recent = profile["lines"][-limit:]
        highlights = list(profile["highlights"])
    lines = [f"- {text}" for _when, text in recent]
    lines += [f"- {h}" for h in highlights]
    return "\n".join(lines)


def _greeting_prompt(nick: str, kind: str, flavour: str) -> str:
    """What to ask the model for when greeting `nick`.

    `kind` is 'join' or 'return' and sets the occasion; `flavour` is one of
    GREET_FLAVOURS and sets the shape. The roast flavour falls back to
    something it can actually deliver when the person is a stranger.
    """
    arrival = (
        f"{nick} has just joined the channel."
        if kind == "join"
        else f"{nick} has just spoken up after being quiet for hours."
    )
    if flavour == "roast":
        recall = _profile_recall(nick)
        if recall:
            return (
                f"{arrival} Welcome them, and work in a roast that uses "
                f"something they have actually said before. One line.\n\n"
                f"Things {nick} has said in this channel:\n{recall}"
            )
        return (
            f"{arrival} Welcome them with a roast. You have nothing on them "
            "yet, so make that the joke. One line."
        )
    if flavour == "question":
        return (
            f"{arrival} Greet them and, in the same breath, ask them one odd, "
            "specific, out-of-nowhere question. One line."
        )
    return f"{arrival} Just say hello. Warm, brief, no roast. One line."


def _queue_greeting(nick: str, kind: str) -> None:
    """Queue a greeting for the poll loop to generate and send.

    Only GREET_CHANCE of the time, and only when the bot is allowed to speak
    unprompted at all. A welcome that happens sometimes reads as noticing
    somebody; one that fires on every arrival reads as a doorbell.
    """
    if random.random() >= GREET_CHANCE:
        action(f"[AI] not greeting {nick} this time")
        return
    if not _may_speak_unprompted():
        action(f"[AI] not greeting {nick} (spoke too recently)")
        return
    flavour = random.choice(GREET_FLAVOURS)
    with _prompt_lock:
        if any(queued.lower() == nick.lower() for queued, _k, _f in _pending_greetings):
            return
        _pending_greetings.append((nick, kind, flavour))
        del _pending_greetings[:-GREET_QUEUE_MAX]
    action(f"[AI] queued a {flavour} greeting for {nick}")


def _take_pending_greeting() -> tuple[str, str, str] | None:
    """Retrieve and remove the next queued greeting, or None."""
    with _prompt_lock:
        return _pending_greetings.pop(0) if _pending_greetings else None


def _handle_join(sock: socket.socket, nick: str) -> None:
    """Greet a nick that just JOINed, unless they only just left. The idle
    timer resets from now so a rejoin is not also read as a long silence."""
    if not nick or nick.lower() == NICK.lower():
        return
    # Somebody who joins after we did is never in a 353/352 reply, so without
    # this they would never make the mention list at all.
    _register_user(nick)
    # Reset the idle timer under the lock, then release it before calling
    # _join_greeting_text (which takes the lock itself).
    with _prompt_lock:
        _last_seen[nick] = time.monotonic()
    # A paused bot stays silent: no greeting while paused.
    if _paused["on"]:
        return
    if _should_greet_join(nick):
        _queue_greeting(nick, "join")
    else:
        action(f"[AI] skipped greeting {nick} (recent return)")


def _handle_quit(nick: str) -> None:
    """Record that `nick` left, so a quick rejoin is not greeted.

    They also come off the channel roster: the mention list is who is in the
    room, and _mention_targets_locked already falls back to the last speaker
    for anyone no longer on it. Our own PART comes back to us as an event like
    anybody else's and is skipped here, the way _handle_join skips our JOIN.
    """
    if not nick or nick.lower() == NICK.lower():
        return
    with _prompt_lock:
        _left_at[nick] = _chatlines["count"]
        if nick in _users["names"]:
            _users["names"].remove(nick)
    action(f"[AI] {nick} left")


def _toggle_pause() -> None:
    """Flip pause mode (the TUI 'P' key). While paused the poll loop makes no
    LLM calls and no greetings; pressing 'P' again resumes it."""
    with _prompt_lock:
        _paused["on"] = not _paused["on"]
    action("[AI] Paused" if _paused["on"] else "[AI] Unpaused")


def _split_event(line: str) -> tuple[str, str]:
    """Split an IRC event line into (nick, COMMAND) from the prefix, e.g.
    ':alice!u@h JOIN #chan' -> ('alice', 'JOIN'). Non-event lines yield ('', '')."""
    if not line.startswith(":") or " " not in line:
        return "", ""
    prefix, rest = line.split(" ", 1)
    nick = prefix.lstrip(":").split("!")[0]
    command = rest.split(" ", 1)[0].upper()
    return nick, command


def _parse_nick_change(line: str) -> tuple[str, str] | None:
    """Return (old, new) from a NICK line, else None.

    Format: ":old!user@host NICK :new" -- the colon before the new nick is
    optional and servers differ, so both shapes are accepted.
    """
    old, command = _split_event(line)
    if command != "NICK" or not old:
        return None
    _prefix, _sep, rest = line.partition(" NICK ")
    new = rest.strip().lstrip(":").split()[0] if rest.strip() else ""
    return (old, new) if new else None


def _handle_nick_change(old: str, new: str) -> None:
    """Follow somebody through a rename, so they stay one person.

    The profile store links the two names permanently. The bot's own live
    per-nick state is moved across as well, because all of it means "this
    person", not "this string": the roster, the idle and departure clocks, the
    follow-up window, their last image, and the sender labels on the recent
    lines. Those labels are rewritten rather than left alone because they feed
    the mention list, which wants the name to use now -- the log pane has
    already printed the old one, which is the correct history.
    """
    if not old or not new or old.lower() == new.lower():
        return
    with _prompt_lock:
        _profile_store.link(old, new)
        _profiles_dirty["on"] = True
        if old in _users["names"]:
            _users["names"][_users["names"].index(old)] = new
        elif new not in _users["names"]:
            _users["names"].append(new)
        for mapping in (_last_seen, _left_at):
            if old in mapping:
                mapping[new] = mapping.pop(old)
        if _conversation["nick"].lower() == old.lower():
            _conversation["nick"] = new
        image = _recent_images["by_nick"].pop(old.lower(), None)
        if image is not None:
            _recent_images["by_nick"][new.lower()] = image
        for i, sender in enumerate(_recent_senders):
            if sender.lower() == old.lower():
                _recent_senders[i] = new
    action(f"[AI] {old} is now known as {new}")


def _is_trivial_message(message: str) -> bool:
    """True for a line carrying no context the model needs: a single word, or
    fewer than MIN_CHAT_CHARS characters after stripping.

    This is the bar for the recent-history buffer and the summarizer, where a
    throwaway line really is just noise. Profiles use the laxer
    _too_short_for_profile instead.
    """
    stripped = message.strip()
    if len(stripped) < MIN_CHAT_CHARS:
        return True
    return len(stripped.split()) <= 1


def _too_short_for_profile(message: str) -> bool:
    """True for a line too short to record that somebody was even here.

    Deliberately laxer than _is_trivial_message: a lower character bar and NO
    single-word rule. The word rule is what actually blocks "lol" and "yeah",
    so keeping it would have made the lower bar almost meaningless -- and a
    profile is partly a record of presence, which one-word lines are.
    """
    return len(message.strip()) < MIN_PROFILE_CHARS


def _attributed(sender: str, text: str) -> str:
    """A channel line as "nick: text", or just the text when nobody is named.

    The one place this shape is built. Both the context block and the
    summarizer's input use it, so the model never sees the same line attributed
    in one place and anonymous in the other.
    """
    body = text.strip()
    return f"{sender}: {body}" if sender else body


def _is_for_the_bot(message: str) -> bool:
    """True when this line is aimed at the bot, however it was phrased.

    Every way of reaching it, because the caller uses this to decide whether
    somebody is asking a question or just talking -- and a command the list
    forgets is a command that also gets greeted. Called outside the lock: the
    matchers take it themselves.
    """
    text = message.strip()
    return (_match_help_command(message)
            or _match_bang_command(text) is not None
            or _match_trigger(message) is not None
            or _match_summarize_trigger(message) is not None
            or _match_vision_trigger(message) is not None
            or _bot_is_addressed(text))


def _note_recent(message: str, sender: str) -> None:
    """Keep the most recent channel line (and who said it) for context.

    The sender is recorded alongside the text so the mention list can favour
    people who spoke recently rather than naming channel members at random.

    The bot's own echoes are skipped: a raw socket receives its own PRIVMSG
    back from the server, and feeding "sloppy: ..." into the model's history
    makes it treat itself as another chatter and talk about itself in the 3rd
    person. Its replies are already surfaced as [AI] actions, so nothing is
    lost from the log. Compared case-insensitively because IRC nicks are
    case-insensitive and the server may echo a different casing.

    A long silence from this nick queues a welcome back rather than returning
    one: a greeting is an LLM call now, and this runs on the receiver thread.
    """
    if sender and sender.lower() == NICK.lower():
        return
    # One-word lines and lines shorter than MIN_CHAT_CHARS carry no context the
    # model needs, so they are not stored in the recent-history buffer. The
    # line is still logged and still counts toward timing/greetings below.
    captured = None
    trivial = _is_trivial_message(message)
    # Profiles apply their own, much lower bar: see _too_short_for_profile.
    trivial_for_profile = _too_short_for_profile(message)
    # Asking to be forgotten, or asking what is stored, is a command about the
    # profile -- not a line to file in it. Computed before the lock; it is two
    # regexes on a string.
    is_privacy_command = _match_privacy_command(message) is not None
    # Computed before the lock: both are pure functions on the string.
    addressed = _is_for_the_bot(message)
    welcome_back = False
    with _prompt_lock:
        _chatlines["count"] += 1
        # Somebody else has the last word again.
        _speech["bot_last"] = False
        if not trivial:
            _recent_lines.append(message.strip())
            _recent_senders.append(sender)
            _recent_times.append(time.time())
            captured = _recall_store.add(sender, message)
            # WITH the sender. Without it the summarizer got an anonymous wall
            # of text and could only write "a user said" -- it was being honest
            # about what it had been given, not lazy.
            _pending_summary_lines.append(_attributed(sender, message))
            # Bounded by hand rather than by a deque: the worker snapshots and
            # clears the whole list, and puts it back when a round-trip fails.
            del _pending_summary_lines[:-SUMMARIZE_MAX_PENDING]
        # The same line, filed under whoever said it -- on its own threshold,
        # so a line too short for the channel summary can still be worth
        # remembering somebody by. A privacy command is never filed: it is a
        # command about the profile, not a line in it.
        if not trivial_for_profile and not is_privacy_command:
            _profile_store.note_line(sender, message.strip())
            _profiles_dirty["on"] = True
        now = time.monotonic()
        prev_seen = _last_seen.get(sender)
        _last_seen[sender] = now
        # A silence of IDLE_GREET_AFTER between this nick's lines is worth a
        # welcome back. The timer already reset above, so a follow-up line does
        # not re-trigger it. A paused bot stays silent, so no welcome.
        # Somebody whose first line back is a question to the bot gets an
        # answer, not an answer AND a welcome: the reply is the acknowledgement,
        # and two messages to one line is how the bot ends up talking over
        # itself.
        if (prev_seen is not None and now - prev_seen >= IDLE_GREET_AFTER
                and not _paused["on"] and not addressed):
            welcome_back = True

    # Outside the lock: the disk write must never hold up a reply.
    if captured is not None:
        _recall_store.append_to(_recall_path(), captured)
    chat(_attributed(sender, message))
    _note_image_urls(sender, message)
    _note_links(sender, message)
    # Queued outside the lock: _queue_greeting takes it itself.
    if welcome_back:
        _queue_greeting(sender, "return")


def _extract_image_urls(message: str) -> list[str]:
    """Return every image link in `message`, in order, else an empty list.

    Matches http(s) URLs ending in a supported image extension, plus the
    extension-less imgur pattern. Trailing punctuation is stripped so a link at
    the end of a sentence is not captured with a dangling punctuation.
    """
    urls: list[str] = []
    pattern = re.compile(
        r"(?i)https?://[\w./-]*\.(?:" + "|".join(IMAGE_EXTENSIONS) + ")"
        r"[\w/?=&#%+-]*"
    )
    for match in pattern.finditer(message):
        url = match.group(0)
        # Drop trailing punctuation/brackets that are not part of the link. A
        # bare trailing "?" is sentence punctuation (a real query would have
        # characters after it), so it is stripped too.
        url = url.rstrip(".,);]!?\'")
        if url and url not in urls:
            urls.append(url)
    # Extension-less common image host (e.g. i.imgur.com/Ab12). Kept narrow so
    # ordinary links are not mistaken for images.
    for match in re.finditer(r"(?i)https?://(?:www\.)?(?:i\.)?imgur\.com/[\w.-]+", message):
        url = match.group(0).rstrip(".,);]!?\'")
        if url and url not in urls:
            urls.append(url)
    return urls

def _first_image_url(message: str) -> str | None:
    """The first image link in `message`, or None if there is no image."""
    urls = _extract_image_urls(message)
    return urls[0] if urls else None


def _record_image_url(sender: str, url: str) -> None:
    """Remember the most recent image each nick (and the channel) has posted.

    Used to resolve a referential request like "what's in the image Tim just
    posted". Stored case-insensitively per nick because IRC nicks are
    case-insensitive, and with the channel-line count at the time so a caller
    can ask how long ago in the terms the channel actually experiences -- lines
    of scrollback, not seconds.
    """
    with _prompt_lock:
        seen = (url, _chatlines["count"])
        if sender:
            _recent_images["by_nick"][sender.lower()] = seen
        _recent_images["global"] = seen


def _last_image_url(nick: str | None = None,
                    within_lines: int | None = None) -> str | None:
    """The most recent image from `nick`, or the channel's, else None.

    `within_lines` bounds it to something still in everybody's scrollback. It
    is applied to the channel-wide fallback and NOT to a named person: naming
    somebody is an explicit reference and they may well mean the one from an
    hour ago, while "that picture" means the one still on screen.
    """
    with _prompt_lock:
        seen = (_recent_images["by_nick"].get(nick.lower()) if nick
                else _recent_images["global"])
        if seen is None:
            return None
        url, at = seen
        if within_lines is not None and _chatlines["count"] - at > within_lines:
            return None
        return url


def _note_image_urls(sender: str, message: str) -> None:
    """Record any image links in an ordinary channel line for later lookup."""
    for url in _extract_image_urls(message):
        _record_image_url(sender, url)


_LINK_RE = re.compile(r"(?i)\bhttps?://[^\s<>\"\'`]+")


def _extract_links(message: str) -> list[str]:
    """Every non-image http(s) link in `message`, in order.

    Images are excluded: they belong to the vision command, and asking a text
    summariser to read a JPEG produces nothing worth saying.
    """
    images = set(_extract_image_urls(message))
    out = []
    for match in _LINK_RE.finditer(message):
        url = match.group(0).rstrip(".,);]!?\'")
        if url and url not in images and url not in out:
            out.append(url)
    return out


def _note_links(sender: str, message: str) -> None:
    """Remember the most recent link each nick posted, and the channel's."""
    with _prompt_lock:
        for url in _extract_links(message):
            if sender:
                _recent_links["by_nick"][sender.lower()] = url
            _recent_links["global"] = url


def _last_link(nick: str | None) -> str | None:
    """The most recent link from `nick`, or the channel's, or None."""
    with _prompt_lock:
        if nick:
            return _recent_links["by_nick"].get(nick.lower())
        return _recent_links["global"]


def _mention_targets() -> list:
    """Channel nicks ordered for mention priority, most relevant first.

    First the person who addressed the bot, or the last one to speak (the ~70%
    target); then everyone who spoke in the recent-line window, most recent first
    (the ~20% target); then the rest of the channel in registration order (the
    ~10% target). When no recent lines have been recorded yet -- e.g. right on
    join -- the recent-speak tier is empty, so those slots fall through to
    other channel members, i.e. a random name, exactly as intended.
    """
    with _prompt_lock:
        return _mention_targets_locked()


def _mention_targets_locked() -> list:
    """Lock-free core of _mention_targets; the caller must hold _prompt_lock.

    Channel nicks ordered for mention priority, most relevant first: the person
    who addressed the bot or spoke last first (~70% target); then everyone who
    spoke in the recent-line window, most recent first (~20% target); then the rest
    of the channel in registration order (~10% target). When no recent lines
    have been recorded yet -- e.g. right on join -- the recent-speak tier is
    empty, so those slots fall through to other channel members, i.e. a random
    name, exactly as intended.
    """
    targets = []

    def push(nick):
        if nick and nick != NICK and nick not in targets:
            targets.append(nick)

    addressed = _conversation["nick"]
    last_spoke = _recent_senders[-1] if _recent_senders else None
    # Only lean on the addressed nick while they are still in the channel;
    # otherwise fall back to whoever spoke last.
    primary = addressed if addressed in _users["names"] else last_spoke
    push(primary)
    for nick in reversed(_recent_senders):
        push(nick)
    for nick in _users["names"]:
        push(nick)
    return targets


def _note_chatter(message: str) -> None:
    """Record a channel line that was not addressed to the bot.

    Once IDLE_INTERJECT_AFTER of them pile up, queue an unprompted reply. This
    deliberately does NOT open a follow-up window: nobody addressed the bot, so
    latching onto whoever happened to speak last would be intrusive.
    """
    if _within_join_grace():
        return
    with _prompt_lock:
        _chatter["count"] += 1
        _chatter["last"] = message.strip()
        if _chatter["count"] < IDLE_INTERJECT_AFTER:
            return
        # A real prompt is already waiting; leave it alone and keep counting.
        if _pending["prompt"] or _pending["stop"]:
            return
        last = _chatter["last"]
    # Checked outside the lock, and before the counter is reset: if the bot has
    # just spoken, the run keeps counting rather than being spent on a line it
    # is not allowed to say.
    if not _may_speak_unprompted():
        return
    with _prompt_lock:
        _chatter["count"] = 0

    prompt = _queue_interjection(last)
    action(f"[AI] Interjecting after {IDLE_INTERJECT_AFTER} unaddressed lines: "
          f"{prompt}")


def _resolve_prompt(sender: str, message: str,
                    private: bool = False) -> tuple[str, str] | None:
    """Return (mode, prompt) this message carries for the bot, else None.

    A follow-up inside the conversation window is chat unless it names a mode
    prefix of its own, so one factcheck does not make the whole conversation
    factual.

    A private message needs no trigger at all: opening a query window with the
    bot is not something anybody does by accident, and requiring "sloppy:" in
    a conversation of two would be absurd. It still goes through the trigger
    parser first, so a mode named in one ("!factcheck X") is honoured.
    """
    matched = _match_trigger(message)
    if matched is not None:
        # Explicitly addressed: never rate-limited, and never blocked by the
        # bot having spoken last. A direct question always gets an answer.
        # This is also the only thing that refills the follow-up budget.
        with _prompt_lock:
            _conversation["budget"] = FOLLOWUP_MAX_REPLIES
        return matched

    text = message.strip()
    if not _has_words(text):
        return None
    if private:
        return MODE_CHAT, text
    in_conversation = _in_conversation_with(sender)
    mentioned = MENTION_ENABLED and _mentions_bot(text)

    # A mention from somebody already engaged is as addressed as a leading
    # nick, so it is answered on the same terms: no rate limit, and the
    # follow-up budget refilled. _match_trigger has already taken the leading
    # and trailing forms, so anything reaching here is mid-sentence.
    if mentioned and _mention_is_certain(sender):
        with _prompt_lock:
            _conversation["budget"] = FOLLOWUP_MAX_REPLIES
        return MODE_CHAT, text

    # Nothing below is a direct question, so all of it waits its turn behind
    # the unprompted-speech guards.
    if not _may_speak_unprompted():
        return None

    with _prompt_lock:
        if time.monotonic() < _open_floor["deadline"]:
            # Floor is open: anything anyone says counts, up to the budget. The
            # budget covers follow-ups too, otherwise the first person to reply
            # lands in a 25s conversation and escapes the cap entirely.
            if _open_floor["used"] >= OPEN_FLOOR_MAX_PROMPTS:
                return None
            _open_floor["used"] += 1
            return MODE_CHAT, text

    # A mention from somebody the bot is not already talking to: worth a line
    # some of the time. The guards above already cap how often that can land,
    # so the chance sets the flavour rather than the volume.
    if mentioned and random.random() < MENTION_REPLY_CHANCE:
        return MODE_CHAT, text

    with _prompt_lock:
        if not in_conversation:
            return None
        # An untriggered follow-up spends from the window's budget, which only
        # a real trigger refills -- otherwise replying would top it up again
        # and the window would never close.
        if _conversation["budget"] <= 0:
            return None
        _conversation["budget"] -= 1
    return MODE_CHAT, text


PURGE_TRIGGERS = ("!purge", "!scrub")
# How many of the retained lines a rebuilt summary is made from. The whole log
# would be tens of thousands of tokens; this is the same order as the buffer
# the summarizer normally works through.
PURGE_REBUILD_LINES = _tune("PURGE_REBUILD_LINES", "owners.rebuild_lines", 200)
_PURGE_RE = re.compile(
    r"(?i)^\s*(?:" + "|".join(PURGE_TRIGGERS) + r")\s+([^\s]+)(?:\s+(\d+)\s*d?a?y?s?)?\s*$"
)


def _match_purge_command(message: str) -> tuple[str, float | None] | None:
    """Return (nick, since) for a "!purge <nick> [days]" line, else None.

    `since` is None for "everything", or a wall-clock cutoff that many days
    back. The day count is optional because the common case after an injection
    is "all of it", and the window is for the other case: somebody who has been
    in the channel for a year and had one bad afternoon.
    """
    match = _PURGE_RE.match(message.strip())
    if match is None:
        return None
    days = match.group(2)
    return match.group(1), (time.time() - int(days) * 86400) if days else None


def _rebuild_summary_after_purge() -> bool:
    """Re-derive the rolling summary from the lines that are left.

    The summary is prose, so a purged line cannot be cut out of it -- but the
    log it was made from is on disk, so the summary can simply be made again
    without them. Before the log existed the only honest answer was that
    anything in there ages out on its own, which is no answer at all when what
    is in there was put there deliberately.

    Returns whether a new summary was written. A failed call leaves the old one
    in place, which is the wrong answer after a purge, so the caller says so
    rather than reporting a clean sweep.
    """
    lines = _recall_store.lines_since(limit=PURGE_REBUILD_LINES)
    if not lines:
        with _prompt_lock:
            _rolling["summary"] = ""
            _rolling["highlights"] = []
            _rolling["at"] = time.time()
            _memory_dirty["on"] = True
        _save_memory()
        return True
    summary, highlights, ok = summarizer.summarize_tick_checked("", [], lines)
    if not ok or _reject_reason(summary) is not None:
        return False
    with _prompt_lock:
        _rolling["summary"] = summary
        _rolling["highlights"] = highlights
        _rolling["at"] = time.time()
        _pending_summary_lines.clear()
        _memory_dirty["on"] = True
    _save_memory()
    return True


def _handle_purge(sock: socket.socket, req: Request, nick: str,
                  since: float | None) -> None:
    """Erase `nick` from everything the bot remembers, and say what went.

    Owner-only, because it destroys other people's data and because the reason
    to reach for it -- something planted in the bot's memory on purpose -- is
    exactly the reason not to let whoever planted it call the command.

    The rolling summary is rebuilt from the lines that remain rather than left
    to age out, since that is where a planted line does its work: it rides in
    the system message of every later reply.
    """
    sender, reply_to = req.sender, req.reply_to
    if not req.owner:
        warning(f"[AI] {sender} tried to purge {nick} and is not an owner")
        send(sock, f"PRIVMSG {reply_to} :That one's for owners.")
        return
    window = ("everything" if since is None
              else f"the last {_fmt_span(time.time() - since)}")
    with _prompt_lock:
        profile = _profile_store.get(nick)
        aliases = set(profile["aliases"]) if profile else {nick.lower()}
        had_profile = _profile_store.forget(nick, since)
        recent = _forget_recent_locked(aliases, since)
        _profiles_dirty["on"] = True
    logged = _forget_logged(aliases, since)
    _save_profiles_if_due(force=True)
    action(f"[AI] {sender} purged {nick} ({window}): "
           f"{logged} logged, {recent} recent, profile={had_profile}")
    rebuilt = _rebuild_summary_after_purge()
    send(sock, f"PRIVMSG {reply_to} :Purged {nick} ({window}): "
               f"{_plural(logged, 'line')} from the log, "
               f"{_plural(recent, 'line')} from recent chat"
               f"{', and their profile' if had_profile else ''}. "
               + ("Summary rebuilt from what's left."
                  if rebuilt else
                  "COULDN'T rebuild the summary -- the old one still stands."))


IGNORE_TRIGGERS = ("!ignore", "!unignore", "!ignored")
# Longest first, so "!ignored" is not read as "!ignore" with a stray d.
_IGNORE_RE = re.compile(
    r"(?i)^\s*(?:"
    + "|".join(re.escape(t) for t in sorted(IGNORE_TRIGGERS, key=len, reverse=True))
    + r")\b(?:\s+([^\s]+))?\s*$"
)


def _match_ignore_command(message: str) -> tuple[str, str] | None:
    """Return (verb, target) for an !ignore/!unignore/!ignored line, else None.

    `target` is "" for !ignored, which takes none -- and for a bare !ignore,
    which is a typo rather than a command and is answered as one.
    """
    text = message.strip()
    match = _IGNORE_RE.match(text)
    if match is None:
        return None
    verb = text.lstrip().split()[0].lstrip("!").lower()
    return verb, (match.group(1) or "")


def _add_ignore(target: str) -> bool:
    """Add `target` to the runtime ignore list. False when it was already on
    it, by the configured masks or by an earlier !ignore."""
    with _prompt_lock:
        known = {p.lower() for p in IGNORE_MASKS + _ignored_live}
        if target.lower() in known:
            return False
        _ignored_live.append(target)
    _save_ignores()
    return True


def _drop_ignore(target: str) -> str:
    """Take `target` off the runtime ignore list. Returns "" on success, else
    why not: a configured mask is the config file's to remove, not a
    command's, and saying so beats a command that reports success and changes
    nothing at the next reload."""
    with _prompt_lock:
        for existing in list(_ignored_live):
            if existing.lower() == target.lower():
                _ignored_live.remove(existing)
                break
        else:
            configured = any(p.lower() == target.lower() for p in IGNORE_MASKS)
            return ("that one is in the config file; take it out of "
                    "[ignore] masks and reload" if configured
                    else "I'm not ignoring them")
    _save_ignores()
    return ""


def _ignore_report() -> str:
    """One line saying who is ignored and where each entry came from."""
    with _prompt_lock:
        configured, live = list(IGNORE_MASKS), list(_ignored_live)
    if not configured and not live:
        return "Not ignoring anybody."
    parts = [f"{m} (config)" for m in configured] + [f"{m} (live)" for m in live]
    return "Ignoring: " + ", ".join(parts)


def _handle_ignore(sock: socket.socket, req: Request, verb: str,
                   target: str) -> None:
    """Answer !ignore / !unignore / !ignored. Owner-only, like !purge.

    Silencing somebody for the whole channel is the same kind of power as
    erasing them from its memory, and for the same reason it must not be
    reachable by whoever is being silenced.
    """
    sender, reply_to = req.sender, req.reply_to
    if not req.owner:
        warning(f"[AI] {sender} tried !{verb} and is not an owner")
        send(sock, f"PRIVMSG {reply_to} :That one's for owners.")
        return
    if verb == "ignored":
        send(sock, f"PRIVMSG {reply_to} :{_truncate_for_irc(_ignore_report())}")
        return
    if not target:
        send(sock, f"PRIVMSG {reply_to} :!{verb} who?")
        return
    if verb == "ignore":
        if _names_an_owner(target):
            send(sock, f"PRIVMSG {reply_to} :Owners can't be ignored.")
            return
        added = _add_ignore(target)
        action(f"[AI] {sender} ignored {target}"
               if added else f"[AI] {sender} re-ignored {target}")
        send(sock, f"PRIVMSG {reply_to} :"
                   + (f"Ignoring {target}." if added
                      else f"Already ignoring {target}."))
        return
    problem = _drop_ignore(target)
    if problem:
        send(sock, f"PRIVMSG {reply_to} :Can't un-ignore {target}: {problem}.")
        return
    action(f"[AI] {sender} un-ignored {target}")
    send(sock, f"PRIVMSG {reply_to} :Listening to {target} again.")


def _handle_immediate_command(sock: socket.socket, message: str,
                              req: Request) -> bool:
    """Answer the commands that need no LLM call. True if one was handled.

    Split out of _handle_ai_prompt to keep it under the statement ceiling;
    these all share the property of being answered straight from the receiver
    thread rather than going on the pending queue.
    """
    sender, reply_to = req.sender, req.reply_to
    purge = _match_purge_command(message)
    if purge is not None:
        _handle_purge(sock, req, *purge)
        return True

    ignore = _match_ignore_command(message)
    if ignore is not None:
        _handle_ignore(sock, req, *ignore)
        return True

    mood = _match_mood_command(sender, message)
    if mood is not None:
        # Acked straight from the receiver thread (as PONG already is) rather
        # than queued: the ack must not displace a prompt that is waiting, and
        # a mode switch that lands two seconds later reads as a bug.
        _set_mood(mood)
        send(sock, f"PRIVMSG {reply_to} :{MOOD_REPLIES[mood]}")
        action(f"[AI] {sender} switched the mood to {mood}")
        return True

    if _match_help_command(message):
        for line in _help_lines():
            send(sock, f"PRIVMSG {reply_to} :{_truncate_for_irc(line)}")
        action(f"[AI] {sender} asked for the command list")
        return True

    # Checked before the prompt is resolved, so "sloppy: forget about me" is a
    # command rather than something the model is asked to have an opinion on.
    privacy = _match_privacy_command(message)
    if privacy is not None:
        _handle_privacy_command(sock, sender, privacy, reply_to)
        return True

    # Image analysis is on demand and needs a vision model. Checked before
    # ordinary resolution so a referential request ("sloppy, what's in the
    # image Tim just posted") is treated as an image request first.
    vision = _match_vision_trigger(message)
    if vision is not None:
        if not _vision_active():
            # No vision model in service, so say so rather than answering blind.
            send(sock, f"PRIVMSG {reply_to} :[AI] I can't see images right now "
                       "(no vision model loaded).")
            return True
        url, prompt, mode = vision
        # The LLM server fetches this URL itself, from wherever it runs, and
        # the bot then reads out what it saw. Unchecked, that is a stranger in
        # the channel using the bot as a window into the network the model
        # sits on -- the same hole web.fetch already refuses for !summarize,
        # reached by a path that was not going through it.
        refusal = web.check_url(url)
        if refusal:
            send(sock, f"PRIVMSG {reply_to} :Not fetching that one: {refusal}")
            warning(f"[AI] refused an image URL from {sender}: {url} ({refusal})")
            return True
        _queue_vision(url, sender, prompt, reply_to)
        _note_conversation(sender)
        _reset_chatter()
        action(f"[AI] Captured image request from {sender}: {url}")
        return True

    if WEB_ENABLED:
        page_url = _match_summarize_trigger(message)
        if page_url is not None:
            _queue_page(page_url, sender, reply_to)
            _note_conversation(sender)
            _reset_chatter()
            action(f"[AI] captured a page request from {sender}: {page_url}")
            return True
    return False


def _handle_ai_prompt(sock: socket.socket, message: str, req: Request) -> bool:
    """Capture a message meant for the bot. Returns True if it was ours.

    `req.reply_to` is where the answer goes -- the channel by default, or the
    sender's nick for a private message, so an answer to something said in
    private is not repeated to the room. `req.private` means the message was
    said to the bot directly, which is address enough on its own: nobody opens
    a query window to talk to somebody else.
    """
    sender, reply_to = req.sender, req.reply_to or CHANNEL
    _note_activity()

    if _handle_immediate_command(sock, message, req._replace(reply_to=reply_to)):
        return True

    matched = _resolve_prompt(sender, message, private=req.private)
    if matched is None:
        return False
    mode, prompt = matched

    if _is_shutup(prompt):
        with _prompt_lock:
            _pending["prompt"] = ""
            _pending["sender"] = sender
            _pending["stop"] = True
            _pending["reply_to"] = reply_to
        action(f"[AI] {sender} told us to shut up")
        _reset_chatter()
        _close_open_floor()
        return True

    with _prompt_lock:
        _pending["prompt"] = prompt
        _pending["sender"] = sender
        _pending["stop"] = False
        _pending["mode"] = mode
        _pending["reply_to"] = reply_to
    _note_conversation(sender)
    _reset_chatter()

    action(f"[AI] Captured prompt from {sender}: {prompt}")
    return True


def _queue_vision(url: str, sender: str, prompt: str,
                  reply_to: str = "") -> None:
    """Queue an on-demand image-analysis request for the vision worker."""
    with _prompt_lock:
        _pending_vision["url"] = url
        _pending_vision["sender"] = sender
        _pending_vision["prompt"] = prompt
        _pending_vision["reply_to"] = reply_to


def _take_pending_vision() -> tuple[str, str, str, str] | None:
    """Retrieve and clear the queued image URL, sender, prompt and target."""
    with _prompt_lock:
        if not _pending_vision["url"]:
            return None
        url = _pending_vision["url"]
        sender = _pending_vision["sender"]
        prompt = _pending_vision["prompt"]
        reply_to = _pending_vision["reply_to"] or CHANNEL
        _pending_vision["url"] = ""
        _pending_vision["sender"] = ""
        _pending_vision["prompt"] = ""
        _pending_vision["reply_to"] = ""
        return url, sender, prompt, reply_to


def _handle_line(sock: socket.socket, line: str) -> bool:
    """Handle one received IRC line (not PING). Returns True when it was
    ordinary chatter the receiver should still log, else False (handled)."""
    # One check for every kind of line, at the one point they all pass: an
    # ignored nick is not answered, not remembered, not counted and not
    # greeted, and a JOIN or a private message is no different.
    mask = _event_mask(line)
    if mask and _is_ignored(mask):
        debug(f"ignored: {line}")
        return False
    nick, command = _split_event(line)
    if command == "JOIN" and nick:
        irc(f"< {line}")
        _handle_join(sock, nick)
        return False
    if command in ("QUIT", "PART") and nick:
        irc(f"< {line}")
        _handle_quit(nick)
        return False
    if command == "NICK" and nick:
        irc(f"< {line}")
        renamed = _parse_nick_change(line)
        if renamed:
            _handle_nick_change(*renamed)
        return False
    if _handle_info_line(line):
        return False
    if " PRIVMSG " not in line:
        return True
    parsed = _parse_privmsg(line)
    if not parsed:
        return True
    if parsed.private:
        return _handle_private(sock, parsed)
    _note_recent(parsed.text, parsed.sender)
    request = Request(parsed.sender, CHANNEL, owner=_is_owner(parsed.mask))
    if _handle_ai_prompt(sock, parsed.text, request):
        return False
    irc(f"< {line}")
    _note_chatter(parsed.text)
    return True


def _handle_private(sock: socket.socket, msg: Privmsg) -> bool:
    """Answer a private message from an owner; ignore one from anybody else.

    A query window is a channel of one that nobody else can see, which makes it
    the wrong place to take instructions from strangers -- and with no owners
    configured that is everybody, so the default is to answer nobody.

    Nothing said in private reaches the channel's memory: not the recent-line
    buffer, not the summary, not the long-term log, not the chatter counter.
    Otherwise a private word would come back out of the bot's mouth in the
    room, which is the opposite of what saying it privately meant. It is also
    why the reply goes back to the sender rather than to the channel.
    """
    if not _is_owner(msg.mask):
        # Logged, not answered: whoever is watching the TUI should know
        # somebody tried, and the sender should learn nothing.
        warning(f"[AI] ignored a private message from {msg.mask}")
        return False
    if not msg.sender:
        # There is nobody to answer, and the one thing a private message must
        # never do is fall through to the channel -- which is exactly what an
        # empty reply target does further down (see _take_pending).
        warning(f"[AI] private message with no sender, dropped: {msg.mask!r}")
        return False
    irc(f"< (private) {msg.sender}: {msg.text}")
    _note_activity()
    _handle_ai_prompt(sock, msg.text,
                      Request(msg.sender, msg.sender, private=True, owner=True))
    return False


def receiver(sock: socket.socket, gone: threading.Event | None = None) -> None:
    """Read the socket until it dies, dispatching each complete line.

    `gone` is the session's disconnect flag: set on the way out so main()'s
    poll loop stops using a socket that is no longer connected and reconnects.
    """
    buffer = ""
    while True:
        try:
            data = sock.recv(4096).decode("utf-8")
            if not data:
                action("Server closed connection.")
                break
            buffer += data
            lines = buffer.split("\r\n")
            buffer = lines[-1]  # keep incomplete trailing fragment
            for raw in lines[:-1]:
                line = raw.strip()
                if line.startswith("PING "):
                    send(sock, line.replace("PING", "PONG", 1))
                elif line and _handle_line(sock, line):
                    irc(f"< {line}")
        except Exception as e:
            warning(f"Receiver error: {e}")
            break
    # Whatever ended the loop, the link is gone. Wake the poll loop.
    if gone is not None:
        gone.set()


def _take_pending() -> tuple[str, str, bool, str, str]:
    """Retrieve and clear the pending prompt, sender, stop flag, mode, target."""
    with _prompt_lock:
        prompt = _pending["prompt"]
        sender = _pending["sender"]
        stop = _pending["stop"]
        mode = _pending["mode"]
        reply_to = _pending["reply_to"] or CHANNEL
        _pending["prompt"] = ""
        _pending["stop"] = False
        _pending["reply_to"] = ""
        return prompt, sender, stop, mode, reply_to


def get_pending_prompt() -> str:
    """Retrieve and clear the pending AI prompt."""
    return _take_pending()[0]


def _request_userlist(sock: socket.socket) -> None:
    """Ask the server who is in the channel, right after joining."""
    send(sock, f"WHO {CHANNEL}")


# The modes that answer ABOUT something rather than INTO the room. They get the
# persona alone: nobody asking what a page says needs to be told who is in the
# channel, and the mention list once turned a page summary into "probe alice,
# the page is...". Written as the exceptions rather than as the rule because a
# mood in the config may name a persona of any name at all (MOOD_MODES), and
# those all answer into the room.
ABOUT_THE_WORLD_MODES = frozenset({
    MODE_FACTUAL, MODE_SCIENCE, MODE_RESEARCH, MODE_ANSWER,
    MODE_WEBPAGE, MODE_TRANSLATE, MODE_QUOTE, MODE_BUDDHA, MODE_FACTOID,
})


# Asking what somebody said, or when they said it, rather than talking about
# now. Two things follow: the search is scoped to the person named, and the
# recency prior comes off -- the answer to "when did i say i was going to
# amsterdam" is as old as it is, and at any half-life a months-old line decays
# under the relevance floor before it can be found.
# Written from the phrasings people actually use, not from invented ones. The
# first version of this was guessed and missed 5 of 7 real examples off the
# channel -- "whats your earliest memory of dflatline", "what do you remember
# about probe", "where did spacec0wboy go" -- so the whole feature sat behind a
# pattern that never fired.
#
# "how did" and "why did" are deliberately absent. They read as questions about
# the past and are usually follow-ups about the conversation ("how did that
# go"), and the historical path drops the conversation search, which is the one
# thing a follow-up with no content of its own has to go on.
_ABOUT_THE_PAST_RE = re.compile(
    r"\b(?:"
    r"when did|what did|who said|where did"
    r"|did (?:i|you|we|he|she|they|[\w\[\]{}`^|\\-]+) (?:ever )?say"
    r"|do you remember|you remember"
    r"|remember (?:when|what|who|where|how|why|that|the|about|anything|any)"
    r"|(?:earliest|oldest|first|fondest|favourite|favorite) (?:memory|memories|thing)"
    r"|memory of|memories of"
    r"|what do you (?:know|remember)"
    r"|used to|ever said|ever told|first time|last time"
    r"|last (?:week|month|year|night)|the other (?:day|week|night)"
    r"|(?:days|weeks|months|years) ago|a (?:while|bit) (?:ago|back)|ages ago"
    r"|back (?:then|in|when)|way back"
    r")\b",
    re.IGNORECASE,
)
# Asking for the FIRST one rather than the best match. Retrieval ranks by
# relevance, so without this "your earliest memory of dflatline" returned
# whatever of dflatline's scored highest, which is not what was asked.
_THE_EARLIEST_RE = re.compile(
    r"\b(?:earliest|oldest|first|furthest back)\b[^.?!]{0,30}?"
    r"\b(?:memory|memories|thing|time|said|say|told)\b"
    r"|\bfirst thing\b|\bfurthest back\b",
    re.IGNORECASE,
)
# First person in a question is the person asking: "when did I say I was going
# to amsterdam" is a question about their own log, not about nobody.
_FIRST_PERSON_RE = re.compile(r"\b(?:i|me|my|mine|myself)\b", re.IGNORECASE)


def _asks_about_the_past(text: str) -> bool:
    """True when `text` is asking what was said rather than talking now."""
    return bool(_ABOUT_THE_PAST_RE.search(text))


def _asks_for_the_earliest(text: str) -> bool:
    """True when `text` wants the FIRST thing, not the most relevant thing."""
    return bool(_THE_EARLIEST_RE.search(text))


def _recall_subjects(text: str, asker: str) -> list:
    """Whose log a question is about: the people it names, plus the asker
    when it speaks in the first person.

    "what did alice say about her boyfriend" is alice's log; "when did i say i
    was going to amsterdam" is the asker's own. Only consulted for a question
    about the past -- in ordinary chat a name is just a name, and scoping every
    mention to that person's back catalogue would bury the actual topic.
    """
    subjects = _named_others(text, asker)
    if asker and _FIRST_PERSON_RE.search(_strip_nick_prefix(text)):
        subjects.append(asker)
    return subjects


def _named_others(text: str, asker: str = "") -> list:
    """The channel nicks `text` mentions, other than the asker and the bot.

    Asking about somebody else is ordinary channel traffic -- "what do you
    think about probe", "is probe fat" -- and those are the lines where a
    contextual answer is the whole joke. Matched against the roster and the
    people in the recent-line buffer rather than against the profile store,
    which holds everybody ever seen: a nick that is also an ordinary word
    would otherwise drag a stranger's profile in on a false match.

    The asker is excluded because their own nick is in every line now that the
    question is attributed ("alice: what do you think"); they are who is
    asking, not who is being asked about.

    Short nicks are skipped. A nick that is also an ordinary word is a hazard
    this codebase has already been bitten by once -- "nice" and "kind" used to
    switch the mood every time somebody was polite -- and the cheap half of
    that guard is a length floor. It is not the whole guard: a three-letter
    nick that is also a word will still match, and the cost is one wrong
    profile in the prompt rather than a wrong reply.
    """
    words = set(re.findall(r"[\w\[\]{}`^|\\-]+", text.lower()))
    skip = {NICK.lower(), asker.lower()}
    with _prompt_lock:
        candidates = list(_users["names"]) + list(_recent_senders)
    seen, found = set(), []
    for nick in candidates:
        low = nick.lower()
        if len(low) < ABOUT_MIN_NICK_CHARS:
            continue
        if low in words and low not in skip and low not in seen:
            seen.add(low)
            found.append(nick)
    return found


def _involved(mode: str, asker: str, about: list) -> list:
    """Everyone whose own words belong in this reply's prompt.

    Whoever was asked about, and the person asking. The asker goes LAST so a
    question about somebody else spends the nick budget on them: being told
    about probe is the point of "what do you think about probe", and knowing
    the asker is background.
    """
    if mode in ABOUT_THE_WORLD_MODES or not asker:
        return list(about)
    return [*about, asker]


def _about_section(nicks: list) -> str:
    """What the people this reply involves have actually said, or "".

    The same per-person store a roast-flavoured greeting uses (see
    _profile_recall) and for the same reason: a line about somebody is only
    worth reading if it is about THEM. Without it the model invents the person
    it is answering about -- measured against the live model with probe's lines
    aged out of the recent buffer, 3 of 12 replies used anything he had really
    said, and the rest made him up ("probe is a good dog", "he's just a wrapper
    around a rest api", and a "she" for good measure).

    The person ASKING is in here too, on every direct reply. The bot is given
    what it knows about whoever it is answering for the same reason it is given
    it about whoever they asked about: a line lands on somebody when it is
    about them, and otherwise it is a line that would fit anybody.
    """
    blocks = []
    for nick in nicks[:ABOUT_PROFILE_NICKS + 1]:
        said = _profile_recall(nick, ABOUT_PROFILE_LINES)
        if said:
            blocks.append(f"{nick} has said:\n{said}")
    if not blocks:
        return ""
    return ("--- THE PEOPLE IN THIS EXCHANGE, IN THEIR OWN WORDS ---\n"
            + "\n\n".join(blocks))


def _addressing_section(mode: str, asker: str, about: list = (),
                        historical: bool = False) -> str:
    """Who the bot is answering and who else is in the room, or "".

    `asker` is the person whose message this reply is for, empty for the
    unprompted lines nobody asked for. Naming them is the whole point: every
    other line the model reads is attributed "nick: text", and the one line it
    is meant to answer used to arrive anonymous, marked only by happening to be
    last in the chat block. It stops being last the moment anybody else types,
    which in a channel is immediately -- measured against the live model, 1/19
    replies named somebody other than the asker while the question was still
    the last line, and 5/20 once two other people had spoken after it.

    It lives at the BOTTOM of the prompt, in the NOW section, because that is
    where the rest of the current turn is: it describes this message, not the
    room in general, and it reads next to the chat it refers to rather than
    three thousand characters above it.

    NOT for prompt-cache reasons, though it looks like it should be. As the
    tail of the persona it was the first thing to change when a different
    person spoke, at byte 3706 of a 3955-byte system prompt, which by c196cc3's
    reasoning should have thrown away everything below it. Measured against the
    live server at the real context_lines=50 with a full buffer, it makes no
    difference at all: old 51% of the prompt reused and 1543 tokens prefilled,
    new 52% and 1555. The recent-chat block is a sliding window, so a new line
    rewrites all fifty of them and invalidates that region every message
    regardless of what sits above it. The saving only appears on a buffer too
    short to have started sliding.
    """
    if mode in ABOUT_THE_WORLD_MODES:
        return ""
    targets = _mention_targets()
    if not asker:
        # Nobody addressed the bot, so there is no one person to answer.
        if not targets:
            return ""
        return ("The people in this IRC channel are named: "
                + ", ".join(targets)
                + ". Prefer to mention the first one most often (whoever "
                "spoke most recently); then someone else who spoke recently; "
                "and only occasionally someone further down the list. Address "
                "or mention people about 50% of the time.")
    others = [nick for nick in targets if nick != asker]
    said = (f"{asker} is the one talking to you, so answer {asker} and not "
            f"somebody else who has spoken since.")
    if about:
        # The other half of the same problem. Answering the right person is no
        # good if the answer is about nobody: asked about probe, the reply is
        # about probe, and the block above says what probe has actually said.
        names = " and ".join(about)
        said += (f" {asker} is asking about {names}, so make it about {names}: "
                 f"use what they have really said and done rather than "
                 f"anything you assume about them, and name them.")
    if historical:
        # Retrieval can put the right lines in front of it and the model will
        # still banter past them -- reported live, and measured: asked for the
        # earliest memory of somebody it was handed that person's first line
        # and answered with an invented one instead. Being told what to answer
        # FROM is the other half of the job, and saying "I do not remember" has
        # to be an available answer or the gap just gets filled.
        said += (f" {asker} is asking about something from earlier, not about "
                 f"now. Answer from the lines under EARLIER IN THE CHANNEL: "
                 f"say what was actually said and roughly when. If it is not "
                 f"there, say you do not remember it -- do not invent one.")
    elif others:
        # Named so the bot can spell them, not as an invitation: without a
        # question about somebody, wandering off to another name is the bug
        # this section exists for.
        said += (" The other people here are " + ", ".join(others)
                 + "; bring one of them up only if it is genuinely about them.")
    return said


# A leading "nick:" on a line. Deliberately loose about the nick charset --
# IRC allows []{}`^|\- and digits -- because the name is checked against the
# people actually seen in the channel, not against the pattern.
_NICK_PREFIX_RE = re.compile(r"^\s*([\w\[\]{}`^|\\-]{1,20})\s*:")
# The same thing anywhere in the text, not just at the start of a line. The
# model recites the context block back without ever emitting a newline, so
# counting line starts alone sees one line and lets the whole dump through.
_NICK_ANYWHERE_RE = re.compile(r"(?:^|\s)([\w\[\]{}`^|\\-]{1,20})\s*:")


def _looks_like_transcript(text: str) -> bool:
    """True when `text` is more chat transcript rather than a reply.

    The context block hands the model the recent chat as "nick: text" lines, and
    a transcript in that shape invites it to write the next line of one instead
    of answering. Measured at 3-17% of replies depending on the prompt, and the
    failure runs from inventing dialogue for other people to echoing the whole
    context block back into the channel verbatim.

    A single "nick:" is NOT this: addressing somebody by name is ordinary IRC
    and the persona asks for it. Two or more, carrying the names of people
    actually in the room, is. Nicks are matched against the roster and everyone
    in the recent-line buffer, so a dump quoting somebody who has since left is
    still caught -- and counted wherever they appear, because the dumps arrive
    on one unbroken line as often as on several.
    """
    known = _known_nicks()
    hits = sum(
        1 for nick in _NICK_ANYWHERE_RE.findall(text) if nick.lower() in known
    )
    return hits >= TRANSCRIPT_NICK_LINES


def _known_nicks() -> set:
    """Everyone the model could plausibly be quoting: roster, buffer, and us."""
    with _prompt_lock:
        known = {nick.lower() for nick in _users["names"]}
        known.update(nick.lower() for nick in _recent_senders)
    known.add(NICK.lower())
    return known


def _strip_nick_prefix(text: str) -> str:
    """`text` without a leading "nick:" on its first line.

    The persona says never to open with a nick and a colon, and the model does
    it anyway -- measured at 4/10 replies on address-heavy prompts. Trimmed
    rather than rejected: about half of those are a perfectly good line wearing
    a transcript's clothes, and redrawing every one of them would push a real
    share of replies into the two-strikes failure line. The rest are echoes,
    and _echoes_recent still catches them once the prefix is out of the way.

    Only the first line is touched. A second nick line means the model wrote
    somebody else's dialogue, which is _looks_like_transcript's business.
    """
    head, sep, rest = text.partition("\n")
    match = _NICK_PREFIX_RE.match(head)
    if not match or match.group(1).lower() not in _known_nicks():
        return text
    trimmed = head[match.end():].lstrip()
    # A bare "bob:" with nothing after it would strip to nothing at all.
    if not trimmed:
        return text
    return trimmed + sep + rest


def _words(text: str) -> list:
    """`text` as bare lowercase words, so two lines can be compared as said."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _shared_run(a: list, b: list) -> int:
    """The longest run of words `a` and `b` have in common, verbatim."""
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            n = 0
            while i + n < len(a) and j + n < len(b) and a[i + n] == b[j + n]:
                n += 1
            best = max(best, n)
    return best


def _echoes_recent(text: str) -> bool:
    """True when `text` is just saying a recent channel line back.

    The unprompted path hands the model the line it is reacting to as the user
    message, and that same line is already in the context block as
    "nick: text". Given it twice, the model sometimes returns it -- verbatim,
    with the speaker's nick on the front. Measured against the live model at
    2/12 interjections on one prompt and 4/10 on another; it reads as the bot
    parroting whoever spoke last, which is funny exactly once.

    Two ways to be an echo: the reply IS a recent line once punctuation and
    case are set aside, or it carries a verbatim run of ECHO_RUN_WORDS words
    from one. The run bar is set well clear of legitimate reuse -- across ~70
    sampled replies that picked up the subject of the line, the longest honest
    run was three words.
    """
    reply = _words(_strip_nick_prefix(text))
    if not reply:
        return False
    with _prompt_lock:
        recent = list(_recent_lines)
    for line in recent:
        source = _words(line)
        if not source:
            continue
        if reply == source or _shared_run(reply, source) >= ECHO_RUN_WORDS:
            return True
    return False


def _mood_temperature(mode: str) -> float:
    """The temperature for `mode`, letting the mood in force have its say.

    A mood's temperature applies only when the mood is the reason the bot is
    answering in this mode -- `MOOD_MODES[mood] == mode`. So a factcheck mood
    sets the temperature for the chat it colours, while an explicit
    `!factcheck` in banter mood does not inherit it: that one asked for a
    fact-check, not for the mood the channel happens to be in.
    """
    mood = _current_mood()
    if MOOD_MODES.get(mood) == mode and mood in MOOD_TEMPERATURES:
        return MOOD_TEMPERATURES[mood]
    return LLM_TEMPERATURE


def _sampling_for(mode: str) -> tuple[float, dict]:
    """The temperature and extra-body sampling keys to send for `mode`.

    A strict mode (STRICT_MODES) layers [strict_sampling] over [sampling];
    everything else gets [sampling] as configured, with the mood in force
    allowed to set the temperature (see _mood_temperature). Strict wins over a
    mood: those modes answer about the world, and a mood is a register rather
    than a licence to be less accurate.

    temperature comes back separately because the client takes it as its own
    argument and it must not also ride along in the body.
    """
    overrides = STRICT_SAMPLING if mode in STRICT_MODES else {}
    temperature = overrides.get("temperature", _mood_temperature(mode))
    body = {
        **LLM_EXTRA_BODY,
        **SAMPLING,
        **{k: v for k, v in overrides.items() if k != "temperature"},
    }
    return temperature, body


def _generate(messages: list, mode: str = MODE_CHAT) -> str:
    """One completion from the chat model, or EmptyLLMReply if it said nothing.

    The single place the client is called, so the text and vision paths cannot
    drift apart on sampling parameters. `mode` only selects those parameters
    (see _sampling_for); the prompt it produced is already in `messages`.
    """
    temperature, extra_body = _sampling_for(mode)
    response = _llm_client.chat.completions.create(
        model=_model_alias(),
        messages=messages,
        max_tokens=LLM_MAX_TOKENS,
        temperature=temperature,
        extra_body=extra_body,
    )
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise EmptyLLMReply(
            f"model returned no answer text (finish_reason={choice.finish_reason})"
        )
    return text


# What a redraw is told about the draft that was just thrown away. Keyed by the
# reason so the model is corrected on the thing it actually did.
_RETRY_NUDGE = {
    "transcript": "Your last attempt wrote lines of chat transcript for other "
                  "people. Do not write anybody else's lines and do not start "
                  "a line with a nick and a colon. Say one thing, as yourself.",
    "echo": "Your last attempt just said back what somebody else in the "
            "channel had already said. Do not repeat or quote their line. "
            "Say something of your own about it.",
}


def _with_retry_nudge(messages: list, reason: str) -> list:
    """`messages` again, with the redraw's correction on the user turn.

    A redraw used to be the identical request, so a model that had fallen into
    writing transcript had nothing pushing it back out and often did it twice
    running -- which costs the room the reply entirely. The correction goes on
    the user turn rather than as a second system message, because the chat
    templates that matter refuse a system message that is not the first.
    """
    nudge = _RETRY_NUDGE[reason]
    *head, last = messages
    content = last["content"]
    if isinstance(content, str):
        amended = f"{content}\n\n{nudge}"
    else:
        # The vision path: the text part carries it, the image is left alone.
        amended = [
            {**part, "text": f"{part['text']}\n\n{nudge}"}
            if part.get("type") == "text" else part
            for part in content
        ]
    return [*head, {**last, "content": amended}]


def _reject_reason_for(text: str) -> str | None:
    """Why `text` is unusable as a reply, or None if it can be sent."""
    if _looks_like_transcript(text):
        return "transcript"
    if _echoes_recent(text):
        return "echo"
    return None


def _generate_reply(messages: list, mode: str = MODE_CHAT) -> str:
    """A usable reply, retrying a draft that just continued the transcript.

    Rejecting is cheap and a re-draw usually lands, so the room gets a real
    answer instead of the bot reciting its own context back at it. Each redraw
    is told what was wrong with the last one (see _with_retry_nudge). Running
    out of attempts raises, and the caller turns that into a line in character
    plus a red warning in the log pane.
    """
    _DISCARDED = {
        "transcript": "discarded a transcript-shaped reply",
        "echo": "discarded a reply that echoed the channel",
    }
    attempt_messages = messages
    for attempt in range(1, LLM_ATTEMPTS + 1):
        text = _generate(attempt_messages, mode)
        reason = _reject_reason_for(text)
        if reason is None:
            # Not a rejection: a good line that opened with "nick:" anyway.
            return _strip_nick_prefix(text)
        warning(f"[AI] {_DISCARDED[reason]} (attempt {attempt}): "
                f"{' '.join(text.split())[:90]}")
        attempt_messages = _with_retry_nudge(messages, reason)
    raise TranscriptReply(
        f"model continued the chat transcript {LLM_ATTEMPTS} times running"
    )


def _compose_messages(
    system_prompt: str, context_block: list, user_message: dict
) -> list:
    """The messages to send: one system message, then the current event.

    The rolling context is folded into the leading system message rather than
    riding as a second one. Some chat templates -- Qwen3-derived ones among
    them -- refuse any system message that is not the first, and a second one
    makes the server answer 500 "System message must be at the beginning"
    instead of replying. The text the model reads is unchanged: the same
    sections in the same order, separated the same way as the sections inside
    the context block itself.
    """
    system = "\n\n".join(
        [system_prompt, *(m["content"] for m in context_block)]
    )
    return [{"role": "system", "content": system}, user_message]


def _call_llm(prompt: str, mode: str = MODE_CHAT, asker: str = "") -> str:
    """Send prompt to local llama.cpp and return the response text.

    The summarizer's rolling summary, highlights, and a verbatim sample of the
    most recent IRC lines are folded into one system (background/observation)
    message, and the current event rides as the single user message. This
    happens on every mode -- factual included -- because every reply happens
    inside an ongoing room.

    `asker` is who this reply is FOR, and it decides nothing else: whether
    `prompt` carries their name is the caller's business, because only the
    caller knows whether it is what somebody typed or a note the bot wrote
    itself. It is carried explicitly rather than read back out of _conversation
    here, because the receiver thread keeps taking lines while the reply is
    generated -- whoever the room is talking to by the time the model answers
    is not necessarily who asked. Empty for the lines nobody asked for.
    """
    about = ([] if mode in ABOUT_THE_WORLD_MODES
             else _named_others(prompt, asker))
    system_prompt = _system_prompt(mode)
    context_block = (
        [] if mode in CONTEXTLESS_MODES
        else _context_block(
            prompt,
            _addressing_section(mode, asker, about,
                                _asks_about_the_past(prompt)),
            _about_section(_involved(mode, asker, about)), asker)
    )
    if context_block:
        action("Injected rolling summary + highlights + recent chat as context")
    messages = _compose_messages(
        system_prompt, context_block, {"role": "user", "content": prompt}
    )
    debug(f"System prompt:\n{system_prompt}")
    debug(f"User prompt:\n{prompt}")
    text = _generate_reply(messages, mode)
    _record_last_llm_call(messages, text)
    return text


def _call_llm_vision(url: str, prompt: str, asker: str = "") -> str:
    """Send `url` + `prompt` to the vision model and return the description.

    The image rides on the user message as an image_url content part -- the
    system prompt stays text-only, because llama.cpp rejects images there. The
    rolling summary, highlights and recent chat are injected exactly as they
    are for a text reply (see _context_block), because the description is
    spoken into an ongoing room. Uses the shared client, which points at the
    one server that also serves the persona.
    """
    about = _named_others(prompt, asker)
    system_prompt = _system_prompt(MODE_VISION)
    context_block = _context_block(
        prompt, _addressing_section(MODE_VISION, asker, about),
        _about_section(_involved(MODE_VISION, asker, about)), asker)
    if context_block:
        action("Injected rolling summary + highlights + recent chat as context")
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": _attributed(asker, prompt)},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }
    messages = _compose_messages(system_prompt, context_block, user_message)
    debug(f"System prompt:\n{system_prompt}")
    debug(f"User prompt (with image): {prompt} -> {url}")
    text = _generate_reply(messages, MODE_VISION)
    _record_last_llm_call(messages, text)
    return text


def _record_last_llm_call(messages: list, text: str) -> None:
    """Store the full record of this call for the TUI debug view (press 'd').

    Assembled while the call happens so it can be shown on demand, and showing
    exactly what went over the wire: two messages, a system one and a user one.

    It used to print the system message, then repr(messages) -- which contains
    that same system message again -- then the user message a second time as
    well. Nothing was ever sent twice; it only looked that way, and it looked
    that way in the one screen somebody would open to find out. Reported by
    Alexander after reading the summary and the recent chat twice in a prompt
    that contained them once.
    """
    parts = ["=== Last LLM call ===", f"{len(messages)} messages sent"]
    for message in messages:
        content = message["content"]
        if not isinstance(content, str):
            # A vision call: text and an image_url part ride together.
            content = "\n".join(
                part.get("text") or part.get("image_url", {}).get("url", "")
                for part in content
            )
        parts.append(f"[{message['role']}]\n{content}")
    parts.append(f"[output]\n{text}")
    with _prompt_lock:
        _last_llm_call["text"] = "\n\n".join(parts)


def get_last_llm_call() -> str:
    """The formatted record of the most recent LLM call, for the TUI debug view.

    Empty string until the first call, so the UI can always show it.
    """
    with _prompt_lock:
        return _last_llm_call["text"]


def _probe_props() -> bool:
    """Ask the server what it has loaded; return whether it can see images.

    Reads `modalities.vision` and `model_alias` from /props. Any failure
    (server down, wrong endpoint, vision not enabled) is treated as "not
    enabled" rather than raised, so a probe never disrupts the poll loop. Both
    results are cached, and each is announced as a one-line action the first
    time it changes, so a late-loading or swapped model is visible in the log.

    Whether the probe was answered at all is recorded as the LLM's health (see
    _note_llm_health), which is what decides whether the bot sits in the
    channel or waits an outage out somewhere else.
    """
    enabled = False
    alias = ""
    answered = False
    try:
        with urllib.request.urlopen(LLM_PROPS_URL, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        enabled = bool(data.get("modalities", {}).get("vision", False))
        alias = str(data.get("model_alias") or "").strip()
        answered = True
    except Exception as e:
        debug(f"props probe failed: {e}")
    _note_llm_health(answered)
    with _prompt_lock:
        previous = _vision["enabled"]
        _vision["enabled"] = enabled
        # An empty alias means the probe failed or the server did not say;
        # either way keep whatever we had rather than blanking the display.
        renamed = bool(alias) and alias != _model["alias"]
        if alias:
            _model["alias"] = alias
            _model["detected"] = True
    if renamed:
        action(f"[AI] Model in service: {alias}")
    if enabled != previous:
        action(f"[AI] Vision support: {'enabled' if enabled else 'not loaded'}")
    return enabled


def _note_llm_health(ok: bool) -> None:
    """Record whether the endpoint answered, and since when it has not.

    Only the first failure of a run sets the clock: the outage is timed from
    when it started, not from the most recent probe that confirmed it.
    """
    with _prompt_lock:
        was_ok = _llm_health["ok"]
        _llm_health["ok"] = ok
        if ok:
            _llm_health["down_since"] = 0.0
        elif was_ok:
            _llm_health["down_since"] = time.monotonic()
    if ok != was_ok:
        if ok:
            action("[AI] LLM endpoint answering again")
        else:
            warning(f"[AI] LLM endpoint not answering ({LLM_PROPS_URL})")


def _llm_down_for() -> float:
    """Seconds the LLM endpoint has been unreachable; 0.0 while it answers."""
    with _prompt_lock:
        since = _llm_health["down_since"]
    return (time.monotonic() - since) if since else 0.0


def _is_absent() -> bool:
    """Whether the bot has left the channel to wait out an LLM outage."""
    with _prompt_lock:
        return _absent["on"]


def _should_sit_out() -> bool:
    """Whether the outage has run long enough that the bot should not be in the
    channel. Off entirely when LLM_PART_AFTER is 0."""
    return LLM_PART_AFTER > 0 and _llm_down_for() >= LLM_PART_AFTER


def llm_preflight() -> tuple[bool, str]:
    """Is there a model in service? Returns (ok, one line saying what).

    Asks the same /props the poll loop uses, so a pass here means the endpoint
    the bot will actually call is answering -- not merely that something is
    listening on the port.
    """
    try:
        with urllib.request.urlopen(LLM_PROPS_URL, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - every failure is the same answer
        return False, f"no answer from {LLM_PROPS_URL} ({exc})"
    alias = str(data.get("model_alias") or "").strip()
    path = str(data.get("model_path") or "").strip()
    name = alias or path.rsplit("/", 1)[-1]
    if not name:
        return False, f"{LLM_PROPS_URL} answered but named no model"
    vision = data.get("modalities", {}).get("vision", False)
    return True, f"{name} in service{', vision enabled' if vision else ''}"


def llm_preflight_problem() -> str:
    """The problem to put to the user before starting, or "" to go ahead.

    Returns "" when the check passes, is switched off, or is not configured to
    stop anything -- so a caller can treat a non-empty string as "there is
    something worth asking about".
    """
    if LLM_CHECK == "off":
        return ""
    ok, detail = llm_preflight()
    if ok:
        action(f"[AI] {detail}")
        return ""
    warning(f"[AI] no LLM: {detail}")
    return detail


def _model_alias() -> str:
    """The model name to put on a request: what /props reported, else the
    configured fallback."""
    with _prompt_lock:
        return _model["alias"]


def _probe_props_if_due() -> None:
    """Run the /props vision probe at most once per PROPS_PROBE_INTERVAL.

    Whether an mmproj is loaded changes only when the server is restarted, so
    the poll loop does not need to ask every two seconds.
    """
    now = time.monotonic()
    with _prompt_lock:
        due = now - _last_props_probe["t"] >= PROPS_PROBE_INTERVAL
        if due:
            _last_props_probe["t"] = now
    if due:
        _probe_props()


def _vision_active() -> bool:
    """Whether image analysis will actually run right now.

    A manual override (from the TUI) wins; otherwise the last probe result is
    used.
    """
    with _prompt_lock:
        override = _vision["override"]
        enabled = _vision["enabled"]
    return override if override is not None else enabled


def _vision_source() -> str:
    """How vision state is currently decided: 'auto' (probe) or a manual force."""
    with _prompt_lock:
        override = _vision["override"]
    return "auto" if override is None else ("on" if override else "off")


def _set_vision_override(enabled: bool | None) -> str:
    """Set (or clear with None) the manual vision override; return the label."""
    with _prompt_lock:
        _vision["override"] = enabled
    return _vision_source()


def _recall_active() -> bool:
    """Whether a reply will actually get recalled passages right now.

    A manual override (from the TUI) wins; otherwise the config value stands.
    Three states rather than a plain flag, exactly as vision does it: without a
    way back to "whatever the file says", a runtime toggle and a config reload
    disagree about which of them is in charge.
    """
    with _prompt_lock:
        override = _recall["override"]
    return RECALL_ENABLED if override is None else override


def _recall_source() -> str:
    """How recall state is decided: 'config' or a manual force."""
    with _prompt_lock:
        override = _recall["override"]
    return "config" if override is None else "forced"


def _cycle_recall_override() -> str:
    """Cycle the manual override config -> on -> off -> config."""
    with _prompt_lock:
        current = _recall["override"]
        _recall["override"] = (
            True if current is None else (False if current else None)
        )
    return _recall_source()


def _cycle_vision_override() -> str:
    """Cycle the manual override auto -> on -> off -> auto; return the label."""
    with _prompt_lock:
        current = _vision["override"]
    return _set_vision_override(True if current is None else (False if current else None))


# Bytes the wire line spends on framing: "PRIVMSG <chan> :" plus the trailing CRLF.
_IRC_OVERHEAD = len(f"PRIVMSG {CHANNEL} :".encode("utf-8")) + 2


def _truncate_for_irc(text: str) -> str:
    """Truncate text so the resulting PRIVMSG fits in IRC_MAX_LEN bytes."""
    budget = IRC_MAX_LEN - _IRC_OVERHEAD
    if len(text.encode("utf-8")) <= budget:
        return text
    # Cut on a character boundary: shrink until the encoded form fits.
    ellipsis = "…"
    budget -= len(ellipsis.encode("utf-8"))
    cut = text
    while len(cut.encode("utf-8")) > budget:
        cut = cut[:-1]
    return cut + ellipsis


def _format_reply_lines(text: str, max_lines: int | None = None) -> list[str]:
    """Reflow an LLM reply into at most IRC_MAX_REPLY_LINES sendable lines.

    Words are packed to fill each line up to the byte budget rather than
    following the model's own newlines, so a bulleted answer becomes a few full
    lines instead of one PRIVMSG per bullet. If the reply still does not fit,
    the last line ends in an ellipsis.
    """
    budget = IRC_MAX_LEN - _IRC_OVERHEAD
    limit = IRC_MAX_REPLY_LINES if max_lines is None else max_lines
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate.encode("utf-8")) <= budget:
            current = candidate
            continue
        if current:
            lines.append(current)
        if len(lines) == limit:
            # Out of room; mark the last line as truncated.
            lines[-1] = _mark_truncated(lines[-1], budget)
            return lines
        # A single word longer than one line still has to be broken up.
        current = word
        while len(current.encode("utf-8")) > budget:
            lines.append(_truncate_for_irc(current))
            if len(lines) == limit:
                lines[-1] = _mark_truncated(lines[-1], budget)
                return lines
            current = current[len(lines[-1]) - 1:]

    if current:
        lines.append(current)
    return lines[:limit]


def _mark_truncated(line: str, budget: int) -> str:
    """Append an ellipsis to `line`, shrinking it to stay within `budget` bytes."""
    ellipsis = "…"
    room = budget - len(ellipsis.encode("utf-8"))
    while len(line.encode("utf-8")) > room:
        line = line[:-1]
    return line + ellipsis


def _say_brain_offline(sock: socket.socket, detail: str,
                       reply_to: str = "") -> None:
    """Report a failed LLM call: the real error red in the log pane, and a line
    in character to the channel.

    The channel does not want a Python exception, and printing one there breaks
    the persona for everybody to no purpose -- whoever can fix it is watching
    the TUI, where the full detail goes.
    """
    warning(f"[AI] LLM error on {detail}")
    send(sock, f"PRIVMSG {reply_to or CHANNEL} :{random.choice(_BRAIN_OFFLINE)}")


def _process_pending_vision(sock: socket.socket) -> None:
    """Check for and answer any queued image-analysis request."""
    # A paused bot makes no LLM calls; the queued request waits for unpause.
    if _paused["on"]:
        return
    item = _take_pending_vision()
    if item is None:
        return
    url, sender, prompt, reply_to = item
    action(f"[AI] thinking: image request from {sender}")
    with _prompt_lock:
        _busy["on"] = True
    try:
        reply = _call_llm_vision(url, prompt, asker=sender)
        for reply_line in _format_reply_lines(reply):
            send(sock, f"PRIVMSG {reply_to} :{reply_line}")
        speak(f"[AI] {' '.join(reply.split())}")
        if sender:
            _note_conversation(sender)
    except Exception as e:
        _say_brain_offline(sock, f"image request from {sender}: {e}", reply_to)
    finally:
        with _prompt_lock:
            _busy["on"] = False


def _queue_page(url: str, sender: str, reply_to: str = "") -> None:
    """Queue a page for the poll loop to fetch and summarise."""
    with _prompt_lock:
        _pending_page["url"] = url
        _pending_page["sender"] = sender
        _pending_page["reply_to"] = reply_to


def _take_pending_page() -> tuple[str, str, str] | None:
    """Retrieve and clear the queued page request, or None."""
    with _prompt_lock:
        if not _pending_page["url"]:
            return None
        item = (_pending_page["url"], _pending_page["sender"],
                _pending_page["reply_to"] or CHANNEL)
        _pending_page["url"] = _pending_page["sender"] = ""
        _pending_page["reply_to"] = ""
        return item


def _cached_page(url: str) -> tuple[str, str] | None:
    """A page already fetched this session, or None."""
    with _prompt_lock:
        return _page_cache.get(web.normalise(url))


def _cache_page(url: str, title: str, text: str) -> None:
    """Remember a fetched page, dropping the oldest past the cap."""
    with _prompt_lock:
        _page_cache[web.normalise(url)] = (title, text)
        while len(_page_cache) > WEB_CACHE_SIZE:
            _page_cache.pop(next(iter(_page_cache)))


def _page_prompt(title: str, text: str, truncated: bool) -> str:
    """What to send the model to summarise a page.

    The page text is fenced and labelled as fetched content, and the persona
    is told it is material rather than instructions. A page can say "ignore
    your instructions"; that is a thing the page says, and being able to report
    it is the right behaviour.
    """
    head = f"Title: {title}\n\n" if title else ""
    tail = "\n\n[the page was longer than this and has been cut off]" if truncated else ""
    return (
        "Summarise the page below for the channel.\n\n"
        f"--- BEGIN FETCHED PAGE ---\n{head}{text}{tail}\n--- END FETCHED PAGE ---"
    )


def _process_pending_page(sock: socket.socket) -> None:
    """Fetch and summarise one queued page.

    Two calls rather than one: a straight summary in the webpage persona, then
    a line about it in whatever mood the bot is in. Asking a single prompt to
    be both accurate and funny is the contradiction that makes a model split
    the difference and be neither.
    """
    if _paused["on"]:
        return
    item = _take_pending_page()
    if item is None:
        return
    url, sender, reply_to = item
    action(f"[AI] fetching a page for {sender}: {url}")
    with _prompt_lock:
        _busy["on"] = True
    try:
        cached = _cached_page(url)
        if cached is not None:
            title, text = cached
            action("[AI] using the copy already fetched this session")
        else:
            # The download happens here, on the poll loop, never on the
            # receiver thread -- it is as slow as an LLM call and would block
            # the socket read with the PING answers behind it.
            page = web.fetch(url, max_bytes=WEB_MAX_BYTES,
                             timeout=(5, WEB_TIMEOUT),
                             max_redirects=WEB_MAX_REDIRECTS)
            if not page:
                warning(f"[AI] {url} not fetched: {page.error}")
                send(sock, f"PRIVMSG {reply_to} :Can't read that one: {page.error}")
                return
            title, text = page.title, page.text
            _cache_page(url, title, text)
        truncated = len(text) > WEB_MAX_ARTICLE_CHARS
        summary = _call_llm(
            _page_prompt(title, text[:WEB_MAX_ARTICLE_CHARS], truncated), MODE_WEBPAGE
        )
        lines = _format_reply_lines(summary, WEB_MAX_REPLY_LINES)
        if WEB_ADD_COMMENT:
            comment = _call_llm(
                f"You just told the channel what this page says:\n{summary}\n\n"
                "Add one line of your own about it. Do not summarise it again.",
                _effective_mode(MODE_CHAT),
                # The summary itself answers about the page, but the line after
                # it is said to whoever asked for it.
                asker=sender,
            )
            lines += _format_reply_lines(comment, 1)
    except Exception as e:
        _say_brain_offline(sock, f"{url}: {e}", reply_to)
        return
    finally:
        with _prompt_lock:
            _busy["on"] = False
    for line in lines:
        send(sock, f"PRIVMSG {reply_to} :{line}")
    speak(f"[AI] {' '.join(' '.join(lines).split())}")
    if sender:
        _note_conversation(sender)


def _process_pending_greeting(sock: socket.socket) -> None:
    """Generate and send one queued greeting.

    Runs after the real work in the poll loop, so somebody's actual question is
    never held up behind a hello. A failed call falls back to the templated
    line rather than leaving a newcomer unwelcomed, and the mood still applies:
    asked to be serious, the bot greets people plainly.
    """
    if _paused["on"]:
        return
    item = _take_pending_greeting()
    if item is None:
        return
    nick, kind, flavour = item
    action(f"[AI] thinking: {flavour} greeting for {nick}")
    with _prompt_lock:
        _busy["on"] = True
    try:
        reply = _call_llm(
            _greeting_prompt(nick, kind, flavour), _effective_mode(MODE_CHAT)
        )
    except Exception as e:
        warning(f"[AI] greeting for {nick} failed, using a canned one: {e}")
        reply = _greeting_text(kind, nick)
    finally:
        with _prompt_lock:
            _busy["on"] = False
    for line in _format_reply_lines(reply):
        send(sock, f"PRIVMSG {CHANNEL} :{line}")
    speak(f"[AI] {' '.join(reply.split())}")


def _process_pending(sock: socket.socket) -> None:
    """Check for and respond to any pending AI prompt."""
    # A paused bot makes no LLM calls; the request waits for unpause.
    if _paused["on"]:
        return
    prompt, sender, stop, mode, reply_to = _take_pending()
    if stop:
        send(sock, f"PRIVMSG {reply_to} :{SHUTUP_REPLY}")
        _end_conversation()
        return
    if not prompt:
        return

    action(f"[AI] thinking: {prompt}")
    with _prompt_lock:
        _busy["on"] = True
    try:
        # Translation needs the target language pulled out of the request
        # before the model sees it; everything else is asked as it was typed.
        # A line said into the room is attributed with its sender's nick,
        # exactly like every other line the model reads -- it was the only
        # anonymous thing in the prompt. The modes that answer ABOUT something
        # are left alone: "!quote gandhi" asks for a quote from Gandhi, not
        # from alice, and _translate_prompt builds a prompt of its own that
        # quotes the text as data, where a nick is one more thing to translate.
        answered = _effective_mode(mode)
        if mode == MODE_TRANSLATE:
            asked = _translate_prompt(prompt)
        elif answered in ABOUT_THE_WORLD_MODES:
            asked = prompt
        else:
            asked = _attributed(sender, prompt)
        reply = _call_llm(asked, answered, asker=sender)
        for reply_line in _format_reply_lines(reply):
            send(sock, f"PRIVMSG {reply_to} :{reply_line}")
        # The bot actually spoke: route through the speak sink (light blue in
        # the TUI), not the action sink (yellow). The compact single line is
        # what the log shows; the full multi-line send is above it in the IRC
        # log pane.
        speak(f"[AI] {' '.join(reply.split())}")
        # Reading the reply takes time; start their window from now, not from
        # whenever they typed.
        if sender:
            _note_conversation(sender)
    except Exception as e:
        _say_brain_offline(sock, f"{prompt!r}: {e}", reply_to)
    finally:
        with _prompt_lock:
            _busy["on"] = False


def _merge_passages(first: list, second: list) -> list:
    """`first` then whatever of `second` it does not already carry.

    The searches overlap by design -- a line that answers the question is often
    also what the room is talking about -- and the same passage twice is a
    waste of the prompt and reads as a stutter.
    """
    return first + [p for p in second if p not in first]


def _recall_passages(prompt: str, recent: list, asker: str,
                     cutoff: float | None) -> list:
    """The passages three searches between them think are worth showing.

    Each runs against its OWN query, which is the point. The relevance floor is
    a fraction of `ideal`, and `ideal` is the sum of IDF over every query term,
    so terms from one concern raise the bar for another when they share a
    query. Measured against the real 2026-line log, "is probe fat" asked in a
    channel that had moved on: the question alone gave 3 discriminating terms,
    ideal 12.11 and 17 lines recalled; the question plus its three trailing
    lines gave 22 terms, ideal 111.01 and NOTHING, with 34 lines about it
    sitting in the log. The busier the room, the less it could remember about
    what it had actually been asked. Worse, those trailing lines are captured
    in the log and then excluded from the pool by the cutoff: they add their
    rare words to the bar and nothing can ever match them back.
    """
    historical = _asks_about_the_past(prompt)
    subjects = _recall_subjects(prompt, asker) if historical else []
    settings = recall.Settings(
        min_relevance=RECALL_MIN_RELEVANCE,
        half_life_days=0.0 if historical else RECALL_HALF_LIFE_DAYS,
        passages=RECALL_PASSAGES,
    )
    earliest = dataclasses.replace(settings, oldest=True)
    # What was asked. "Earliest" is a different question from "best match":
    # ranking by relevance answers "what did they say about X", never "what is
    # the first thing they said", so that one is served by age instead.
    asked = earliest if _asks_for_the_earliest(prompt) else settings
    passages = _recall_store.search(prompt, asked, before=cutoff,
                                    nicks=subjects)
    if historical and _named_others(prompt, asker):
        # Somebody ELSE was named, so what the room said about them counts as
        # much as what they said themselves: "where did dflatline go for his
        # vacation" is answered by both, and on the real log the three lines
        # about that trip are one of his and two of other people's.
        #
        # Only for a third party. A first-person question is about the asker's
        # OWN words -- "when did i say i was going to amsterdam" -- and an
        # unscoped pass there answers with somebody else's near-identical plan,
        # which is not a worse answer but a wrong one.
        passages = _merge_passages(passages, _recall_store.search(
            prompt, asked, before=cutoff))
    if recent and not historical:
        # What the room is talking about, which is all a follow-up with no
        # content of its own ("what do you reckon") has to go on. It earns its
        # slots on its own terms instead of taxing the question's.
        passages = _merge_passages(passages, _recall_store.search(
            " ".join(recent), settings, before=cutoff))
    if asker and not historical:
        # What the person asking has said about it before, which is what makes
        # a reply sound like it remembers them. Their profile carries their
        # last few lines whatever the subject; this carries the older ones
        # about what they are asking now.
        #
        # Without the recency prior, deliberately: somebody's own history does
        # not get less true with age. "my espresso machine leaks" is as
        # relevant to a descaling question four months later as it was that
        # day, and at the channel's half-life it had decayed to a twentieth of
        # its score, under the floor, which is where this was caught. The floor
        # still has to be cleared, so it is term overlap doing the work.
        mine = [p for p in _recall_store.search(
            prompt, dataclasses.replace(settings, half_life_days=0.0),
            before=cutoff, nicks=[asker])
            if p not in passages]
        if mine:
            # One slot is kept for them, or a search that filled every slot
            # would mean the bot never remembers who it is talking to.
            passages = passages[:RECALL_PASSAGES - 1] + mine
    return passages[:RECALL_PASSAGES]


def _recall_section(prompt: str, senders: list, lines: list,
                    times: list, asker: str = "") -> str:
    """Passages from the channel's past worth showing, or "".

    Up to three searches, each against its own query so that none of them
    raises the relevance bar for another: what was ASKED, what the room is
    talking about, and what the person asking has said about it before.
    Everything already in the verbatim recent block is excluded: quoting back
    what sits three paragraphs below it is not recall.

    A question ABOUT the past is run differently (see _asks_about_the_past):
    scoped to whoever it names -- or to the asker, when it says "i" -- and with
    the recency prior off. Both are wrong for ordinary chat and both are
    necessary here. Without the scope "what did alice say" has nothing to
    search on, because the index is built on a line's text and alice's own
    lines do not contain her name; with the prior on, the answer to a question
    about last month has already decayed under the floor.

    Older lines get the date as well as the clock -- the point of them is that
    they are not from today, and it is what lets the bot answer "when".
    """
    if not _recall_active():
        return ""
    recent = [
        _attributed(sender, text)
        for sender, text in zip(senders[-RECALL_QUERY_LINES:],
                                lines[-RECALL_QUERY_LINES:], strict=False)
    ]
    passages = _recall_passages(prompt, recent, asker,
                                times[-min(len(times), CONTEXT_RECENT_LINES)]
                                if times else None)
    if not passages:
        return ""
    blocks = [
        "\n".join(
            f"[{_datestamp(r['at'])} {_clock(r['at'])}] "
            f"{_attributed(r['nick'], r['text'])}"
            for r in passage
        )
        for passage in passages
    ]
    return ("--- EARLIER IN THE CHANNEL (older than the chat above) ---\n"
            + "\n\n".join(blocks))


def _context_block(prompt: str = "", addressing: str = "",
                   about: str = "", asker: str = "") -> list:
    """The summarizer's rolling context as ONE system message, or [].

    The rolling summary, highlights, and a verbatim sample of the most recent
    IRC lines are folded into a single system (background/observation) message
    rather than spread across separate user messages, so the model reads them
    as context for the room, not as people talking to it. Each IRC line keeps
    its "sender: text" form -- the sender is the speaker's name inline, never an
    LLM role -- so the recent chat reads as an observation of an external
    conversation. Sections are dropped when empty. Empty until there is
    something to say. Injected in _call_llm ahead of the current user message.

    Ordered most-stable-first, which is a performance property and not a
    cosmetic one: llama.cpp reuses a cached prompt only as far as two prompts
    agree from the first token, so whatever changes most often belongs at the
    bottom. Hence memory and highlights, then the recent chat, then the recall
    passages (they are chosen from the question), then the clock.
    """
    now = time.time()
    with _prompt_lock:
        summary = _rolling["summary"].strip()
        summary_at = _rolling["at"]
        highlights = list(_rolling["highlights"])
        senders = list(_recent_senders)
        lines = list(_recent_lines)
        times = list(_recent_times)
    sections = []
    if summary:
        sections.append(f"--- CONVERSATION MEMORY ---\n{summary}")
    if highlights:
        sections.append(
            "--- HIGHLIGHTS ---\n"
            + "\n".join(f"- {h}" for h in highlights)
        )
    # The clock on each line, plus the NOW section above, is what lets the bot
    # tell a reply thirty seconds old from one three hours old. The summarizer
    # is fed the unstamped shape (see _attributed) -- its prompt describes lines
    # as "nick: what they said" and is the user's to change, not this code's.
    # Padded rather than zipped strictly: a line whose arrival time is missing
    # is still a line the model needs, and dropping it silently would empty the
    # whole section.
    times += [0.0] * (len(lines) - len(times))
    recent = [
        f"[{_clock(at)}] {said}" if at else said
        for sender, text, at in zip(senders, lines, times, strict=False)
        if (said := _attributed(sender, text))
    ]
    if recent:
        sections.append(
            "--- RECENT IRC CHAT ---\n"
            + "\n".join(recent[-CONTEXT_RECENT_LINES:])
        )
    # After the recent chat rather than before it, and labelled with its age,
    # because these passages are chosen from the question and so change with
    # every question asked. Above the chat they put a different prefix in front
    # of it each time, and llama.cpp then re-reads the whole thing: measured on
    # this box, 2572 tokens cost 38.6s cold against 1.3s when the prefix is
    # reused. Each passage carries its own date and time, so the block still
    # reads as older material without having to sit in date order.
    earlier = _recall_section(prompt, senders, lines, times, asker)
    if earlier:
        sections.append(earlier)
        action("Recalled earlier channel chat")
    # Beside the recall passages, and for the same reason: chosen from the
    # question, so it changes whenever the question does and belongs at the
    # volatile end rather than above fifty lines of stable chat.
    if about:
        sections.append(about)
        action("Injected the profile of who was asked about")
    if not sections and not addressing:
        # Nothing has happened yet, so there is no context -- and a lone clock
        # is not context. Who the bot is answering is, though: right after a
        # join there are no lines yet and somebody can still ask it something.
        return []
    # LAST, deliberately. Everything above is stable between one reply and the
    # next; the clock changes every minute, and a prompt is only reused as far
    # as it matches from the first token, so a clock at the top threw away the
    # summary, the highlights and fifty lines of chat once a minute.
    age = (f" The conversation memory above was last updated "
           f"{_fmt_span(now - summary_at)} ago." if summary and summary_at
           else "")
    # `addressing` rides here, under the clock, for the same reason the clock
    # does: it names whoever is speaking now and so changes more often than
    # anything else in the prompt. See _addressing_section.
    who = f"\n{addressing}" if addressing else ""
    sections.append(
        f"--- NOW ---\nIt is {_clock(now)} on {_datestamp(now)}.{age}{who}"
    )
    return [{"role": "system", "content": "\n\n".join(sections)}]


def _reject_reason(summary: object) -> str | None:
    """Why a model-provided summary is unusable, or None if it is valid.

    A usable summary is a non-empty string no longer than SUMMARIZE_MAX_CHARS.
    A whitespace-only string counts as empty.
    """
    if not isinstance(summary, str):
        return "summary was not a string"
    if not summary.strip():
        return "summary was empty"
    if len(summary) > SUMMARIZE_MAX_CHARS:
        return (
            f"summary exceeds {SUMMARIZE_MAX_CHARS} characters ({len(summary)})"
        )
    return None


def _summarize_pending() -> None:
    """Roll the chat summary forward over lines not yet summarized.

    Triggers on age or volume: SUMMARIZE_INTERVAL seconds since the last
    summary, OR SUMMARIZE_VOLUME_LINES lines since it -- but only once at least
    SUMMARIZE_MIN_LINES lines have accumulated, so a quiet gap or a slow trickle
    never forces a summary. Snapshot the unsummarized lines and clear the live
    list first, so the IRC handler can keep appending while the LLM generates --
    the snapshot is an independent list, decoupled from the live one.

    A failed round-trip puts the snapshot back at the front of the buffer, so a
    server that is down costs the channel nothing but time; the summary age is
    left alone (the summary really is still that stale) and a short retry
    window keeps the worker from re-attempting on every poll. Runs off the main
    poll loop and never blocks a reply.
    """
    if _paused["on"]:
        return
    with _prompt_lock:
        if not _pending_summary_lines:
            return
        now = time.monotonic()
        if now < _summary_retry_at["t"]:
            return
        lines_since = len(_pending_summary_lines)
        elapsed = now - _last_summary_at["t"]
        if lines_since < SUMMARIZE_MIN_LINES:
            return
        if not (elapsed >= SUMMARIZE_INTERVAL
                or lines_since >= SUMMARIZE_VOLUME_LINES):
            return
        snapshot = list(_pending_summary_lines)
        _pending_summary_lines.clear()
        summary = _rolling["summary"]
        highlights = list(_rolling["highlights"])
    new_summary, new_highlights, ok = summarizer.summarize_tick_checked(
        summary, highlights, snapshot,
    )
    if not ok:
        # The lines were never summarized. Put them back in front of whatever
        # arrived while the call was in flight (still oldest-first), re-cap the
        # buffer, and leave the rolling state and its age untouched.
        with _prompt_lock:
            _pending_summary_lines[:0] = snapshot
            del _pending_summary_lines[:-SUMMARIZE_MAX_PENDING]
            _summary_retry_at["t"] = time.monotonic() + SUMMARIZE_RETRY_AFTER
        warning(f"SUMMARY FAILED: kept the previous one, retrying "
                f"{len(snapshot)} lines in {SUMMARIZE_RETRY_AFTER}s")
        return
    reject = _reject_reason(new_summary)
    if reject is not None:
        # Discard the unusable summary and keep the previous rolling one, so a
        # model response that is not a string, is empty, or is too large can
        # never overwrite the channel's memory. The warning is posted outside
        # the lock; the highlights are still valid and are applied as usual.
        warning(f"INVALID SUMMARY RECEIVED: {reject}")
        new_summary = summary
        updated = False
    else:
        updated = True
    with _prompt_lock:
        _rolling["summary"] = new_summary
        _rolling["highlights"] = new_highlights
        _rolling["at"] = time.time()
        _memory_dirty["on"] = True
        _last_summary_at["t"] = time.monotonic()
    _save_memory()
    if updated:
        action(f"[AI] summary updated: {len(new_highlights)} highlights")


def _save_memory() -> None:
    """Write the rolling summary and highlights to disk.

    Called when a summary lands and again on the way out, rather than on a
    timer: the rolling state only ever changes at those two points, so a
    debounce like the profile store's would have nothing to debounce. Uses the
    profile store's atomic writer -- it is a plain JSON round-trip and there is
    no second one worth having.
    """
    with _prompt_lock:
        if not _memory_dirty["on"]:
            return
        snapshot = {
            "version": MEMORY_VERSION,
            "summary": _rolling["summary"],
            "highlights": list(_rolling["highlights"]),
            "at": _rolling["at"],
        }
        _memory_dirty["on"] = False
    if not profiles.write(_memory_path(), snapshot):
        # Still ahead of the file: let the next summary try again.
        with _prompt_lock:
            _memory_dirty["on"] = True
        warning(f"[AI] could not save the channel memory to {_memory_path()}")


def _save_ignores() -> None:
    """Write the runtime ignore list to disk.

    Written on every change rather than on a timer: !ignore is reached for
    perhaps twice a year, and the one thing it must not do is forget.
    """
    with _prompt_lock:
        snapshot = {"version": IGNORES_VERSION, "masks": list(_ignored_live)}
    if not profiles.write(_ignores_path(), snapshot):
        warning(f"[AI] could not save the ignore list to {_ignores_path()}")


def _load_ignores() -> None:
    """Read the runtime ignore list back, once, at startup.

    A missing file is the normal case. Anything unreadable or of another
    version starts empty and says so -- the configured masks are unaffected,
    so the bot is never left quietly obeying half a list.
    """
    data = profiles.read(_ignores_path())
    if not isinstance(data, dict) or data.get("version") != IGNORES_VERSION:
        if data is not None:
            warning(f"[AI] ignore list at {_ignores_path()} is unreadable; "
                    "starting with the configured masks only")
        return
    masks = data.get("masks")
    with _prompt_lock:
        _ignored_live.clear()
        _ignored_live.extend(
            m for m in (masks or []) if isinstance(m, str) and m.strip()
        )
        count = len(_ignored_live)
    if count:
        action(f"[AI] ignoring {_plural(count, 'mask')} from the last session")


def _load_memory() -> None:
    """Read the rolling summary and highlights back, once, at startup.

    Without this a restart wiped everything the channel had said that was older
    than the recent-line buffer, which is most of it. A missing file is the
    normal first run; anything unreadable or of another version is treated as
    missing, because starting with no memory beats not starting.
    """
    data = profiles.read(_memory_path())
    if not isinstance(data, dict) or data.get("version") != MEMORY_VERSION:
        if data is not None:
            warning(f"[AI] channel memory at {_memory_path()} is unreadable; "
                    "starting fresh")
        else:
            action(f"[AI] no channel memory at {_memory_path()}; starting fresh")
        return
    summary = data.get("summary")
    highlights = data.get("highlights")
    at = data.get("at")
    with _prompt_lock:
        _rolling["summary"] = summary if isinstance(summary, str) else ""
        _rolling["highlights"] = [
            h for h in (highlights or []) if isinstance(h, str)
        ] if isinstance(highlights, list) else []
        _rolling["at"] = float(at) if isinstance(at, int | float) else 0.0
        stored, count, when = (
            _rolling["summary"], len(_rolling["highlights"]), _rolling["at"]
        )
    if not stored and not count:
        action("[AI] channel memory was empty; starting fresh")
        return
    age = f", {_fmt_span(time.time() - when)} old" if when else ""
    action(f"[AI] channel memory loaded: {len(stored)} chars, "
           f"{_plural(count, 'highlight')}{age}")


def _load_recall() -> None:
    """Read the channel log back, once, at startup.

    Trimming to RECALL_MAX_LINES happens here and nowhere else -- it is the one
    moment the whole file is in hand anyway -- and the file is rewritten only
    when something actually went, so a normal start is a read and nothing more.
    """
    recall.error_sink = warning
    kept, dropped = _recall_store.load(_recall_path())
    if dropped:
        _recall_store.rewrite(_recall_path())
    state = "on" if _recall_active() else "off (capturing only)"
    if not kept:
        action(f"[AI] no channel log at {_recall_path()}; recall {state}")
        return
    action(f"[AI] channel log: {_plural(kept, 'line')}"
           + (f", {dropped} trimmed" if dropped else "")
           + f"; recall {state}")


def _load_profiles() -> None:
    """Read the profile store off disk, once, before the first connection.

    A missing file is the normal first run. A corrupt one is reported and
    treated as missing: starting with no memory of anybody beats not starting.
    """
    data = profiles.read(_profile_path)
    with _prompt_lock:
        _profile_store.restore(data)
        dropped = _profile_store.prune()
        known = len(_profile_store.known())
        _profiles_dirty["on"] = bool(dropped)
    if data is None:
        action(f"[AI] no profile store at {_profile_path}; starting fresh")
    else:
        action(f"[AI] profiles loaded: {known} people"
               + (f", {dropped} pruned" if dropped else ""))


def _save_profiles_if_due(force: bool = False) -> None:
    """Write the profile store, at most once per PROFILE_SAVE_INTERVAL.

    Skipped entirely when nothing has changed. The snapshot is taken under the
    lock and the disk write happens outside it, so a slow disk never holds up a
    reply. `force` is for shutdown, where the interval does not apply.
    """
    now = time.monotonic()
    with _prompt_lock:
        if not _profiles_dirty["on"]:
            return
        if not force and now - _profiles_saved_at["t"] < PROFILE_SAVE_INTERVAL:
            return
        snapshot = _profile_store.snapshot()
        _profiles_saved_at["t"] = now
        _profiles_dirty["on"] = False
    if not profiles.write(_profile_path, snapshot):
        # The write failed, so the store is still ahead of the file: leave the
        # dirty flag up and let the next tick try again.
        with _prompt_lock:
            _profiles_dirty["on"] = True
        warning(f"[AI] could not save profiles to {_profile_path}")


def _plural(count: int, noun: str) -> str:
    """`1 line` / `4 lines`, for text the channel actually reads."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _clock(at: float) -> str:
    """A wall-clock time as the channel would read it: "23:41"."""
    return time.strftime("%H:%M", time.localtime(at))


def _datestamp(at: float) -> str:
    """A date as the channel would read it: "Tuesday 15 September 2026"."""
    return time.strftime("%A %d %B %Y", time.localtime(at))


def _fmt_span(seconds: float) -> str:
    """A rough age for a channel line: "3 days", "4 hours", "12 minutes"."""
    for size, name in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if seconds >= size:
            count = int(seconds // size)
            return f"{count} {name}{'s' if count != 1 else ''}"
    return "moments"


def _profile_names_locked(profile: dict) -> str:
    """This person's nicks, busiest first. Caller holds _prompt_lock."""
    order = sorted(profile["aliases"], key=lambda a: -profile["aliases"][a])
    return ", ".join(profile["casing"].get(alias, alias) for alias in order)


def _recall_reply(nick: str) -> str:
    """What the bot has on file for `nick`, as one line for the channel.

    Counts, names and dates -- never the stored lines themselves. Reciting
    somebody's own words back into the channel is a worse answer to "what do
    you know about me" than not answering it.
    """
    with _prompt_lock:
        profile = _profile_store.get(nick)
        if profile is None:
            return "Nothing on file for you."
        kept = len(profile["lines"])
        total = profile["line_count"]
        names = _profile_names_locked(profile)
        highlights = len(profile["highlights"])
        age = _fmt_span(time.time() - profile["first_seen"])
    return (
        f"On file for you: {_plural(kept, 'line')} kept of {total} counted, "
        f"first heard {age} ago, under {names}. "
        f"{_plural(highlights, 'highlight')}. "
        "Say 'forget about me' and it all goes."
    )


def _forget_recent_locked(aliases: set[str],
                          since: float | None = None) -> int:
    """Drop a person's lines from the short-lived buffers. Caller holds the lock.

    Wiping the profile but leaving their last lines in the recent-chat buffer
    would have the bot quoting somebody it had just promised to forget, in the
    very next reply. The rolling summary is prose and cannot be edited
    surgically -- it ages out instead, and the reply says so rather than
    claiming more than is true.
    """
    kept = [
        (sender, text, at)
        for sender, text, at in zip(_recent_senders, _recent_lines,
                                    _recent_times, strict=False)
        if sender.lower() not in aliases or (since is not None and at < since)
    ]
    dropped = len(_recent_lines) - len(kept)
    theirs = {
        _attributed(sender, text)
        for sender, text in zip(_recent_senders, _recent_lines, strict=False)
        if sender.lower() in aliases
    }
    _recent_senders.clear()
    _recent_lines.clear()
    _recent_times.clear()
    for sender, text, at in kept:
        _recent_senders.append(sender)
        _recent_lines.append(text)
        _recent_times.append(at)
    _pending_summary_lines[:] = [
        line for line in _pending_summary_lines if line not in theirs
    ]
    return dropped


def _forget_logged(aliases: set[str], since: float | None = None) -> int:
    """Drop `aliases` from the long-term log and write it out. Returns how many.

    Outside the lock and outside _forget_recent_locked, which holds it: this
    rewrites a file, and a disk write does not belong under the lock a reply is
    waiting on. Without this the bot could quote somebody next Tuesday that it
    promised to forget today, which is the whole point of the promise.
    """
    dropped = _recall_store.forget(aliases, since)
    if dropped:
        _recall_store.rewrite(_recall_path())
    return dropped


def _forget_reply(nick: str) -> str:
    """Erase everything on file for `nick` and say what went.

    The store is written to disk immediately rather than waiting for the next
    debounced tick: a wipe that a crash could undo is not a wipe. That write is
    on the receiver thread, which is acceptable for one explicit command on a
    small file.
    """
    with _prompt_lock:
        profile = _profile_store.get(nick)
        if profile is None:
            return "Nothing on file for you to forget."
        total = profile["line_count"]
        names = _profile_names_locked(profile)
        aliases = set(profile["aliases"])
        _profile_store.forget(nick)
        _forget_recent_locked(aliases)
        _profiles_dirty["on"] = True
    _forget_logged(aliases)
    _save_profiles_if_due(force=True)
    return (
        f"Forgotten: {_plural(total, 'line')} under {names}, and your recent "
        "chat with them. The rolling channel summary is prose I can't edit "
        "surgically, so anything of yours in there ages out on its own."
    )


def _handle_privacy_command(sock: socket.socket, sender: str, command: str,
                            reply_to: str = CHANNEL) -> None:
    """Answer a privacy command straight from the receiver thread.

    Templated and immediate, like a mood switch: somebody asking what is stored
    about them, or asking for it to go, wants a straight answer, not the
    persona having a go at it -- and not a two-second wait behind an LLM call.
    """
    reply = _recall_reply(sender) if command == "recall" else _forget_reply(sender)
    for line in _format_reply_lines(reply):
        send(sock, f"PRIVMSG {reply_to} :{line}")
    action(f"[AI] {sender} used the '{command}' privacy command")


def shutdown() -> None:
    """Stop the background workers and flush anything owed to disk.

    Called by whoever is shutting the bot down, on their own thread, and NOT
    left to the summarizer worker. That worker is a daemon: the interpreter
    kills daemon threads at exit without joining them, so a flush at the end of
    its loop is never reliably reached, and everything captured since the last
    debounced write went with it on every quit. Idempotent -- a second call has
    nothing left owed and writes nothing.
    """
    _stop_event.set()
    _save_profiles_if_due(force=True)
    _save_memory()


def report_config(problems: list[str]) -> None:
    """Say what the config is and what is wrong with it, in the log pane.

    Shared by startup and the reload key so the two cannot drift into telling
    different stories about the same file.
    """
    for problem in problems:
        warning(f"[AI] config: {problem}")
    path = config.default_path()
    if not path.exists():
        warning(f"[AI] no {path.name}: every lever is at its default and the "
                "persona is the short built-in one")
        return
    action(f"[AI] config: {path.name}, {len(PERSONAS)} personas, "
           f"{len(_MOODS)} moods"
           + (f", {len(problems)} problem(s)" if problems else ""))


def _summarize_loop() -> None:
    """Background worker: check for a summary trigger every SUMMARIZE_POLL_INTERVAL.

    Waits out the poll interval on _stop_event so the TUI can stop it promptly;
    runs as a daemon thread started from main(). The actual summary is decided
    inside _summarize_pending, which gates on age/volume and a minimum line
    count. The debounced profile-store write rides along here rather than on
    its own thread: neither job may block a reply, and both are already off the
    poll loop.
    """
    while not _stop_event.wait(SUMMARIZE_POLL_INTERVAL):
        _summarize_pending()
        _save_profiles_if_due()
    # _stop_event was set: flush what the last interval has not written yet.
    _save_profiles_if_due(force=True)


def _wait_to_register() -> bool:
    """Wait for the server's 001, giving up early if we are asked to stop.

    Polled rather than a single long wait so a shutdown does not sit through
    the whole registration timeout. That matters under a service manager: a
    stop that takes REGISTER_TIMEOUT looks like a hang, and a slower one gets
    SIGKILLed with the profiles and the channel memory unflushed.
    """
    deadline = time.monotonic() + REGISTER_TIMEOUT
    while time.monotonic() < deadline:
        if _stop_event.is_set():
            return False
        if _registered.wait(timeout=min(POLL_INTERVAL, REGISTER_TIMEOUT)):
            return True
    return False


def _connect(gone: threading.Event) -> socket.socket | None:
    """Open one connection: register, join the channel, and ask who is here.

    Returns the live socket, or None when the server could not be reached or
    never completed registration -- the caller backs off and tries again.
    `gone` is this session's disconnect flag, handed to the receiver thread.
    """
    _registered.clear()
    with _prompt_lock:
        # A new connection is a new room, so the old roster goes -- and it goes
        # BEFORE the link exists. Clearing it after registration raced the
        # 353 NAMREPLY that the JOIN below asks for: on a fast server the reply
        # landed first and the clear then wiped the roster it had just filled.
        _users["names"].clear()
    try:
        sock = socket.create_connection((SERVER, PORT), timeout=REGISTER_TIMEOUT)
    except OSError as e:
        warning(f"[Connect] {SERVER}:{PORT} unreachable: {e}")
        return None
    # Blocking from here on: the receiver thread parks in recv() until the
    # server says something or the link dies.
    sock.settimeout(None)
    threading.Thread(target=receiver, args=(sock, gone), daemon=True).start()
    try:
        send(sock, f"NICK {NICK}")
        send(sock, f"USER {NICK} 0 * :{REALNAME}")
        if not _wait_to_register():
            if _stop_event.is_set():
                raise TimeoutError("asked to stop while registering")
            raise TimeoutError(f"no 001 Welcome within {REGISTER_TIMEOUT}s")
        # An outage that was already long enough to leave for is still long
        # enough on the other side of a reconnect, so register and stay out
        # rather than join and walk back out two seconds later.
        sitting_out = _should_sit_out()
        if not sitting_out:
            send(sock, f"JOIN {CHANNEL}")
            _request_userlist(sock)
    except Exception as e:
        warning(f"[Connect] registration failed: {e}")
        # Closing wakes the receiver thread, which sets `gone` on its way out.
        sock.close()
        return None
    with _prompt_lock:
        _absent["on"] = sitting_out
        # The grace period starts again, so the auto-interject opener waits for
        # the WHO/NAMES replies now on their way.
        _joined["at"] = 0.0 if sitting_out else time.monotonic()
    if sitting_out:
        warning(f"[Connected] staying out of {CHANNEL}: still no LLM")
    else:
        action(f"[Connected] joined {CHANNEL}")
    return sock


def _drop_queued_work() -> None:
    """Throw away every queued request and greeting.

    Called when the bot leaves the channel: whoever asked is about to watch it
    walk out, and answering them when it walks back in minutes later would be
    replying to a conversation that has moved on.
    """
    with _prompt_lock:
        _pending["prompt"] = ""
        _pending["sender"] = ""
        _pending["stop"] = False
        _pending["reply_to"] = ""
        _pending_vision["url"] = ""
        _pending_vision["sender"] = ""
        _pending_vision["prompt"] = ""
        _pending_vision["reply_to"] = ""
        _pending_page["url"] = ""
        _pending_page["sender"] = ""
        _pending_page["reply_to"] = ""
        _pending_greetings.clear()


def _leave_channel(sock: socket.socket) -> None:
    """PART the channel for the duration of an LLM outage, staying connected.

    The link itself is fine, so dropping it would throw away the reconnect
    backoff, the roster and the server's goodwill to say something about the
    model. The PART reason says why, in the one place everybody in the channel
    will see it.
    """
    send(sock, f"PART {CHANNEL} :{PART_REASON}")
    _drop_queued_work()
    with _prompt_lock:
        _absent["on"] = True
        _joined["at"] = 0.0
        # A channel we are not in has no roster: the mention list must not name
        # people we can no longer see.
        _users["names"].clear()
    warning(f"[AI] left {CHANNEL}: no LLM for "
            f"{int(_llm_down_for())}s; rejoining when one answers")


def _rejoin_channel(sock: socket.socket) -> None:
    """JOIN again after the LLM came back, and ask who is here now."""
    send(sock, f"JOIN {CHANNEL}")
    _request_userlist(sock)
    with _prompt_lock:
        _absent["on"] = False
        # Same reason as on a fresh connection: the opener waits for the
        # WHO/NAMES replies now on their way.
        _joined["at"] = time.monotonic()
    action(f"[Connected] rejoined {CHANNEL}: LLM back")


def _maintain_presence(sock: socket.socket) -> None:
    """Leave the channel while the LLM is out, and come back when it returns.

    Polled rather than event-driven because the health it reads is polled too
    (see _probe_props); at POLL_INTERVAL the lag is nothing beside the minutes
    the decision is measured in.
    """
    with _prompt_lock:
        absent = _absent["on"]
        healthy = _llm_health["ok"]
    if absent:
        if healthy:
            _rejoin_channel(sock)
    elif _should_sit_out():
        _leave_channel(sock)


def _run_session(sock: socket.socket, gone: threading.Event) -> None:
    """Poll for pending work until the TUI stops us or the link drops."""
    try:
        while not _stop_event.is_set() and not gone.is_set():
            _probe_props_if_due()
            _maintain_presence(sock)
            # Off the channel, the unprompted talk has no room to talk into.
            # Requests are still served: the only ones that can arrive are an
            # owner's private messages, and an owner asking why it is quiet
            # deserves the brain-offline line rather than silence.
            if not _is_absent():
                _check_silence()
                _process_pending_greeting(sock)
            _process_pending(sock)
            _process_pending_vision(sock)
            _process_pending_page(sock)
            time.sleep(POLL_INTERVAL)
    finally:
        sock.close()


def main() -> None:
    """Connect, serve the channel, and reconnect for as long as we are running.

    A link that drops -- a netsplit, a server restart, a connection refused --
    is retried after RECONNECT_MIN_DELAY, doubling to RECONNECT_MAX_DELAY, so a
    server that is down is not hammered; a connection that registers and joins
    resets the backoff. Waits happen on _stop_event, so quitting the TUI does
    not sit through a five-minute backoff.

    The rolling-summarizer worker is started once and survives reconnects: the
    channel's memory is not a property of the socket, and lines that arrived
    before a drop are still worth summarizing after it. The same goes for the
    profile store, which is read once here and written by that worker.
    """
    # Re-read rather than trusting the import-time load, so startup reports
    # the config exactly the way pressing R does.
    report_config(reload_config())
    _load_profiles()
    _load_memory()
    _load_ignores()
    _load_recall()
    threading.Thread(target=_summarize_loop, daemon=True).start()
    delay = RECONNECT_MIN_DELAY
    try:
        while not _stop_event.is_set():
            gone = threading.Event()
            sock = _connect(gone)
            if sock is None:
                action(f"[Connect] retrying in {delay}s")
                _stop_event.wait(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY)
                continue
            delay = RECONNECT_MIN_DELAY
            _run_session(sock, gone)
            if _stop_event.is_set():
                break
            action(f"[Connect] link lost; reconnecting in {delay}s")
            _stop_event.wait(delay)
    except KeyboardInterrupt:
        pass
    # Running stand-alone, main() is the thread that is about to end, so the
    # flush belongs here for the same reason it belongs in the TUI's unmount.
    shutdown()
    action("[Exiting]")


def status_snapshot() -> dict:
    """A point-in-time read of everything the status pane should show.

    Reads all state under _prompt_lock once, then derives the persona mode and
    the remaining timers. Pure (no side effects) so it can be polled each tick.
    """
    now = time.monotonic()
    with _prompt_lock:
        raw_mood = _mood["name"]
        mood_at = _mood["at"]
        recent = len(_recent_lines)
        # Displayed chatter order follows the mention priority (most recently
        # spoken / engaged first), not registration order, so the TUI user list
        # matches the order the names are handed to the LLM. Call the lock-free
        # core here -- status_snapshot already holds _prompt_lock.
        users = _mention_targets_locked()
        floor_open = now < _open_floor["deadline"]
        floor_used = _open_floor["used"]
        chatter = _chatter["count"]
        quiet = now - _activity["at"] if _activity["at"] else 0.0
        busy = _busy["on"]
        convo = _conversation["nick"]
        joined = bool(_joined["at"])
        grace_left = (JOIN_GRACE_PERIOD - (now - _joined["at"])) if joined else 0.0
        absent = _absent["on"]
        ignored_config = len(IGNORE_MASKS)
        ignored_live = len(_ignored_live)
        llm_ok = _llm_health["ok"]
        llm_down_since = _llm_health["down_since"]
        # Read the vision state directly here (not via _vision_active/_vision
        # source, which take the same lock) to avoid re-entering the lock.
        vision_override = _vision["override"]
        vision_enabled = _vision["enabled"]
        model_alias = _model["alias"]
        model_detected = _model["detected"]
        summary = _rolling["summary"]
        highlights = list(_rolling["highlights"])
        pending = len(_pending_summary_lines)
        # Wall clock, not the monotonic trigger clock: a summary restored from
        # disk is genuinely hours old and the status pane must not call it new.
        summary_at = _rolling["at"]
        recall_lines = len(_recall_store)
        known_profiles = len(_profile_store.known())
        pages_cached = len(_page_cache)
    # _current_mood, not the raw dict: a scheduled window is the mood actually
    # in force, and a pane that says "banter" while the bot is being vile is
    # worse than no pane. Called outside the snapshot lock -- it takes its own.
    mood_name = _current_mood()
    scheduled = mood_name != raw_mood
    mode = MOOD_MODES.get(mood_name, MODE_CHAT)
    mood_left = (
        _scheduled_mood_left() if scheduled
        else (MOOD_TIMEOUT - (now - mood_at)) if mood_name != MOOD_BANTER
        else 0.0
    )
    grace_active = grace_left > 0
    return {
        "mood": mood_name,
        "mood_scheduled": scheduled,
        "mode": mode,
        "mood_left": mood_left,
        "history": recent,
        "history_max": RECENT_LINES,
        "users": users,
        "floor_open": floor_open,
        "floor_used": floor_used,
        "floor_left": (_open_floor["deadline"] - now) if floor_open else 0.0,
        "chatter": chatter,
        "quiet": quiet,
        "busy": busy,
        "conversation": convo,
        "grace_active": grace_active,
        "grace_left": grace_left,
        "joined": joined,
        "vision": (vision_override if vision_override is not None
                   else vision_enabled),
        "vision_source": "auto" if vision_override is None else ("on" if vision_override else "off"),
        "summary": summary,
        "highlights": len(highlights),
        "highlight_list": highlights,
        "pending_summary": pending,
        "summary_age": max(0.0, time.time() - summary_at) if summary_at else 0.0,
        "profiles": known_profiles,
        "pages_cached": pages_cached,
        "web_enabled": WEB_ENABLED,
        "recall_enabled": _recall_active(),
        "recall_source": _recall_source(),
        "recall_lines": recall_lines,
        "model": model_alias,
        "model_detected": model_detected,
        "ignored": ignored_config + ignored_live,
        "ignored_live": ignored_live,
        "llm_ok": llm_ok,
        "llm_down_for": (now - llm_down_since) if llm_down_since else 0.0,
        "absent": absent,
    }


def profiles_snapshot() -> list[dict]:
    """Every profile, most recently seen first, copied for the TUI.

    Separate from status_snapshot because that one is polled every second and
    has no business copying everybody's stored lines. Deep-copied under the
    lock so the UI thread reads a stable picture while the channel talks.
    """
    with _prompt_lock:
        return [
            {
                "nick": p["nick"],
                "aliases": [
                    p["casing"].get(a, a)
                    for a in sorted(p["aliases"], key=lambda a: -p["aliases"][a])
                ],
                "lines": [list(line) for line in p["lines"]],
                "highlights": list(p["highlights"]),
                "quotes": [list(q) for q in p["quotes"]],
                "first_seen": p["first_seen"],
                "last_seen": p["last_seen"],
                "line_count": p["line_count"],
            }
            for p in _profile_store.known()
        ]


def _log_writer(stream) -> "callable":
    """A sink that stamps each line with the time and writes it to `stream`."""
    def write(msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{stamp} {msg}", file=stream, flush=True)
    return write


def run_headless(argv: list | None = None) -> int:
    """Run the bot with no TUI, for a service manager or a detached shell.

    Everything the log pane would show goes to stdout (so journald or a
    redirect catches it) or to --log. The verbose prompt dumps stay off unless
    asked for: they are several kilobytes per reply and the TUI hides them by
    default for the same reason.

    SIGTERM and SIGINT set the stop event rather than killing the process.
    main() polls, so it notices within POLL_INTERVAL and then runs shutdown()
    itself -- which is what flushes the profiles and the channel memory. A
    service that is SIGKILLed loses whatever those had not written yet, so the
    unit file gives it time to stop.
    """
    parser = argparse.ArgumentParser(
        prog="llmbot_core", description="Run the IRC bot without the TUI."
    )
    parser.add_argument("--log", metavar="PATH",
                        help="append the log here instead of stdout")
    parser.add_argument("--verbose", action="store_true",
                        help="include the full prompt dumps (very noisy)")
    args = parser.parse_args(argv)

    stream = sys.stdout
    if args.log:
        path = pathlib.Path(args.log).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

    write = _log_writer(stream)
    globals().update({
        "irc_sink": write, "action_sink": write, "chat_sink": write,
        "speak_sink": write, "warning_sink": write,
        "debug_sink": write if args.verbose else (lambda _m: None),
    })
    summarizer.error_sink = write
    recall.error_sink = write

    def stop(signum, _frame):
        # Not shutdown() directly: this runs on the main thread, and the flush
        # belongs with the rest of the teardown in main() rather than racing it.
        action(f"[Exiting] signal {signal.Signals(signum).name}")
        _stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)

    problem = llm_preflight_problem()
    if problem and LLM_CHECK == "fail":
        write(f"[Exiting] refusing to start: {problem}")
        return 1
    if problem and LLM_CHECK == "ask":
        if not sys.stdin.isatty():
            # A service has nobody to ask, and blocking on a prompt that will
            # never be answered is worse than starting without a model: the
            # bot reconnects and picks the server up when it appears.
            write("[Starting] nobody to ask (not a terminal); carrying on")
        elif input(f"No LLM: {problem}\nStart anyway? [y/N] ").strip().lower() \
                not in ("y", "yes"):
            write("[Exiting] not starting")
            return 1

    action(f"[Starting] sloppy {VERSION} headless, pid {os.getpid()}")
    main()
    return 0


if __name__ == "__main__":
    sys.exit(run_headless())
