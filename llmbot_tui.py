#!/usr/bin/env python3
"""Textual front-end for the IRC AI bot.

Imports llmbot_core (the business logic, a fork of bot.py) and paints two
panes:

  * top-left -> the log          every action the bot takes (being addressed,
                   replying, switching mood, interjecting, injecting its chat
                   history, shutting down) and every line of channel chat in
                   "nick: text" format. Actions are bold and bright; plain chat
                   lines are not.
  * top-right -> status          (mood/mode, chat-history buffer count,
                   open-floor state, chatter count, reply/quiet/join timers).

The raw IRC log is intentionally not shown — it is noise, not bot state. The
bot runs in a background thread; the module sinks are wired to the app so every
meaningful line is posted back to the UI thread. Run with:

    python3 llmbot_tui.py

Only the Python standard library's curses-free Textual is used, and it is
already installed in the system Python that runs this bot.
"""

from __future__ import annotations

import threading
import time

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.binding import Binding
from textual.message import Message
from rich.text import Text
from textual.screen import ModalScreen
from textual.widgets import Button, RichLog, Static, TextArea

import llmbot_core as bot
import profiles


class LogLine(Message):
    """A line for the log pane, posted from the background thread."""

    def __init__(
        self,
        text: str,
        *,
        action: bool = False,
        speak: bool = False,
        warning: bool = False,
    ) -> None:
        super().__init__()
        self.data = text
        self.is_action = action
        self.is_speak = speak
        self.is_warning = warning


def _plural(count: int, noun: str) -> str:
    """`1 line`, `2 lines`. The pane said "1 lines logged" for a year."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _fmt_duration(seconds: float) -> str:
    """`125.0` -> `2m05s`; sub-60 stays in seconds."""
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _format_status(snap: dict) -> str:
    """Render a status snapshot into a single-column, monospace block."""
    mode_left = _fmt_duration(snap["mood_left"]) if snap["mood"] != "banter" else "—"
    floor = "active" if snap["floor_open"] else "inactive"
    if snap["floor_open"]:
        floor += f" ({snap['floor_used']}/{bot.OPEN_FLOOR_MAX_PROMPTS}, " \
                 f"{_fmt_duration(snap['floor_left'])} left)"
    quiet = _fmt_duration(snap["quiet"])
    quiet += " idle" if snap["quiet"] >= bot.SILENCE_TIMEOUT else ""
    if snap.get("absent"):
        grace = (f"afk — no LLM for "
                 f"{_fmt_duration(snap['llm_down_for'])}")
    elif snap["grace_active"]:
        grace = f"{_fmt_duration(snap['grace_left'])} grace"
    elif snap["joined"]:
        grace = "up"
    else:
        grace = "connecting"
    busy = "replying" if snap["busy"] else "idle"
    users = ", ".join(snap["users"]) if snap["users"] else "(none yet)"
    # Say when the name is only what we would ask for, not what answered: with
    # the server down the pane would otherwise claim a model is loaded.
    model = snap["model"] if snap["model_detected"] else f"{snap['model']} (no reply)"
    vsrc = snap["vision_source"]
    if vsrc == "auto":
        vision = f"auto ({'enabled' if snap['vision'] else 'disabled'})"
    else:
        vision = f"{vsrc} (forced)"
    # Who and what it is, then what it knows, then how it is behaving and who
    # is in the room. The identity rows are read once on sitting down; the
    # timers are the ones somebody watches.
    lines = [
        f"Nick        : {bot.NICK}",
        f"Version     : {bot.VERSION}",
        f"Model       : {model}",
        f"Vision      : {vision}",
        f"Mood        : {_mood_row(snap, mode_left)}",
        _SEPARATOR,
        *_memory_rows(snap),
        _SEPARATOR,
        f"Open floor  : {floor}",
        f"Chatter     : {snap['chatter']}/{bot.IDLE_INTERJECT_AFTER} to interject",
        f"Quiet       : {quiet}",
        f"Bot         : {busy}",
        f"Join        : {grace}",
        f"Profiles    : {snap['profiles']} known",
        f"Ignored     : {_ignored_row(snap)}",
        f"Users       : {len(snap['users'])} — {users}",
    ]
    return "\n".join(lines)


# Between the identity rows, what the bot remembers, and how it is behaving.
# Three groups in a pane this tall read as one long list without it.
_SEPARATOR = "─" * 28


def _mood_row(snap: dict, mode_left: str) -> str:
    """The mood, what is left of it, and where it came from -- one row.

    "auto:" marks a mood the timetable chose rather than one somebody asked
    for. It is a prefix rather than a third item in the brackets because
    `wholesome (15m00s left, scheduled)` is 48 columns and the pane is 43: it
    wrapped, and a wrapped row pushes the ones below it off the bottom.

    The persona is named only when it is not the mood's own name: in a mood
    called `mean` answering in the `mean` persona, saying so twice is noise.
    """
    notes = []
    if snap["mode"] not in ("chat", snap["mood"]):
        notes.append(f"{snap['mode']} persona")
    if mode_left != "—":
        notes.append(f"{mode_left} left")
    prefix = "auto: " if snap.get("mood_scheduled") else ""
    return prefix + snap["mood"] + (f" ({', '.join(notes)})" if notes else "")


def _ignored_row(snap: dict) -> str:
    """How many masks the bot is deaf to, and where they came from."""
    ignored, live = snap["ignored"], snap["ignored_live"]
    if not ignored:
        return "nobody"
    return f"{_plural(ignored, 'mask')} ({live} live, {ignored - live} config)"


def _memory_rows(snap: dict) -> list[str]:
    """The rows about what the bot remembers and can look up.

    Split out of _format_status to keep it under the complexity ceiling; they
    are also the rows that read as a group, between the two separators.
    """
    # One line only. A summary runs to SUMMARIZE_MAX_CHARS and carries its
    # highlights: at a 120x40 terminal that wraps to roughly fifty rows in a
    # pane that has about twenty-five, and a Static clips rather than scrolls,
    # so the tail was silently lost under the docked hint row. The text itself
    # lives in the 'S' pop-up (see SummaryView).
    if snap["summary"].strip():
        summary = (f"{_fmt_duration(snap['summary_age'])} old, "
                   f"{snap['pending_summary']} pending")
    else:
        summary = f"none yet ({snap['pending_summary']} pending)"
    # Said both ways round on purpose. Reading "on" off the ABSENCE of a note
    # means you cannot tell it from a row you have not understood yet.
    source = snap["recall_source"]
    state = (f"on ({source})" if snap["recall_enabled"]
             else f"off ({source}, still logging)")
    web_note = "" if snap["web_enabled"] else " (!summarize off)"
    # Measured against the pane, not guessed: at a 120-column terminal the
    # status pane is 43 columns, and a row longer than that wraps onto a second
    # line. Only the nick list is allowed to, which is why it is last.
    return [
        f"Summary     : {summary}",
        f"Highlights  : {snap['highlights']}",
        f"Chat history: {snap['history']}/{snap['history_max']}",
        f"Recall      : {state} — {_plural(snap['recall_lines'], 'line')}",
        f"Pages       : {snap['pages_cached']} cached{web_note}",
    ]


def _summary_report(snap: dict) -> str:
    """The rolling conversation memory as a block of text for the pop-up."""
    summary = snap["summary"].strip()
    if not summary:
        return (
            "No conversation memory yet.\n\n"
            f"{snap['pending_summary']} line(s) are waiting to be summarized. "
            f"The summarizer needs at least {bot.SUMMARIZE_MIN_LINES}, and "
            f"fires every {bot.SUMMARIZE_INTERVAL // 60} minutes or every "
            f"{bot.SUMMARIZE_VOLUME_LINES} lines, whichever comes first."
        )
    out = [f"Updated {_fmt_duration(snap['summary_age'])} ago.", "", summary, ""]
    if snap["highlight_list"]:
        out.append("Highlights:")
        out += [f"  - {h}" for h in snap["highlight_list"]]
        out.append("")
    out.append(f"{snap['pending_summary']} line(s) waiting for the next update.")
    return "\n".join(out)


def _fmt_age(epoch: float) -> str:
    """`epoch` as an age relative to now: "3d", "4h", "12m", "just now".

    Profile times are wall-clock, not the monotonic clock the bot's timers use,
    so they survive a restart -- and an absolute date is less use here than
    "when did I last hear from them".
    """
    if not epoch:
        return "never"
    seconds = max(0.0, time.time() - epoch)
    for size, suffix in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{int(seconds // size)}{suffix} ago"
    return "just now"


def _profiles_report(people: list[dict]) -> str:
    """Everything the bot remembers about the chatters, as one block of text.

    Ordered most-recently-heard-from first. Each person gets their names (the
    one the bot uses first, then the other nicks it has linked to them), how
    much they have said, and the stored lines themselves -- this is the raw
    material a later distilling pass will read, so seeing it verbatim is the
    point of the view.
    """
    if not people:
        return (
            "Nobody on file yet.\n\n"
            "A profile is started the first time somebody says something "
            f"substantial (over {bot.MIN_CHAT_CHARS} characters, more than one "
            f"word). The last {profiles.PROFILE_LINES} such lines are kept per "
            "person, and survive a restart."
        )
    out = []
    for person in people:
        out.append(f"=== {person['nick']} ===")
        others = [a for a in person["aliases"] if a != person["nick"]]
        if others:
            out.append(f"also known as: {', '.join(others)}")
        out.append(
            f"{person['line_count']} lines total, "
            f"{len(person['lines'])} kept | "
            f"first seen {_fmt_age(person['first_seen'])}, "
            f"last {_fmt_age(person['last_seen'])}"
        )
        if person["highlights"]:
            out.append("highlights:")
            out += [f"  - {h}" for h in person["highlights"]]
        if person["quotes"]:
            out.append("quotes:")
            out += [f"  \"{q[1]}\"" for q in person["quotes"]]
        out.append("recent lines:")
        out += [f"  [{_fmt_age(when)}] {text}" for when, text in person["lines"]]
        out.append("")
    return "\n".join(out)


# Static hint row pinned to the bottom of the status pane (see CSS #status-hints).
_STATUS_HINTS = (
    "I = Inspect last LLM call\n"
    "S = Show conversation memory\n"
    "U = Show user profiles\n"
    "C = Configure (edit sloppy.toml)\n"
    "R = Reload configuration\n"
    "V = Toggle vision\n"
    "L = Toggle long-term recall\n"
    "P = Pause / resume\n"
    "Q = Quit"
)


class LLMBotApp(App[None]):
    """The bot's TUI."""

    CSS = """
    #log {
        width: 62%;
        border-right: thick $error;
        background: $panel;
        color: $text;
    }
    #status {
        width: 38%;
        border-left: thick $warning;
        padding: 0 1 0 1;
    }
    #status-info {
        /* Flows naturally from the top of the status pane. */
    }
    #status-hints {
        /* Pinned to the bottom via dock (align is a no-op in this layout). */
        dock: bottom;
    }
    """

    # Both cases of every key: the shift state should never decide whether a
    # hotkey works.
    BINDINGS = [
        ("i", "show_llm_debug", "Inspect LLM call"),
        ("I", "show_llm_debug", "Inspect LLM call"),
        ("s", "show_summary", "Show conversation memory"),
        ("S", "show_summary", "Show conversation memory"),
        ("u", "show_profiles", "Show user profiles"),
        ("U", "show_profiles", "Show user profiles"),
        ("c", "configure", "Edit sloppy.toml"),
        ("C", "configure", "Edit sloppy.toml"),
        ("r", "reload_config", "Reload configuration"),
        ("R", "reload_config", "Reload configuration"),
        ("v", "toggle_vision", "Toggle vision"),
        ("V", "toggle_vision", "Toggle vision"),
        ("l", "toggle_recall", "Toggle long-term recall"),
        ("L", "toggle_recall", "Toggle long-term recall"),
        ("p", "toggle_pause", "Pause / resume"),
        ("P", "toggle_pause", "Pause / resume"),
        ("q", "quit", "Quit"),
        ("Q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield RichLog(id="log", auto_scroll=True, highlight=False)
            with Vertical(id="status"):
                yield Static(id="status-info")
                yield Static(id="status-hints")

    def on_mount(self) -> None:
        # Route the core's meaningful lines into the log via thread-safe
        # messages. The raw IRC sink is dropped — it is noise, not bot state.
        bot.irc_sink = lambda _text: None
        bot.action_sink = self._on_action
        bot.chat_sink = self._on_chat
        bot.speak_sink = self._on_speak
        bot.warning_sink = self._on_warning
        bot.debug_sink = lambda _text: None
        self.set_interval(1.0, self._refresh_status)
        self._refresh_status()
        # Asked before the bot thread starts, so quitting does not leave a
        # half-connected bot behind. The probe is a local HTTP call with a
        # short timeout; doing it on a worker keeps the UI painting meanwhile.
        self.run_worker(self._start_bot, thread=True)

    def _start_bot(self) -> None:
        problem = bot.llm_preflight_problem()
        if problem and bot.LLM_CHECK == "fail":
            self.call_from_thread(self.exit)
            return
        if problem and bot.LLM_CHECK == "ask":
            self.call_from_thread(self._ask_about_llm, problem)
            return
        self.call_from_thread(self._run_bot)

    def _ask_about_llm(self, problem: str) -> None:
        def decided(carry_on: bool | None) -> None:
            if carry_on:
                self._run_bot()
            else:
                self.exit()
        self.push_screen(NoLLMView(problem), decided)

    def _run_bot(self) -> None:
        self._bot_thread = threading.Thread(target=bot.main, daemon=True)
        self._bot_thread.start()

    def _on_action(self, text: str) -> None:
        self.post_message(LogLine(text, action=True))

    def _on_speak(self, text: str) -> None:
        self.post_message(LogLine(text, speak=True))

    def _on_chat(self, text: str) -> None:
        self.post_message(LogLine(text, action=False))

    def _on_warning(self, text: str) -> None:
        self.post_message(LogLine(text, warning=True))

    def on_log_line(self, msg: LogLine) -> None:
        log = self.query_one("#log", RichLog)
        if msg.is_speak:
            # The bot actually spoke: bright blue + bold, so it stands out
            # from the yellow action lines and the plain chat lines. "light_blue"
            # is not a recognised Rich style name (it rendered as plain white).
            log.write(Text(msg.data, style="bold bright_blue"))
        elif msg.is_warning:
            # A rejected summarizer output: bold red, so it stands out from the
            # yellow action lines and the blue speak lines.
            log.write(Text(msg.data, style="bold red"))
        elif msg.is_action:
            # Bold + bright so the bot's own actions stand out from chat.
            # A Text object (not a markup string) keeps brackets in lines
            # like "[AI] ..." literal instead of being parsed as tags.
            log.write(Text(msg.data, style="bold bright_yellow"))
        else:
            log.write(Text(msg.data))

    def _refresh_status(self) -> None:
        snap = bot.status_snapshot()
        self.query_one("#status-info", Static).update(_format_status(snap))
        self.query_one("#status-hints", Static).update(_STATUS_HINTS)

    def action_show_llm_debug(self) -> None:
        """Pop up the last LLM call for inspection (press 'i'/'I')."""
        self.push_screen(LLMDebugView())

    def action_show_summary(self) -> None:
        """Pop up the rolling conversation memory (press 's'/'S')."""
        self.push_screen(SummaryView())

    def action_show_profiles(self) -> None:
        """Pop up what the bot remembers about the chatters (press 'u'/'U')."""
        self.push_screen(ProfilesView())

    def action_configure(self) -> None:
        """Edit sloppy.toml in place (press 'c'/'C'). Saving reloads it."""
        self.push_screen(ConfigView())

    def action_reload_config(self) -> None:
        """Re-read sloppy.toml and apply it (press 'r'/'R').

        Reported through the same sinks as everything else, so what changed --
        or what was wrong with the file -- lands in the log pane.
        """
        bot.report_config(bot.reload_config())

    def action_toggle_vision(self) -> None:
        """Cycle the vision mode auto -> on -> off (press 'v').

        auto follows the server probe (does the loaded model see images?); on
        and off force the behaviour regardless of what the probe reports.
        """
        bot._cycle_vision_override()

    def action_toggle_recall(self) -> None:
        """Cycle long-term recall config -> on -> off (press 'l').

        config follows [recall] enabled in sloppy.toml; on and off force it for
        this session, so it can be judged against itself on the same channel
        without editing a file or restarting. Capture keeps running either way.
        """
        bot._cycle_recall_override()

    def action_toggle_pause(self) -> None:
        """Pause/unpause the bot (press 'p'/'P'). While paused it makes no LLM
        calls and no greetings; press it again to resume. The rolling
        summarizer picks up where it left off on resume."""
        bot._toggle_pause()

    def on_unmount(self) -> None:
        # Not just _stop_event: the profile store is flushed here, on the UI
        # thread, because the worker that would otherwise do it is a daemon and
        # does not outlive the interpreter.
        bot.shutdown()


class NoLLMView(ModalScreen[bool]):
    """Asked at startup when llama.cpp is not answering: carry on, or quit?

    A modal rather than a log line because it is a decision, and because the
    alternative -- finding out from a brain-offline line the channel can see --
    is the thing this exists to avoid. Dismissing with Esc carries on, which is
    the safe reading of walking away: the bot works fine without a model right
    up until somebody talks to it, and it picks one up when it appears.
    """

    BINDINGS = [("escape", "carry_on", "Start anyway")]

    CSS = """
    NoLLMView {
        align: center middle;
    }
    #no_llm {
        width: 70%;
        height: auto;
        padding: 1 2;
        border: thick $error;
        border-title-background: $error;
    }
    #no_llm_buttons {
        height: auto;
        margin-top: 1;
    }
    #no_llm_buttons Button {
        margin-right: 2;
    }
    """

    def __init__(self, problem: str) -> None:
        super().__init__()
        self._problem = problem

    def compose(self) -> ComposeResult:
        with Vertical(id="no_llm"):
            yield Static(
                f"{self._problem}\n\n"
                "The bot will connect and behave normally, but every reply will "
                "fail until a model is in service -- and it says so in the "
                "channel when it does.\n\n"
                "Set connection.llm_check in sloppy.toml to warn, fail or off "
                "to stop being asked."
            )
            with Horizontal(id="no_llm_buttons"):
                yield Button("Start anyway  (Esc)", id="no_llm_go")
                yield Button("Quit", id="no_llm_quit", variant="error")

    def on_mount(self) -> None:
        self.border_title = "No LLM"
        self.query_one("#no_llm", Vertical).border_title = "No LLM"

    def action_carry_on(self) -> None:
        self.dismiss(True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id != "no_llm_quit")


class LLMDebugView(ModalScreen[None]):
    """Modal pop-up: everything passed to and returned by the last LLM call.

    Shows the system prompt, the messages exactly as sent to the model, the
    user prompt, and the returned output. Scroll with the wheel/arrows; close
    with Esc, the X key, or the close button. Literal text (the message repr
    contains brackets) is shown verbatim via markup=False.
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    CSS = """
    #llm_debug {
        height: 85%;
        width: 90%;
        border: thick $error;
        border-title-background: $warning;
    }
    #close-btn {
        dock: bottom;
        margin: 1;
        width: 14;
    }
    """

    def compose(self) -> ComposeResult:
        yield RichLog(id="llm_debug", auto_scroll=True, markup=False, wrap=True)
        yield Button("Close  (Esc)", id="close-btn")

    def on_mount(self) -> None:
        self.border_title = "Last LLM call"
        self.query_one("#llm_debug", RichLog).write(bot.get_last_llm_call())

    def on_button_pressed(self) -> None:
        self.dismiss()


class SummaryView(ModalScreen[None]):
    """Modal pop-up: the rolling conversation memory and its highlights.

    This does not live in the status pane because it does not fit there -- a
    capped summary plus five highlights wraps to more rows than the pane has at
    any ordinary terminal size, and a Static clips instead of scrolling. Here it
    wraps and scrolls. Close with Esc or the close button.
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    CSS = """
    #summary_view {
        height: 80%;
        width: 80%;
        border: thick $success;
        border-title-background: $warning;
    }
    #summary-close {
        dock: bottom;
        margin: 1;
        width: 14;
    }
    """

    def compose(self) -> ComposeResult:
        yield RichLog(id="summary_view", markup=False, wrap=True)
        yield Button("Close  (Esc)", id="summary-close")

    def on_mount(self) -> None:
        self.border_title = "Conversation memory"
        self.query_one("#summary_view", RichLog).write(
            _summary_report(bot.status_snapshot())
        )

    def on_button_pressed(self) -> None:
        self.dismiss()


class ConfigView(ModalScreen[None]):
    """Modal editor for sloppy.toml.

    A plain TextArea in TOML mode -- Textual ships the editor and the syntax
    highlighting, so this is a text box and two buttons rather than anything
    resembling an editor of our own. Save writes the file and reloads it, so a
    change is live without leaving the TUI. Esc closes without saving.
    """

    # priority, or the focused TextArea eats them: it binds escape itself and
    # swallows anything it does not recognise.
    BINDINGS = [
        Binding("escape", "dismiss", "Close without saving", priority=True),
        Binding("ctrl+s", "save", "Save and reload", priority=True),
    ]

    CSS = """
    #config_edit {
        height: 85%;
        width: 90%;
        border: thick $secondary;
        border-title-background: $warning;
    }
    #config-buttons {
        dock: bottom;
        height: 3;
        margin: 1;
    }
    #config-buttons Button {
        width: 22;
        margin-right: 2;
    }
    """

    def compose(self) -> ComposeResult:
        try:
            text = bot.config.default_path().read_text(encoding="utf-8")
        except OSError as exc:
            text = f"# could not read {bot.config.default_path()}: {exc}\n"
        yield TextArea.code_editor(text, language="toml", id="config_edit")
        with Horizontal(id="config-buttons"):
            yield Button("Save + reload", id="config-save", variant="success")
            yield Button("Cancel  (Esc)", id="config-cancel")

    def on_mount(self) -> None:
        self.border_title = f"Configuration — {bot.config.default_path()}"

    def action_save(self) -> None:
        """Write the buffer back and apply it, or say why it could not be."""
        text = self.query_one("#config_edit", TextArea).text
        path = bot.config.default_path()
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            bot.warning(f"[AI] could not write {path.name}: {exc}")
            return
        bot.report_config(bot.reload_config())
        self.dismiss()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "config-save":
            self.action_save()
        else:
            self.dismiss()


class ProfilesView(ModalScreen[None]):
    """Modal pop-up: what the bot remembers about the individual chatters.

    A pop-up rather than a pane for the same reason the summary is one -- the
    stored lines for even a handful of people run to hundreds of rows. Close
    with Esc or the close button.
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    CSS = """
    #profiles_view {
        height: 85%;
        width: 90%;
        border: thick $accent;
        border-title-background: $warning;
    }
    #profiles-close {
        dock: bottom;
        margin: 1;
        width: 14;
    }
    """

    def compose(self) -> ComposeResult:
        yield RichLog(id="profiles_view", markup=False, wrap=True)
        yield Button("Close  (Esc)", id="profiles-close")

    def on_mount(self) -> None:
        self.border_title = "User profiles"
        self.query_one("#profiles_view", RichLog).write(
            _profiles_report(bot.profiles_snapshot())
        )

    def on_button_pressed(self) -> None:
        self.dismiss()


if __name__ == "__main__":
    LLMBotApp().run()
