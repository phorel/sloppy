<img width="20%" alt="sloppy-logo-v3" src="https://github.com/user-attachments/assets/03b2044c-ed52-48db-afbe-9a8a332c9d2d" />

# Sloppy the moody bot

**v0.7.5**

Sloppy is a sarcastic, moody and sometimes funny AI bot for IRC. It connects to a local LLM through llama.cpp and brings a unique flavour of awkward crude humor and genuinely useful features. It understands the chat and has persistent context, making it able to chime in or roast people based on things they said earlier.

Works with any LLM running under llama.cpp. The better the model, the better the bot will work. 35B MoE models have proven to be very entertaining chatters that are able to understand the context of a chaotic chat. 9B models work fine aswell, although they're not as good at distilling the chat. It has not been tested with smaller models than 9B, but it should work, results may vary. Personalities are defined by prompts and easy to change in the configuration .toml file.

**Features include**

- Moods that randomly change to defined moods, duration of each mood can be set in config
- A rolling summary of the chat is kept to give the bot contextual awareness
- Optional logging with pattern matching relevancy calculation for longterm context
- Answers whoever addressed it, and knows who they are: the person asking and anyone
  they ask about arrive with their own past lines, so a reply lands on them specifically
- Can be asked what somebody said and when they said it -- "what did alice say about her
  boyfriend last week", "when did i say i was going to amsterdam" -- answered from the log
- Privacy commands allow people to ask what the bot knows about them and make them forget
- "FactCheck, Serious, Research, Science" and similar terms will make the bot respond seriously
- !image <url> analyzes an image and describes the content
- !summarize <url> summarizes a website
- !translate <text> translates words or sentences to any language (default english)
- !Quote, !Buddha gives random quotes
- !Factoid says a random (hopefully interesting or funny) factoid.
- !help displays the bot's commands
- Interjections when it just joins or when the chat is slow and can use a boost
- A TUI interface showing status, LLM calls and responses, ability to enable/disable the vision component and other settings. Includes a configuration text editor for the .toml file.
- Can run headless in tmux or as a service. Automatically detects existing tmux session and reattaches instead of starting a new instance
- Owners can be configured and can talk to the bot in PM. Optionally this can be set to *!*@*
- Owners can !purge [nick] (days) data from the bot's memory if needed
  

**Usage**

With the TUI, in a terminal:

    python3 llmbot_tui.py

Headless, with no terminal to keep open:

    python3 llmbot_core.py              # log to stdout
    python3 llmbot_core.py --log ~/.local/state/sloppy.log
    python3 llmbot_core.py --verbose    # include the full prompt dumps

As a service, which is the tidy way to leave it running -- see
`sloppy.service.example` for a systemd user unit and the commands to install
it. It restarts on failure and shuts down on SIGTERM, flushing the profiles
and the channel memory on the way out.

There is no way to attach the TUI to a bot that is already running headless:
the TUI reads the core's state in-process, and a detach/attach protocol is a
much bigger feature than this. If you want a UI you can come back to, run the
TUI under tmux -- `sloppy.sh` does it for you:

    ./sloppy.sh              start it and attach
    ./sloppy.sh --detach     start it and leave it in the background
    ./sloppy.sh --status     is it running
    ./sloppy.sh --stop       stop it

`ctrl-b d` leaves it running and gives you the terminal back; `./sloppy.sh`
again puts you back in it. Running it twice will not start a second bot on the
same channel. `SLOPPY_TMUX_SESSION` renames the session if you want more than
one.

Use tmux if you want a UI to come back to; use the systemd unit if you want
something that survives a reboot and restarts itself.

**If llama.cpp is not running**

The bot works fine without a model right up until somebody talks to it, at
which point it says a line about its brain being offline -- in the channel, in
character, which is a poor place to learn the server is down. So it looks
first. `connection.llm_check` in `sloppy.toml` decides what happens:

| | |
|---|---|
| `ask` (default) | warn and let you decide -- a pop-up in the TUI, a `y/N` on the terminal headless. Started by a service manager there is nobody to ask, so it warns and carries on. |
| `warn` | say so and carry on |
| `fail` | refuse to start (exit 1). With `Restart=on-failure` the unit keeps trying until llama.cpp is up. |
| `off` | do not look |

**If llama.cpp goes away while it is running**

A model that dies mid-session is worse than one that never started: the bot
sits in the channel answering every question with a line about its brain being
offline. After `connection.llm_part_after_seconds` (300 by default, 0 to switch
it off) of the endpoint not answering, it parts the channel with
`brain offline, back when the model is` and waits. It stays connected to IRC
and keeps asking `/props` once a minute, so it rejoins by itself the moment a
model answers again. An empty chair says the bot is not working, and says it
once.

**Owner commands**

Set `[owners] masks` (see `sloppy.toml`) and those hostmasks get:

    !purge <nick>          erase them from everything the bot remembers
    !purge <nick> 3        ...but only the last 3 days of it

It clears the long-term log, the recent-line buffer and their profile, then
rebuilds the rolling summary from the lines that remain -- the summary rides in
the system message of every reply, so it is where something planted in the
bot's memory keeps working, and waiting for it to age out is not an answer.

    !ignore <nick or mask>     stop hearing somebody, now
    !unignore <nick or mask>   take that back
    !ignored                   who is on the list, and where each came from

An ignored nick is dropped at the door: not answered, not remembered, not
counted towards how talkative the channel is, not logged for recall, not
summarised, no JOIN greeting, no private message. Set the permanent ones in
`[ignore] masks` (globs against `nick!user@host`, and a bare `spammer` means
that nick from anywhere); `!ignore` is the quick one for somebody abusing the
bot right now, and it is written to `ignores.json` so it survives a restart.
`!unignore` takes back what `!ignore` added; a mask from the config file is the
config file's to remove. Owners are never ignored, whatever the list says.

Owners are also the only people the bot answers in a private query, and it
answers them there rather than in the channel.

NB
'bot.py' contains a very early legacy version of the bot from before it got TUI, it misses most of the features that make sloppy more than just a basic chatbot. 

Sloppy the bot mostly proudly coded itself: First versions coded by KAT Coder 2.5 and Tiel Coder. Claude was then used for quality assurance and is used for the rest of the bot's development.
