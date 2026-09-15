# DiscordSpeakingBridge

A small Vencord userplugin that bridges Discord's own "who is currently
speaking" state (the same signal behind the green speaking ring in the UI)
to `vc-translator`'s `translate_vc.py`, over a local WebSocket. This is the
*only* source of speaker labels the translator has - see the main repo
README's "Speaker identification" section for how it's used on the Python
side.

It only ever sends `{userId, username, speaking, timestamp}` for the voice
channel you're already in - no audio, no message content, nothing else.

## Setup

1. Clone Vencord if you don't already have a dev checkout:
   ```
   git clone https://github.com/Vendicated/Vencord
   cd Vencord
   pnpm install
   ```
2. Copy `discordSpeakingBridge.ts` into `src/userplugins/`:
   ```
   mkdir -p src/userplugins
   cp /path/to/vc-translator/vencord-plugin/discordSpeakingBridge.ts src/userplugins/
   ```
3. Build and inject per Vencord's own instructions (`pnpm build`, then
   `pnpm inject` or run it via the Vencord installer, depending on how your
   Discord client is set up).
4. In Discord, open Vencord's settings → Plugins, find
   **DiscordSpeakingBridge**, and enable it. Its one setting is the port
   `translate_vc.py`'s bridge server listens on - defaults to `8765`,
   matching `translate_vc.py`'s own default (`--discord-bridge-port`).

## Verifying it's actually working

Confirmed live: Discord's `SPEAKING` FluxDispatcher event fires as
`{type: "SPEAKING", context, userId, speakingFlags, voiceDb}` (no
`channelId` - it only ever fires for the channel you're currently
connected to, which `handleSpeaking` relies on). If real usernames never
show up in the overlay while the plugin shows as connected:

1. Open Discord's DevTools console (Ctrl+Shift+I in the desktop client).
2. Join a voice channel with at least one other person talking, and watch
   for `[DiscordSpeakingBridge] connected` in the console.
3. If that shape ever changes in a future Discord client update, paste
   this into the console while someone talks to see what's actually firing,
   and adjust `handleSpeaking` accordingly:
   ```js
   const orig = Vencord.Webpack.Common.FluxDispatcher.dispatch;
   Vencord.Webpack.Common.FluxDispatcher.dispatch = function(e) {
     if (/SPEAK|VOICE/i.test(e.type)) console.log(e.type, e);
     return orig.call(this, e);
   };
   ```

## Without this plugin

Everything else in `vc-translator` still works - system-audio captions are
just unlabeled (no speaker name) whenever this bridge isn't connected or
the plugin isn't enabled. Your own `[You]` mic captions are unaffected
either way.
