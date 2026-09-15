/*
 * DiscordSpeakingBridge - Vencord userplugin
 *
 * Bridges Discord's own "who is currently speaking" state - the same
 * signal that drives the green speaking ring in the UI - to a local
 * companion process (vc-translator's translate_vc.py) over a WebSocket, so
 * it can use ground-truth speaker identity/timing instead of guessing from
 * a mixed-audio voiceprint. Purely one-way telemetry: no audio ever leaves
 * the client, only {userId, username, speaking, timestamp} for the voice
 * channel you're already in.
 *
 * Setup: copy this file to <vencord checkout>/src/userplugins/
 * discordSpeakingBridge.ts, `pnpm build`, then enable "DiscordSpeakingBridge"
 * in Vencord's plugin settings. See vencord-plugin/README.md.
 *
 * The FluxDispatcher "SPEAKING" event's shape was confirmed live:
 * {type: "SPEAKING", context, userId, speakingFlags, voiceDb} - notably no
 * channelId, so handleSpeaking below relies on it only ever firing for the
 * channel you're currently connected to.
 */

import { definePluginSettings } from "@api/Settings";
import definePlugin, { OptionType } from "@utils/types";
import { findStoreLazy } from "@webpack";
import { ChannelStore, FluxDispatcher, GuildMemberStore, SelectedChannelStore, UserStore } from "@webpack/common";

const VoiceStateStore = findStoreLazy("VoiceStateStore");

const settings = definePluginSettings({
    port: {
        type: OptionType.NUMBER,
        description: "Port the local translate_vc.py bridge server listens on",
        default: 8765,
    },
});

let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let lastSyncedChannel: string | null = null;

function displayName(userId: string, guildId: string | null): string {
    if (guildId) {
        const nick = GuildMemberStore.getNick(guildId, userId);
        if (nick) return nick;
    }
    const user = UserStore.getUser(userId);
    return (user as any)?.globalName || user?.username || userId;
}

function send(payload: Record<string, unknown>) {
    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
    }
}

function connect() {
    if (ws) return;
    try {
        ws = new WebSocket(`ws://127.0.0.1:${settings.store.port}`);
    } catch (e) {
        console.error("[DiscordSpeakingBridge] failed to open socket", e);
        scheduleReconnect();
        return;
    }
    ws.onopen = () => {
        console.log("[DiscordSpeakingBridge] connected");
        syncChannel(SelectedChannelStore.getVoiceChannelId());
    };
    ws.onclose = ws.onerror = () => {
        ws = null;
        scheduleReconnect();
    };
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connect();
    }, 3000);
}

function syncChannel(channelId: string | null) {
    lastSyncedChannel = channelId;
    if (!channelId) return;

    const guildId = ChannelStore.getChannel(channelId)?.guild_id ?? null;
    const states = VoiceStateStore.getVoiceStatesForChannel(channelId) ?? {};
    const members: Record<string, string> = {};
    for (const userId in states) {
        members[userId] = displayName(userId, guildId);
    }
    send({ event: "channel_sync", members });
}

function handleSpeaking({ userId, speakingFlags }: { userId: string; speakingFlags: number; }) {
    // the real SPEAKING dispatch carries no channelId (confirmed live: only
    // {type, context, userId, speakingFlags, voiceDb}) - it's only ever
    // fired for the channel you're currently connected to anyway
    const myChannel = SelectedChannelStore.getVoiceChannelId();
    if (!myChannel) return;
    if (userId === UserStore.getCurrentUser()?.id) return;

    if (myChannel !== lastSyncedChannel) syncChannel(myChannel);

    const guildId = ChannelStore.getChannel(myChannel)?.guild_id ?? null;
    send({
        event: speakingFlags ? "speaking_start" : "speaking_stop",
        userId,
        username: displayName(userId, guildId),
        ts: Date.now(),
    });
}

function handleVoiceStateUpdates() {
    const myChannel = SelectedChannelStore.getVoiceChannelId();
    if (myChannel !== lastSyncedChannel) syncChannel(myChannel);
}

export default definePlugin({
    name: "DiscordSpeakingBridge",
    description:
        "Bridges who's-currently-speaking + username to a local WebSocket server (vc-translator), for ground-truth speaker identity instead of voice-fingerprint guessing.",
    authors: [],
    settings,

    start() {
        connect();
        FluxDispatcher.subscribe("SPEAKING", handleSpeaking);
        FluxDispatcher.subscribe("VOICE_STATE_UPDATES", handleVoiceStateUpdates);
    },

    stop() {
        FluxDispatcher.unsubscribe("SPEAKING", handleSpeaking);
        FluxDispatcher.unsubscribe("VOICE_STATE_UPDATES", handleVoiceStateUpdates);
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
        ws?.close();
        ws = null;
    },
});
