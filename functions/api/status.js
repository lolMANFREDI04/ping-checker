// Cloudflare Pages Function: GET /api/status?name=xxx
// Controlla lo stato Discord via Lanyard, tenendo conto del blocco admin.

const DISCORD_USER_ID = "719637230980825090";
const NOMI_AUTORIZZATI = ["marco", "giulia", "luca", "sara", "andrea", "elena"];

export async function onRequestGet(context) {
  const { request, env } = context;
  const url = new URL(request.url);
  const nome = (url.searchParams.get("name") || "").trim().toLowerCase();

  // Headers CORS
  const headers = {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  };

  // Verifica nome autorizzato
  if (!nome || !NOMI_AUTORIZZATI.includes(nome)) {
    return new Response(
      JSON.stringify({ status: "error", message: "Nome non autorizzato." }),
      { status: 403, headers }
    );
  }

  // Controlla se c'è un blocco attivo (KV)
  try {
    const blockRaw = await env.PING_KV.get("block");
    if (blockRaw) {
      const blockData = JSON.parse(blockRaw);
      const until = new Date(blockData.until);
      const now = new Date();

      if (until > now) {
        const diffMs = until - now;
        const diffH = Math.floor(diffMs / 3600000);
        const diffM = Math.floor((diffMs % 3600000) / 60000);

        return new Response(
          JSON.stringify({
            status: "blocked",
            message: `Non disponibile per ancora ${diffH}h ${diffM}m.`,
            until: blockData.until,
          }),
          { headers }
        );
      } else {
        // Blocco scaduto, rimuovilo
        await env.PING_KV.delete("block");
      }
    }
  } catch (e) {
    // KV non disponibile, continua senza blocco
  }

  // Interroga Lanyard per lo stato Discord
  try {
    const lanyardRes = await fetch(
      `https://api.lanyard.rest/v1/users/${DISCORD_USER_ID}`
    );
    const lanyard = await lanyardRes.json();

    if (!lanyard.success) {
      return new Response(
        JSON.stringify({
          status: "error",
          message:
            "Impossibile verificare lo stato. Assicurati di essere nel server Lanyard su Discord.",
        }),
        { headers }
      );
    }

    const discordStatus = lanyard.data.discord_status; // "online" | "idle" | "dnd" | "offline"
    const isOnline = discordStatus !== "offline";

    return new Response(
      JSON.stringify({
        status: isOnline ? "online" : "offline",
        discord_status: discordStatus,
      }),
      { headers }
    );
  } catch (e) {
    return new Response(
      JSON.stringify({
        status: "error",
        message: "Errore nella connessione a Discord: " + e.message,
      }),
      { status: 500, headers }
    );
  }
}
