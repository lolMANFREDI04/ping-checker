// Cloudflare Pages Function: /api/block
// GET  → restituisce stato attuale del blocco
// POST → attiva/disattiva il blocco (body: { action: "block"|"unblock", hours?: number })

export async function onRequestGet(context) {
  const { env } = context;
  const headers = {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  };

  try {
    const blockRaw = await env.PING_KV.get("block");
    if (blockRaw) {
      const blockData = JSON.parse(blockRaw);
      const until = new Date(blockData.until);

      if (until > new Date()) {
        return new Response(
          JSON.stringify({ active: true, until: blockData.until }),
          { headers }
        );
      } else {
        await env.PING_KV.delete("block");
      }
    }
  } catch (e) {
    // KV non disponibile
  }

  return new Response(JSON.stringify({ active: false }), { headers });
}

export async function onRequestPost(context) {
  const { request, env } = context;
  const headers = {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  };

  let body;
  try {
    body = await request.json();
  } catch (e) {
    return new Response(
      JSON.stringify({ success: false, error: "Body JSON non valido." }),
      { status: 400, headers }
    );
  }

  if (body.action === "block") {
    const hours = Math.min(Math.max(parseInt(body.hours) || 1, 1), 720); // min 1h, max 30 giorni
    const until = new Date(Date.now() + hours * 3600000).toISOString();

    try {
      await env.PING_KV.put("block", JSON.stringify({ until }));
    } catch (e) {
      return new Response(
        JSON.stringify({ success: false, error: "Errore KV: " + e.message }),
        { status: 500, headers }
      );
    }

    return new Response(
      JSON.stringify({ success: true, until, hours }),
      { headers }
    );
  }

  if (body.action === "unblock") {
    try {
      await env.PING_KV.delete("block");
    } catch (e) {
      // Ignora errori di delete
    }

    return new Response(JSON.stringify({ success: true }), { headers });
  }

  return new Response(
    JSON.stringify({ success: false, error: "Azione non valida. Usa 'block' o 'unblock'." }),
    { status: 400, headers }
  );
}
