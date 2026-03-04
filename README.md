# Ping Checker

Verifica se il PC è acceso controllando lo stato Discord tramite [Lanyard](https://github.com/Phineas/lanyard).

## Struttura

```
ping-checker/
├── index.html              ← Pagina principale (input nome)
├── admin.html              ← Pannello admin (blocco per X ore)
├── functions/api/
│   ├── status.js           ← API: controlla stato Discord + blocco
│   └── block.js            ← API: gestione blocco (GET/POST)
├── package.json
└── wrangler.toml
```

## Setup & Deploy su Cloudflare Pages

### 1. Prerequisito: Lanyard
Entra nel server Discord di **Lanyard**: https://discord.gg/lanyard  
Questo permette a Lanyard di monitorare il tuo stato Discord e renderlo disponibile via API.

### 2. Installa Wrangler (CLI Cloudflare)
```bash
npm install -g wrangler
wrangler login
```

### 3. Crea il namespace KV
```bash
wrangler kv namespace create PING_KV
```
Copia l'ID restituito e sostituiscilo in `wrangler.toml` al posto di `DA_SOSTITUIRE_CON_ID_REALE`.

### 4. Test in locale
```bash
cd ping-checker
npm run dev
```
Apri http://localhost:8080

### 5. Deploy in produzione
```bash
npm run deploy
```
Al primo deploy ti chiederà di creare il progetto. Poi:

1. Vai nella **Cloudflare Dashboard** → Pages → ping-checker → Settings → Functions
2. Aggiungi il binding KV:
   - **Variable name**: `PING_KV`
   - **KV namespace**: seleziona quello creato al passo 3
3. Rideploya o triggera un nuovo deploy

### 6. Dominio personalizzato
Dashboard → Pages → ping-checker → Custom domains → Aggiungi il tuo dominio.

## Uso

| Nome inserito | Azione |
|---|---|
| Nome autorizzato (Marco, Giulia, ecc.) | Controlla stato Discord → mostra se PC acceso/spento |
| `lool` | Apre il pannello admin |
| Nome non valido | Errore "non autorizzato" |

### Pannello Admin
- **Blocca per X ore**: chiunque inserisca un nome valido vedrà "Non disponibile"
- **Sblocca**: ripristina il check normale dello stato Discord
