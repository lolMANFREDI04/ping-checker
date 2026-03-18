"""
Qwen3-TTS Voice Clone - GUI Avanzata
======================================
Interfaccia grafica Gradio completa per:
  - Gestione profili voce con salvataggio persistente su disco (cache prompt)
  - Generazione rapida e con voce salvata
  - Toggle streaming / non-streaming
  - Analytics con grafici tempi di generazione
  - Guida al perfezionamento voci

Configurato per CPU (senza CUDA/ROCm).

Uso:
    python gui_tts.py
    Poi apri il browser all'indirizzo mostrato nel terminale.
"""

import os
import json
import time
import shutil
import torch
import soundfile as sf
import numpy as np
import gradio as gr
from pathlib import Path
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

# --- Configurazione ---
VOICES_DIR = Path("voices")
OUTPUT_DIR = Path("outputs")
STATS_FILE = Path("generation_stats.json")

VOICES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German",
             "French", "Russian", "Portuguese", "Spanish", "Italian"]

MODEL_CHOICES = {
    "1.7B (migliore qualità)": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B (più veloce, meno RAM)": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

# --- Stato globale ---
model = None
loaded_model_name = None  # chiave attualmente caricata in MODEL_CHOICES
voice_prompts_cache = {}  # nome -> VoiceClonePromptItem (cache in-memory)


# =============================================================================
# DATABASE VOCI (metadati JSON per voce)
# =============================================================================

def _voice_meta_path(name: str) -> Path:
    return VOICES_DIR / name / "metadata.json"


def load_voice_meta(name: str) -> dict:
    p = _voice_meta_path(name)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_voice_meta(name: str, meta: dict):
    with open(_voice_meta_path(name), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _migrate_old_prompt_files(name: str):
    """Rinomina i vecchi ref_code.pt / ref_spk_embedding.pt aggiungendo il suffisso modello."""
    voice_dir = VOICES_DIR / name
    old_code = voice_dir / "ref_code.pt"
    old_emb = voice_dir / "ref_spk_embedding.pt"
    if old_code.exists() and old_emb.exists():
        meta = load_voice_meta(name)
        mtag = meta.get("model", "1.7B")  # fallback al modello originale
        old_code.rename(voice_dir / f"ref_code_{mtag}.pt")
        old_emb.rename(voice_dir / f"ref_spk_embedding_{mtag}.pt")


def get_voice_names():
    names = []
    if VOICES_DIR.exists():
        for d in sorted(VOICES_DIR.iterdir()):
            if d.is_dir() and (d / "metadata.json").exists():
                _migrate_old_prompt_files(d.name)
                names.append(d.name)
    return names


# =============================================================================
# STATISTICHE GENERAZIONE
# =============================================================================

def load_stats() -> list:
    if STATS_FILE.exists():
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_stat(entry: dict):
    stats = load_stats()
    stats.append(entry)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


# =============================================================================
# CARICAMENTO/SALVATAGGIO PROMPT SU DISCO
# =============================================================================

def save_prompt_to_disk(name: str, item: VoiceClonePromptItem, model_tag: str):
    voice_dir = VOICES_DIR / name
    torch.save(item.ref_code, voice_dir / f"ref_code_{model_tag}.pt")
    torch.save(item.ref_spk_embedding, voice_dir / f"ref_spk_embedding_{model_tag}.pt")


def load_prompt_from_disk(name: str, model_tag: str) -> VoiceClonePromptItem:
    voice_dir = VOICES_DIR / name
    meta = load_voice_meta(name)
    ref_code = torch.load(voice_dir / f"ref_code_{model_tag}.pt", weights_only=True)
    ref_spk_embedding = torch.load(voice_dir / f"ref_spk_embedding_{model_tag}.pt", weights_only=True)
    return VoiceClonePromptItem(
        ref_code=ref_code,
        ref_spk_embedding=ref_spk_embedding,
        x_vector_only_mode=meta.get("x_vector_only_mode", False),
        icl_mode=meta.get("icl_mode", True),
        ref_text=meta.get("ref_text", ""),
    )


def has_prompt_on_disk(name: str, model_tag: str) -> bool:
    voice_dir = VOICES_DIR / name
    return (voice_dir / f"ref_code_{model_tag}.pt").exists() and \
           (voice_dir / f"ref_spk_embedding_{model_tag}.pt").exists()


def get_voice_prompt(name: str, progress=None) -> VoiceClonePromptItem:
    """Ottiene il prompt per il modello attualmente caricato. Ricalcola se necessario."""
    mtag = get_current_model_tag()
    cache_key = f"{name}_{mtag}"

    if cache_key in voice_prompts_cache:
        return voice_prompts_cache[cache_key]

    # Prova a caricare da disco per questo modello
    if has_prompt_on_disk(name, mtag):
        item = load_prompt_from_disk(name, mtag)
        voice_prompts_cache[cache_key] = item
        return item

    # Prompt non esiste per questo modello: ricalcola dall'audio di riferimento
    meta = load_voice_meta(name)
    audio_path = meta.get("audio_path", "")
    ref_text = meta.get("ref_text", "")
    if not audio_path or not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio di riferimento non trovato per '{name}': {audio_path}")

    if progress:
        progress(0.2, desc=f"Ricalcolo prompt per modello {mtag}...")

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=audio_path,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    item = prompt_items[0]
    save_prompt_to_disk(name, item, mtag)
    voice_prompts_cache[cache_key] = item
    return item
    return item


# =============================================================================
# FUNZIONI MODELLO
# =============================================================================

def load_model(model_choice, progress=gr.Progress(track_tqdm=True)):
    global model, loaded_model_name, voice_prompts_cache
    model_id = MODEL_CHOICES.get(model_choice, list(MODEL_CHOICES.values())[0])
    short = "1.7B" if "1.7B" in model_id else "0.6B"

    if model is not None and loaded_model_name == model_choice:
        return f"✅ Modello {short} già caricato in memoria."

    # Se cambiamo modello, svuota cache prompt (i prompt non sono intercambiabili)
    if model is not None and loaded_model_name != model_choice:
        voice_prompts_cache.clear()

    progress(0, desc=f"Scaricamento/caricamento {short}...")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cpu",
        dtype=torch.float32,
    )
    loaded_model_name = model_choice
    progress(1, desc="Modello pronto!")
    return f"✅ Modello {short} caricato con successo (CPU, float32)."


def get_current_model_tag() -> str:
    """Restituisce il tag breve del modello attualmente caricato."""
    if loaded_model_name is None:
        return "?"
    return "1.7B" if "1.7B" in loaded_model_name else "0.6B"


def add_voice(name, audio_file, ref_text, progress=gr.Progress()):
    if not name or not name.strip():
        return "❌ Inserisci un nome per la voce.", gr.update()
    name = name.strip()
    if audio_file is None:
        return "❌ Carica un file audio di riferimento.", gr.update()
    if not ref_text or not ref_text.strip():
        return "❌ Inserisci la trascrizione dell'audio di riferimento.", gr.update()
    if model is None:
        return "❌ Carica prima il modello.", gr.update()

    progress(0.1, desc="Salvataggio audio di riferimento...")
    voice_dir = VOICES_DIR / name
    voice_dir.mkdir(exist_ok=True)
    audio_path = voice_dir / "reference.wav"

    if isinstance(audio_file, tuple):
        sr_in, audio_data = audio_file
        sf.write(str(audio_path), audio_data, sr_in)
    elif isinstance(audio_file, str):
        shutil.copy2(audio_file, str(audio_path))
    else:
        return "❌ Formato audio non riconosciuto.", gr.update()

    progress(0.3, desc="Creazione prompt voce (può richiedere tempo su CPU)...")
    t0 = time.time()
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=str(audio_path),
        ref_text=ref_text.strip(),
        x_vector_only_mode=False,
    )
    prompt_time = time.time() - t0
    item = prompt_items[0]

    progress(0.8, desc="Salvataggio prompt su disco...")
    mtag = get_current_model_tag()
    save_prompt_to_disk(name, item, mtag)
    voice_prompts_cache[f"{name}_{mtag}"] = item

    # Calcola info audio
    audio_data_check, sr_check = sf.read(str(audio_path))
    duration = len(audio_data_check) / sr_check

    mtag = get_current_model_tag()
    meta = {
        "name": name,
        "ref_text": ref_text.strip(),
        "audio_path": str(audio_path),
        "model": mtag,
        "x_vector_only_mode": item.x_vector_only_mode,
        "icl_mode": item.icl_mode,
        "ref_code_shape": list(item.ref_code.shape) if item.ref_code is not None else None,
        "ref_spk_embedding_shape": list(item.ref_spk_embedding.shape),
        "audio_duration_s": round(duration, 1),
        "audio_sample_rate": sr_check,
        "prompt_creation_time_s": round(prompt_time, 1),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_voice_meta(name, meta)

    progress(1.0, desc="Fatto!")
    new_choices = get_voice_names()
    cache_kb = sum(f.stat().st_size for f in voice_dir.iterdir()) / 1024
    return (
        f"✅ Voce '{name}' salvata! Prompt su disco: {cache_kb:.0f} KB, "
        f"tempo calcolo: {prompt_time:.1f}s, durata audio: {duration:.1f}s",
        gr.update(choices=new_choices, value=name),
    )


def delete_voice(name):
    if not name:
        return "❌ Seleziona una voce da eliminare.", gr.update()
    voice_dir = VOICES_DIR / name
    if voice_dir.exists():
        shutil.rmtree(voice_dir)
    if name in voice_prompts_cache:
        del voice_prompts_cache[name]
    # Rimuovi anche tutte le cache per-modello
    keys_to_del = [k for k in voice_prompts_cache if k.startswith(f"{name}_")]
    for k in keys_to_del:
        del voice_prompts_cache[k]
    new_choices = get_voice_names()
    return (
        f"✅ Voce '{name}' eliminata.",
        gr.update(choices=new_choices, value=new_choices[0] if new_choices else None),
    )


def generate_speech(voice_name, text, language, streaming_mode, progress=gr.Progress()):
    if model is None:
        return "❌ Carica prima il modello.", None
    if not voice_name:
        return "❌ Seleziona un profilo voce.", None
    if not text or not text.strip():
        return "❌ Inserisci il testo da sintetizzare.", None

    meta = load_voice_meta(voice_name)
    if not meta:
        return f"❌ Voce '{voice_name}' non trovata.", None

    lang = language if language != "Auto" else "Auto"
    non_streaming = not streaming_mode  # UI: True=streaming, parametro: True=non-streaming

    progress(0.1, desc="Caricamento prompt voce da cache...")
    t_load = time.time()
    prompt_item = get_voice_prompt(voice_name, progress=progress)
    load_time = time.time() - t_load

    mode_label = "Streaming" if streaming_mode else "Non-Streaming"
    progress(0.3, desc=f"Generazione {mode_label} in corso...")

    t_gen = time.time()
    wavs, sr = model.generate_voice_clone(
        text=text.strip(),
        language=lang,
        voice_clone_prompt=[prompt_item],
        non_streaming_mode=non_streaming,
    )
    gen_time = time.time() - t_gen
    audio_duration = len(wavs[0]) / sr

    progress(0.9, desc="Salvataggio audio...")
    timestamp = int(time.time())
    output_path = OUTPUT_DIR / f"{voice_name}_{timestamp}.wav"
    sf.write(str(output_path), wavs[0], sr)

    # Salva statistica
    mtag = get_current_model_tag()
    save_stat({
        "voice": voice_name,
        "model": mtag,
        "text_len": len(text.strip()),
        "language": lang,
        "streaming": streaming_mode,
        "prompt_load_time_s": round(load_time, 3),
        "generation_time_s": round(gen_time, 1),
        "audio_duration_s": round(audio_duration, 1),
        "rtf": round(gen_time / max(audio_duration, 0.1), 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_file": output_path.name,
    })

    progress(1.0, desc="Fatto!")
    badge = " ⚡0.6B" if mtag == "0.6B" else ""
    status = (
        f"✅ [{mtag}] {mode_label} | Generato in {gen_time:.1f}s | "
        f"Audio: {audio_duration:.1f}s | RTF: {gen_time / max(audio_duration, 0.1):.1f}x | "
        f"Prompt caricato in {load_time:.3f}s{badge}"
    )
    return status, (sr, wavs[0])


def generate_speech_direct(audio_file, ref_text, text, language, streaming_mode, progress=gr.Progress()):
    if model is None:
        return "❌ Carica prima il modello.", None
    if audio_file is None:
        return "❌ Carica un file audio di riferimento.", None
    if not text or not text.strip():
        return "❌ Inserisci il testo da sintetizzare.", None

    lang = language if language != "Auto" else "Auto"
    non_streaming = not streaming_mode

    mode_label = "Streaming" if streaming_mode else "Non-Streaming"
    progress(0.1, desc=f"Preparazione ({mode_label})...")

    if isinstance(audio_file, tuple):
        sr_in, audio_data = audio_file
        ref_audio_input = (audio_data, sr_in)
    elif isinstance(audio_file, str):
        ref_audio_input = audio_file
    else:
        return "❌ Formato audio non supportato.", None

    kwargs = {
        "text": text.strip(),
        "language": lang,
        "ref_audio": ref_audio_input,
        "non_streaming_mode": non_streaming,
    }
    if ref_text and ref_text.strip():
        kwargs["ref_text"] = ref_text.strip()
    else:
        kwargs["x_vector_only_mode"] = True

    progress(0.3, desc=f"Generazione {mode_label} in corso...")
    t_gen = time.time()
    wavs, sr = model.generate_voice_clone(**kwargs)
    gen_time = time.time() - t_gen
    audio_duration = len(wavs[0]) / sr

    progress(0.9, desc="Salvataggio...")
    timestamp = int(time.time())
    output_path = OUTPUT_DIR / f"direct_{timestamp}.wav"
    sf.write(str(output_path), wavs[0], sr)

    mtag = get_current_model_tag()
    save_stat({
        "voice": "(diretto)",
        "model": mtag,
        "text_len": len(text.strip()),
        "language": lang,
        "streaming": streaming_mode,
        "prompt_load_time_s": 0,
        "generation_time_s": round(gen_time, 1),
        "audio_duration_s": round(audio_duration, 1),
        "rtf": round(gen_time / max(audio_duration, 0.1), 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_file": output_path.name,
    })

    progress(1.0, desc="Fatto!")
    badge = " ⚡0.6B" if mtag == "0.6B" else ""
    status = (
        f"✅ [{mtag}] {mode_label} | Generato in {gen_time:.1f}s | "
        f"Audio: {audio_duration:.1f}s | RTF: {gen_time / max(audio_duration, 0.1):.1f}x{badge}"
    )
    return status, (sr, wavs[0])


# =============================================================================
# ANALYTICS
# =============================================================================

def build_analytics():
    stats = load_stats()
    if not stats:
        return "Nessuna generazione registrata ancora.", None, None, None, None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # --- Grafico 1: Tempo generazione per voce ---
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    voices = {}
    for s in stats:
        v = s.get("voice", "?")
        voices.setdefault(v, []).append(s["generation_time_s"])
    voice_names = list(voices.keys())
    voice_means = [np.mean(t) for t in voices.values()]
    voice_counts = [len(t) for t in voices.values()]
    colors = plt.cm.Set2(np.linspace(0, 1, len(voice_names)))
    bars = ax1.bar(voice_names, voice_means, color=colors)
    for bar, count in zip(bars, voice_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"n={count}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Tempo medio (s)")
    ax1.set_title("Tempo Medio di Generazione per Voce")
    ax1.tick_params(axis="x", rotation=30)
    fig1.tight_layout()

    # --- Grafico 2: Streaming vs Non-Streaming ---
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    stream_times = [s["generation_time_s"] for s in stats if s.get("streaming")]
    nostream_times = [s["generation_time_s"] for s in stats if not s.get("streaming")]
    data_box = []
    labels_box = []
    if nostream_times:
        data_box.append(nostream_times)
        labels_box.append(f"Non-Stream\n(n={len(nostream_times)})")
    if stream_times:
        data_box.append(stream_times)
        labels_box.append(f"Streaming\n(n={len(stream_times)})")
    if data_box:
        bp = ax2.boxplot(data_box, labels=labels_box, patch_artist=True)
        box_colors = ["#66b3ff", "#99ff99"]
        for patch, color in zip(bp["boxes"], box_colors[:len(data_box)]):
            patch.set_facecolor(color)
    ax2.set_ylabel("Tempo (s)")
    ax2.set_title("Confronto Streaming vs Non-Streaming")
    fig2.tight_layout()

    # --- Grafico 3: RTF e lunghezza testo ---
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    text_lens = [s.get("text_len", 0) for s in stats]
    rtfs = [s.get("rtf", 0) for s in stats]
    models = [s.get("model", "1.7B") for s in stats]
    markers = {"1.7B": "o", "0.6B": "s"}
    color_map = {"1.7B": "#3498db", "0.6B": "#e67e22"}
    for m_tag in ["1.7B", "0.6B"]:
        idxs = [i for i, m in enumerate(models) if m == m_tag]
        if idxs:
            ax3.scatter(
                [text_lens[i] for i in idxs], [rtfs[i] for i in idxs],
                c=color_map[m_tag], marker=markers[m_tag],
                alpha=0.7, edgecolors="k", s=60, label=m_tag,
            )
    ax3.set_xlabel("Lunghezza testo (caratteri)")
    ax3.set_ylabel("RTF (tempo gen / durata audio)")
    ax3.set_title("Real-Time Factor vs Lunghezza Testo (per modello)")
    ax3.legend(loc="upper right")
    fig3.tight_layout()

    # --- Grafico 4: Confronto modelli 1.7B vs 0.6B ---
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    times_17 = [s["generation_time_s"] for s in stats if s.get("model") == "1.7B"]
    times_06 = [s["generation_time_s"] for s in stats if s.get("model") == "0.6B"]
    data_model = []
    labels_model = []
    if times_17:
        data_model.append(times_17)
        labels_model.append(f"1.7B\n(n={len(times_17)})")
    if times_06:
        data_model.append(times_06)
        labels_model.append(f"0.6B\n(n={len(times_06)})")
    if data_model:
        bp4 = ax4.boxplot(data_model, labels=labels_model, patch_artist=True)
        m_colors = ["#3498db", "#e67e22"]
        for patch, color in zip(bp4["boxes"], m_colors[:len(data_model)]):
            patch.set_facecolor(color)
    else:
        ax4.text(0.5, 0.5, "Nessun dato", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_ylabel("Tempo (s)")
    ax4.set_title("Confronto Modelli: 1.7B vs 0.6B")
    fig4.tight_layout()

    # Riepilogo testuale
    total = len(stats)
    avg_gen = np.mean([s["generation_time_s"] for s in stats])
    avg_rtf = np.mean([s["rtf"] for s in stats])
    avg_dur = np.mean([s["audio_duration_s"] for s in stats])
    summary = (
        f"**Totale generazioni:** {total}\n\n"
        f"**Tempo medio generazione:** {avg_gen:.1f}s\n\n"
        f"**RTF medio:** {avg_rtf:.1f}x (tempo / durata audio)\n\n"
        f"**Durata media audio:** {avg_dur:.1f}s\n\n"
    )
    if times_17 and times_06:
        summary += (
            f"**1.7B media:** {np.mean(times_17):.1f}s (RTF {np.mean([s['rtf'] for s in stats if s.get('model')=='1.7B']):.1f}x) vs "
            f"**0.6B media:** {np.mean(times_06):.1f}s (RTF {np.mean([s['rtf'] for s in stats if s.get('model')=='0.6B']):.1f}x)\n\n"
        )
    if stream_times and nostream_times:
        summary += (
            f"**Streaming:** {np.mean(stream_times):.1f}s vs "
            f"**Non-Streaming:** {np.mean(nostream_times):.1f}s"
        )

    return summary, fig1, fig2, fig3, fig4


def clear_stats():
    if STATS_FILE.exists():
        STATS_FILE.unlink()
    return "✅ Statistiche cancellate.", None, None, None, None, ""


def get_voice_details(name):
    if not name:
        return ""
    meta = load_voice_meta(name)
    if not meta:
        return "Profilo non trovato."
    voice_dir = VOICES_DIR / name
    cache_kb = sum(f.stat().st_size for f in voice_dir.iterdir() if f.is_file()) / 1024
    m = meta.get('model', '?')
    badge = " ⚡" if m == "0.6B" else ""
    # Mostra per quali modelli esistono prompt salvati
    available_models = []
    for tag in ("1.7B", "0.6B"):
        if has_prompt_on_disk(name, tag):
            available_models.append(tag)
    avail_str = ", ".join(available_models) if available_models else "nessuno"
    return (
        f"**{meta.get('name', name)}** {f'`[{m}]{badge}`' if m != '?' else ''}\n\n"
        f"- Trascrizione: _{meta.get('ref_text', '?')}_\n"
        f"- Durata audio ref: {meta.get('audio_duration_s', '?')}s\n"
        f"- Sample rate: {meta.get('audio_sample_rate', '?')} Hz\n"
        f"- Modello originale: **{m}**{badge}\n"
        f"- Prompt disponibili per: **{avail_str}**\n"
        f"- Tempo creazione prompt: {meta.get('prompt_creation_time_s', '?')}s\n"
        f"- Dimensione cache disco: {cache_kb:.0f} KB\n"
        f"- ref_code shape: {meta.get('ref_code_shape', '?')}\n"
        f"- Creato: {meta.get('created_at', '?')}"
    )


# =============================================================================
# REQUISITI MACCHINA
# =============================================================================

def check_requirements(model_choice):
    try:
        import psutil
    except ImportError:
        return "❌ Errore: la libreria 'psutil' non è installata. Puoi installarla aprendo un terminale e digitando 'pip install psutil'."
        
    ram_info = psutil.virtual_memory()
    total_ram_gb = ram_info.total / (1024 ** 3)
    available_ram_gb = ram_info.available / (1024 ** 3)
    cpu_count = psutil.cpu_count(logical=True)
    
    if "1.7B" in model_choice:
        min_ram = 8.0
        rec_ram = 16.0
        model_name = "1.7B"
    else:
        min_ram = 4.0
        rec_ram = 8.0
        model_name = "0.6B"

    if total_ram_gb < min_ram:
        status = "❌ INADEGUATA"
        msg = f"La tua RAM totale ({total_ram_gb:.1f} GB) è inferiore al minimo richiesto ({min_ram} GB) per il modello {model_name}."
    elif available_ram_gb < min_ram:
        status = "⚠️ CRITICA"
        msg = f"La tua RAM totale è sufficiente ({total_ram_gb:.1f} GB), ma quella libera ({available_ram_gb:.1f} GB) è inferiore al minimo ({min_ram} GB). Chiudi altri programmi e libera RAM."
    elif total_ram_gb < rec_ram:
        status = "⚠️ AL LIMITE"
        msg = f"La tua RAM ({total_ram_gb:.1f} GB) soddisfa i requisiti minimi ma è sotto i raccomandati ({rec_ram} GB) per {model_name}."
    else:
        status = "✅ OTTIMA"
        msg = f"La tua RAM ({total_ram_gb:.1f} GB) soddisfa i requisiti raccomandati per {model_name}."

    report = (
        f"📊 Verifica Requisiti per {model_name}:\n"
        f"• CPU (Thread/Core logici): {cpu_count}\n"
        f"• RAM Totale: {total_ram_gb:.1f} GB\n"
        f"• RAM Libera (attuale): {available_ram_gb:.1f} GB\n\n"
        f"Stato Macchina: {status}\n"
        f"Diagnosi: {msg}"
    )
    return report

# =============================================================================
# INTERFACCIA GRADIO
# =============================================================================

with gr.Blocks(
    title="Qwen3-TTS Voice Clone",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 🎙️ Qwen3-TTS Voice Clone Studio
        **Clona, gestisci e perfeziona voci con Qwen3-TTS (1.7B o 0.6B).**
        ⚠️ Modalità CPU — la generazione richiede diversi minuti per frase.
        """
    )

    # --- Carica Modello ---
    with gr.Row():
        with gr.Column(scale=2):
            model_selector = gr.Radio(
                choices=list(MODEL_CHOICES.keys()),
                value=list(MODEL_CHOICES.keys())[0],
                label="🧠 Seleziona Modello",
            )
            check_req_btn = gr.Button("🔍 Verifica Requisiti Macchina")
        with gr.Column(scale=1):
            load_btn = gr.Button("📥 Carica Modello", variant="primary")
            model_status = gr.Textbox(
                label="Stato modello / Requisiti",
                value="⏳ Modello non caricato. Seleziona un modello e premi il pulsante.",
                interactive=False,
                lines=5,
            )
            load_btn.click(fn=load_model, inputs=[model_selector], outputs=model_status)
            check_req_btn.click(fn=check_requirements, inputs=[model_selector], outputs=model_status)

    gr.Markdown("---")

    with gr.Tabs():

        # =================================================================
        # TAB 1: Generazione Rapida
        # =================================================================
        with gr.Tab("⚡ Generazione Rapida"):
            gr.Markdown("Carica un audio di riferimento e genera direttamente senza salvare il profilo.")

            with gr.Row():
                with gr.Column():
                    direct_audio = gr.Audio(label="🎤 Audio di riferimento (3+ secondi)", type="numpy")
                    direct_ref_text = gr.Textbox(
                        label="📝 Trascrizione audio (opzionale ma consigliata)",
                        placeholder="Cosa dice l'audio di riferimento...", lines=2,
                    )
                    direct_text = gr.Textbox(
                        label="💬 Testo da sintetizzare",
                        placeholder="Testo che la voce clonata dovrà dire...", lines=3,
                    )
                    with gr.Row():
                        direct_lang = gr.Dropdown(choices=LANGUAGES, value="Italian", label="🌐 Lingua")
                        direct_streaming = gr.Checkbox(label="🔄 Streaming Mode", value=False)
                    direct_gen_btn = gr.Button("🔊 Genera Audio", variant="primary")

                with gr.Column():
                    direct_status = gr.Textbox(label="Stato", interactive=False)
                    direct_output = gr.Audio(label="🎧 Audio Generato", type="numpy")

            direct_gen_btn.click(
                fn=generate_speech_direct,
                inputs=[direct_audio, direct_ref_text, direct_text, direct_lang, direct_streaming],
                outputs=[direct_status, direct_output],
            )

        # =================================================================
        # TAB 2: Gestione Profili Voce
        # =================================================================
        with gr.Tab("👤 Gestione Voci"):
            gr.Markdown("Salva profili voce riutilizzabili. Il prompt viene pre-calcolato e salvato su disco.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ➕ Aggiungi nuova voce")
                    voice_name_input = gr.Textbox(label="Nome voce", placeholder="es: Mario, Francesca...")
                    voice_audio_input = gr.Audio(label="🎤 Audio di riferimento (3+ secondi)", type="numpy")
                    voice_ref_text_input = gr.Textbox(
                        label="📝 Trascrizione esatta dell'audio",
                        placeholder="Scrivi qui la trascrizione...", lines=3,
                    )
                    add_voice_btn = gr.Button("➕ Salva Voce + Prompt", variant="primary")
                    add_status = gr.Textbox(label="Stato", interactive=False)

                with gr.Column():
                    gr.Markdown("### 📋 Voci salvate")
                    voice_list = gr.Dropdown(
                        choices=get_voice_names(), label="Seleziona voce", interactive=True,
                    )
                    voice_details = gr.Markdown("")
                    delete_btn = gr.Button("🗑️ Elimina Voce", variant="stop")
                    delete_status = gr.Textbox(label="Stato", interactive=False)

            add_voice_btn.click(
                fn=add_voice,
                inputs=[voice_name_input, voice_audio_input, voice_ref_text_input],
                outputs=[add_status, voice_list],
            )
            delete_btn.click(
                fn=delete_voice, inputs=[voice_list], outputs=[delete_status, voice_list],
            )
            voice_list.change(fn=get_voice_details, inputs=[voice_list], outputs=[voice_details])

        # =================================================================
        # TAB 3: Genera con Voce Salvata
        # =================================================================
        with gr.Tab("🔊 Genera con Voce Salvata"):
            gr.Markdown("Usa un profilo voce salvato. Il prompt è già su disco — nessun ricalcolo.")

            with gr.Row():
                with gr.Column():
                    gen_voice = gr.Dropdown(choices=get_voice_names(), label="👤 Seleziona voce", interactive=True)
                    refresh_btn = gr.Button("🔄 Aggiorna lista")
                    gen_text = gr.Textbox(label="💬 Testo da sintetizzare", placeholder="Scrivi qui...", lines=4)
                    with gr.Row():
                        gen_lang = gr.Dropdown(choices=LANGUAGES, value="Italian", label="🌐 Lingua")
                        gen_streaming = gr.Checkbox(label="🔄 Streaming Mode", value=False)
                    gen_btn = gr.Button("🔊 Genera Audio", variant="primary")

                with gr.Column():
                    gen_status = gr.Textbox(label="Stato", interactive=False)
                    gen_output = gr.Audio(label="🎧 Audio Generato", type="numpy")

            refresh_btn.click(fn=lambda: gr.update(choices=get_voice_names()), outputs=gen_voice)
            gen_btn.click(
                fn=generate_speech,
                inputs=[gen_voice, gen_text, gen_lang, gen_streaming],
                outputs=[gen_status, gen_output],
            )

        # =================================================================
        # TAB 4: Analytics
        # =================================================================
        with gr.Tab("📊 Analytics"):
            gr.Markdown("Analisi tempi di generazione, confronto modalità e statistiche.")

            analytics_btn = gr.Button("📊 Aggiorna Grafici", variant="primary")
            clear_btn = gr.Button("🗑️ Cancella Statistiche", variant="stop")
            analytics_summary = gr.Markdown("")

            with gr.Row():
                plot_voices = gr.Plot(label="Tempo per Voce")
                plot_streaming = gr.Plot(label="Streaming vs Non-Streaming")
            with gr.Row():
                plot_rtf = gr.Plot(label="RTF vs Lunghezza Testo (per modello)")
                plot_models = gr.Plot(label="Confronto 1.7B vs 0.6B")

            analytics_btn.click(
                fn=build_analytics,
                outputs=[analytics_summary, plot_voices, plot_streaming, plot_rtf, plot_models],
            )
            clear_btn.click(
                fn=clear_stats,
                outputs=[analytics_summary, plot_voices, plot_streaming, plot_rtf, plot_models, analytics_summary],
            )

        # =================================================================
        # TAB 5: Guida Perfezionamento Voci
        # =================================================================
        with gr.Tab("📖 Perfezionamento Voci"):
            gr.Markdown(
                """
                # 🎯 Guida al Perfezionamento delle Voci Clonate

                ## 1. Audio di Riferimento — Requisiti

                ### Durata ideale
                - **Minimo**: 3 secondi (sotto questo limite il modello non ha abbastanza informazioni)
                - **Ottimale**: 5–15 secondi (il modello cattura meglio timbro, ritmo e prosodia)
                - **Massimo utile**: ~30 secondi (oltre non migliora significativamente e aumenta il tempo di processing)

                ### Qualità audio
                | Caratteristica | Ideale | Da evitare |
                |---|---|---|
                | **Rumore di fondo** | Silenzio totale o quasi | Musica, vento, traffico, eco |
                | **Sample rate** | 24kHz o superiore (il modello ricampiona a 24kHz) | Sotto 16kHz |
                | **Formato** | WAV non compresso, mono | MP3 molto compresso, stereo |
                | **Volume** | Costante, ben normalizzato (-3dB a -1dB peak) | Troppo basso, clipping, variazioni estreme |
                | **Ambiente** | Stanza trattata acusticamente | Bagno, corridoio (riverbero) |

                ### Contenuto vocale
                - L'audio deve contenere **solo la voce target** (niente altre persone che parlano)
                - Il parlante deve parlare in modo **naturale e fluido**
                - Evitare **sussurri, grida o toni molto estremi** (a meno che non sia l'effetto desiderato)
                - Includere una **varietà di suoni/fonemi** per dare al modello più informazioni sulla voce

                ---

                ## 2. Trascrizione — Come Scriverla
                La trascrizione è **fondamentale** per la qualità. Il modello la usa per allineare
                testo e audio (modalità ICL), permettendo una clonazione molto più fedele.

                ### Regole
                1. **Trascrivi esattamente** quello che viene detto, parola per parola
                2. **Includi esitazioni**: "eee...", "mmm...", "cioè..."
                3. **Rispetta la punteggiatura**: punti, virgole, punti interrogativi influenzano la prosodia
                4. **Stessa lingua dell'audio**: se l'audio è in italiano, la trascrizione deve essere in italiano
                5. **Niente abbreviazioni**: scrivi "dottore" non "dott.", "per esempio" non "es."

                ### Esempi
                | ❌ Sbagliato | ✅ Corretto |
                |---|---|
                | "Ciao come stai bene grazie" | "Ciao, come stai? Bene, grazie." |
                | "tipo andavo al" | "Tipo... andavo al, eee... al supermercato." |
                | "dott rossi tel 333" | "Dottor Rossi, telefono tre tre tre..." |

                ---

                ## 3. Consigli per il Testo da Generare

                ### Per risultati migliori
                - Usa **frasi complete** con punteggiatura corretta
                - Il modello gestisce bene **10-50 parole** per generazione
                - Per testi lunghi, **dividi in frasi** e genera separatamente
                - Specifica la **lingua corretta** nel dropdown (non affidarti ad "Auto" per l'italiano)

                ### Problemi comuni e soluzioni
                | Problema | Causa | Soluzione |
                |---|---|---|
                | Voce non somiglia | Audio ref troppo corto o rumoroso | Usa audio più lungo e pulito |
                | Pronuncia strana | Trascrizione errata | Correggi la trascrizione ref |
                | Audio tagliato | Testo troppo lungo per max_new_tokens | Dividi in frasi più corte |
                | Suona robotico | x_vector_only_mode attivo | Fornisci sempre la trascrizione |

                ---

                ## 4. Workflow Consigliato per Voce Perfetta

                1. **Registra** un audio pulito di 10-15 secondi in un ambiente silenzioso
                2. **Trascrivi** esattamente, includendo pause e esitazioni
                3. **Salva il profilo** nella tab "Gestione Voci"
                4. **Testa** con una frase breve (5-10 parole) nella tab "Genera con Voce Salvata"
                5. **Confronta** l'audio originale con quello generato
                6. Se non soddisfatto, **prova un altro campione audio** dello stesso parlante
                7. Usa la tab **Analytics** per confrontare i tempi e la qualità tra diverse voci

                ---

                ## 5. Streaming vs Non-Streaming

                | Modalità | Quando usarla |
                |---|---|
                | **Non-Streaming** (default) | Per generazione finale di qualità. Il modello vede tutto il testo prima di generare. |
                | **Streaming** | Per test rapidi. Simula input incrementale del testo. Su CPU la differenza è minima. |

                > **Nota CPU**: Su CPU entrambe le modalità hanno tempi simili.
                > La vera differenza si nota su GPU dove lo streaming riduce la latenza
                > del primo pacchetto audio a ~97ms.

                ---

                ## 6. Ottimizzazione Audio con Strumenti Esterni

                Se il tuo audio di riferimento non è ottimale, puoi migliorarlo con:

                - **Audacity** (gratuito): rimuovi rumore di fondo, normalizza volume, taglia silenzi
                - **ffmpeg**: converti in WAV mono 24kHz:
                  ```
                  ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav
                  ```
                - **Noise reduction online**: usa strumenti come Adobe Podcast Enhance

                ### Checklist pre-caricamento
                - [ ] Audio mono (non stereo)
                - [ ] Sample rate ≥ 16kHz (ideale 24kHz+)
                - [ ] Nessun rumore di fondo udibile
                - [ ] Volume normalizzato
                - [ ] Durata 5-15 secondi
                - [ ] Solo la voce target, niente altri speaker
                - [ ] Trascrizione pronta e verificata
                """
            )

    gr.Markdown(
        """
        ---
        **Note:** Il modello viene scaricato al primo avvio (~4.3 GB). I profili voce sono salvati
        su disco nella cartella `voices/` con il prompt pre-calcolato (~100-200 KB per voce).
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
