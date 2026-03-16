"""
Qwen3-TTS Voice Clone - GUI Avanzata
======================================
Interfaccia grafica Gradio per gestire più voci clonate,
caricare audio di riferimento e generare speech con barra di progresso.

Configurato per CPU (senza CUDA/ROCm).

Uso:
    python gui_tts.py
    Poi apri il browser all'indirizzo mostrato nel terminale.
"""

import os
import json
import time
import torch
import soundfile as sf
import numpy as np
import gradio as gr
from pathlib import Path
from qwen_tts import Qwen3TTSModel

# --- Configurazione ---
VOICES_DIR = Path("voices")          # Cartella profili voci salvate
OUTPUT_DIR = Path("outputs")         # Cartella audio generati
VOICES_DB = VOICES_DIR / "voices.json"

VOICES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German",
             "French", "Russian", "Portuguese", "Spanish", "Italian"]

# --- Stato globale ---
model = None
voice_prompts = {}  # nome_voce -> voice_clone_prompt (cache in memoria)


def load_voices_db():
    """Carica il database dei profili voce dal disco."""
    if VOICES_DB.exists():
        with open(VOICES_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_voices_db(db):
    """Salva il database dei profili voce su disco."""
    with open(VOICES_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def get_voice_names():
    """Restituisce la lista dei nomi dei profili voce salvati."""
    db = load_voices_db()
    return list(db.keys())


def load_model(progress=gr.Progress(track_tqdm=True)):
    """Carica il modello Qwen3-TTS in modalità CPU."""
    global model
    if model is not None:
        return "✅ Modello già caricato in memoria."

    progress(0, desc="Scaricamento/caricamento modello...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    progress(1, desc="Modello pronto!")
    return "✅ Modello caricato con successo (CPU, float32)."


def add_voice(name, audio_file, ref_text, progress=gr.Progress()):
    """Aggiunge un nuovo profilo voce dal file audio di riferimento."""
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

    # Salva l'audio nella cartella della voce
    voice_dir = VOICES_DIR / name
    voice_dir.mkdir(exist_ok=True)

    # Copia il file audio
    audio_path = voice_dir / "reference.wav"
    if isinstance(audio_file, tuple):
        # Gradio restituisce (sample_rate, numpy_array)
        sr_in, audio_data = audio_file
        sf.write(str(audio_path), audio_data, sr_in)
    elif isinstance(audio_file, str):
        import shutil
        shutil.copy2(audio_file, str(audio_path))
    else:
        return "❌ Formato audio non riconosciuto.", gr.update()

    progress(0.3, desc="Creazione prompt voce clonata (può richiedere tempo su CPU)...")

    # Pre-calcola il prompt per riutilizzarlo dopo
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=str(audio_path),
        ref_text=ref_text.strip(),
        x_vector_only_mode=False,
    )
    voice_prompts[name] = prompt_items

    progress(0.9, desc="Salvataggio profilo...")

    # Salva nel database
    db = load_voices_db()
    db[name] = {
        "audio_path": str(audio_path),
        "ref_text": ref_text.strip(),
    }
    save_voices_db(db)

    progress(1.0, desc="Fatto!")
    new_choices = get_voice_names()
    return (
        f"✅ Voce '{name}' aggiunta con successo!",
        gr.update(choices=new_choices, value=name),
    )


def delete_voice(name):
    """Elimina un profilo voce."""
    if not name:
        return "❌ Seleziona una voce da eliminare.", gr.update()

    db = load_voices_db()
    if name in db:
        # Rimuovi file
        voice_dir = VOICES_DIR / name
        if voice_dir.exists():
            import shutil
            shutil.rmtree(voice_dir)
        del db[name]
        save_voices_db(db)

    if name in voice_prompts:
        del voice_prompts[name]

    new_choices = get_voice_names()
    return (
        f"✅ Voce '{name}' eliminata.",
        gr.update(choices=new_choices, value=new_choices[0] if new_choices else None),
    )


def generate_speech(voice_name, text, language, progress=gr.Progress()):
    """Genera audio con la voce clonata selezionata."""
    if model is None:
        return "❌ Carica prima il modello.", None

    if not voice_name:
        return "❌ Seleziona un profilo voce.", None

    if not text or not text.strip():
        return "❌ Inserisci il testo da sintetizzare.", None

    db = load_voices_db()
    if voice_name not in db:
        return f"❌ Voce '{voice_name}' non trovata nel database.", None

    voice_info = db[voice_name]
    lang = language if language != "Auto" else "Auto"

    progress(0.1, desc="Preparazione prompt voce...")

    # Usa il prompt in cache se disponibile, altrimenti ricrealo
    if voice_name not in voice_prompts:
        progress(0.2, desc="Ricalcolo prompt voce (non era in cache)...")
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=voice_info["audio_path"],
            ref_text=voice_info["ref_text"],
            x_vector_only_mode=False,
        )
        voice_prompts[voice_name] = prompt_items

    progress(0.4, desc="Generazione audio in corso (può richiedere diversi minuti su CPU)...")

    start_time = time.time()
    wavs, sr = model.generate_voice_clone(
        text=text.strip(),
        language=lang,
        voice_clone_prompt=voice_prompts[voice_name],
    )
    elapsed = time.time() - start_time

    progress(0.9, desc="Salvataggio audio...")

    # Salva l'output
    timestamp = int(time.time())
    output_path = OUTPUT_DIR / f"{voice_name}_{timestamp}.wav"
    sf.write(str(output_path), wavs[0], sr)

    progress(1.0, desc="Fatto!")

    status = f"✅ Audio generato in {elapsed:.1f}s → {output_path.name}"
    return status, (sr, wavs[0])


def generate_speech_direct(audio_file, ref_text, text, language, progress=gr.Progress()):
    """Genera audio direttamente da un file audio senza salvare il profilo."""
    if model is None:
        return "❌ Carica prima il modello.", None

    if audio_file is None:
        return "❌ Carica un file audio di riferimento.", None

    if not text or not text.strip():
        return "❌ Inserisci il testo da sintetizzare.", None

    lang = language if language != "Auto" else "Auto"

    progress(0.2, desc="Generazione audio in corso (può richiedere diversi minuti su CPU)...")

    start_time = time.time()

    # Prepara audio di riferimento
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
    }
    if ref_text and ref_text.strip():
        kwargs["ref_text"] = ref_text.strip()
    else:
        kwargs["x_vector_only_mode"] = True

    wavs, sr = model.generate_voice_clone(**kwargs)
    elapsed = time.time() - start_time

    progress(0.9, desc="Salvataggio...")

    timestamp = int(time.time())
    output_path = OUTPUT_DIR / f"direct_{timestamp}.wav"
    sf.write(str(output_path), wavs[0], sr)

    progress(1.0, desc="Fatto!")

    status = f"✅ Audio generato in {elapsed:.1f}s → {output_path.name}"
    return status, (sr, wavs[0])


# =============================================================================
# INTERFACCIA GRADIO
# =============================================================================

with gr.Blocks(
    title="Qwen3-TTS Voice Clone",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 🎙️ Qwen3-TTS Voice Clone
        **Clona voci da un campione audio e genera speech con il modello Qwen3-TTS-12Hz-1.7B-Base.**

        ⚠️ Modalità CPU — la generazione richiederà diversi minuti per ogni frase.
        """
    )

    # --- Sezione: Carica Modello ---
    with gr.Row():
        with gr.Column():
            load_btn = gr.Button("📥 Carica Modello", variant="primary", scale=1)
            model_status = gr.Textbox(
                label="Stato modello",
                value="⏳ Modello non caricato. Premi il pulsante per iniziare.",
                interactive=False,
            )
            load_btn.click(fn=load_model, outputs=model_status)

    gr.Markdown("---")

    with gr.Tabs():

        # =====================================================================
        # TAB 1: Generazione Rapida (senza salvare profilo)
        # =====================================================================
        with gr.Tab("⚡ Generazione Rapida"):
            gr.Markdown(
                "Carica un audio di riferimento e genera direttamente, senza salvare il profilo voce."
            )
            with gr.Row():
                with gr.Column():
                    direct_audio = gr.Audio(
                        label="🎤 Audio di riferimento (3+ secondi)",
                        type="numpy",
                    )
                    direct_ref_text = gr.Textbox(
                        label="📝 Trascrizione audio di riferimento (opzionale ma consigliata)",
                        placeholder="Scrivi qui cosa dice l'audio di riferimento...",
                        lines=2,
                    )
                    direct_text = gr.Textbox(
                        label="💬 Testo da sintetizzare",
                        placeholder="Scrivi qui il testo che la voce clonata dovrà dire...",
                        lines=3,
                    )
                    direct_lang = gr.Dropdown(
                        choices=LANGUAGES, value="Auto", label="🌐 Lingua"
                    )
                    direct_gen_btn = gr.Button("🔊 Genera Audio", variant="primary")

                with gr.Column():
                    direct_status = gr.Textbox(label="Stato", interactive=False)
                    direct_output = gr.Audio(label="🎧 Audio Generato", type="numpy")

            direct_gen_btn.click(
                fn=generate_speech_direct,
                inputs=[direct_audio, direct_ref_text, direct_text, direct_lang],
                outputs=[direct_status, direct_output],
            )

        # =====================================================================
        # TAB 2: Gestione Profili Voce
        # =====================================================================
        with gr.Tab("👤 Gestione Voci"):
            gr.Markdown("Salva profili voce riutilizzabili dai tuoi campioni audio.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Aggiungi nuova voce")
                    voice_name_input = gr.Textbox(
                        label="Nome voce",
                        placeholder="es: Mario, Francesca, Narratore...",
                    )
                    voice_audio_input = gr.Audio(
                        label="🎤 Audio di riferimento (3+ secondi)",
                        type="numpy",
                    )
                    voice_ref_text_input = gr.Textbox(
                        label="📝 Trascrizione dell'audio di riferimento",
                        placeholder="Scrivi qui la trascrizione esatta dell'audio...",
                        lines=3,
                    )
                    add_voice_btn = gr.Button("➕ Aggiungi Voce", variant="primary")
                    add_status = gr.Textbox(label="Stato", interactive=False)

                with gr.Column():
                    gr.Markdown("### Voci salvate")
                    voice_list = gr.Dropdown(
                        choices=get_voice_names(),
                        label="Seleziona voce",
                        interactive=True,
                    )
                    delete_btn = gr.Button("🗑️ Elimina Voce Selezionata", variant="stop")
                    delete_status = gr.Textbox(label="Stato", interactive=False)

            add_voice_btn.click(
                fn=add_voice,
                inputs=[voice_name_input, voice_audio_input, voice_ref_text_input],
                outputs=[add_status, voice_list],
            )
            delete_btn.click(
                fn=delete_voice,
                inputs=[voice_list],
                outputs=[delete_status, voice_list],
            )

        # =====================================================================
        # TAB 3: Genera con Voce Salvata
        # =====================================================================
        with gr.Tab("🔊 Genera con Voce Salvata"):
            gr.Markdown("Usa un profilo voce salvato per generare audio.")

            with gr.Row():
                with gr.Column():
                    gen_voice = gr.Dropdown(
                        choices=get_voice_names(),
                        label="👤 Seleziona voce",
                        interactive=True,
                    )
                    refresh_btn = gr.Button("🔄 Aggiorna lista voci")
                    gen_text = gr.Textbox(
                        label="💬 Testo da sintetizzare",
                        placeholder="Scrivi qui il testo...",
                        lines=4,
                    )
                    gen_lang = gr.Dropdown(
                        choices=LANGUAGES, value="Auto", label="🌐 Lingua"
                    )
                    gen_btn = gr.Button("🔊 Genera Audio", variant="primary")

                with gr.Column():
                    gen_status = gr.Textbox(label="Stato", interactive=False)
                    gen_output = gr.Audio(label="🎧 Audio Generato", type="numpy")

            refresh_btn.click(
                fn=lambda: gr.update(choices=get_voice_names()),
                outputs=gen_voice,
            )
            gen_btn.click(
                fn=generate_speech,
                inputs=[gen_voice, gen_text, gen_lang],
                outputs=[gen_status, gen_output],
            )

    gr.Markdown(
        """
        ---
        **Note:**
        - Il modello viene scaricato automaticamente al primo avvio (~3.4 GB).
        - La generazione su CPU è lenta — attendi diversi minuti per frase.
        - Per risultati migliori, usa audio di riferimento puliti di almeno 3 secondi.
        - La trascrizione dell'audio di riferimento migliora la qualità della clonazione.
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
