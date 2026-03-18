"""
Qwen3-TTS - Salvataggio e Ricaricamento Prompt Voce
=====================================================
Dimostra come salvare su disco il VoiceClonePromptItem pre-calcolato
e ricaricarlo senza dover ri-processare l'audio di riferimento ogni volta.

Il costo pesante è in create_voice_clone_prompt() che:
  1. Codifica l'audio con il speech tokenizer (ref_code)
  2. Estrae lo speaker embedding (ref_spk_embedding)
  3. Fa resampling a 24kHz se necessario

Salvando questi dati su disco, le generazioni successive partono direttamente
dalla fase di generazione, risparmiando quel tempo.

Uso:
    # Salva un profilo voce
    python test_voice_cache.py save mia_voce audio.wav "trascrizione audio"

    # Lista profili salvati
    python test_voice_cache.py list

    # Genera con un profilo salvato
    python test_voice_cache.py generate mia_voce "Testo da sintetizzare in italiano"

    # Info su un profilo
    python test_voice_cache.py info mia_voce
"""

import os
import sys
import time
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

CACHE_DIR = Path("voice_cache")
CACHE_DIR.mkdir(exist_ok=True)

model = None


def get_model():
    """Carica il modello una sola volta (lazy loading)."""
    global model
    if model is None:
        print("Caricamento modello (CPU, float32)...")
        t0 = time.time()
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cpu",
            dtype=torch.float32,
        )
        print(f"Modello caricato in {time.time() - t0:.1f}s")
    return model


def save_prompt(name: str, ref_audio: str, ref_text: str):
    """Salva un profilo voce pre-calcolato su disco."""
    if not os.path.exists(ref_audio):
        print(f"Errore: file '{ref_audio}' non trovato.")
        return

    m = get_model()

    print(f"Creazione prompt voce per '{name}'...")
    t0 = time.time()
    items = m.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    elapsed = time.time() - t0
    print(f"Prompt creato in {elapsed:.1f}s")

    # Salva su disco
    voice_dir = CACHE_DIR / name
    voice_dir.mkdir(exist_ok=True)

    item = items[0]

    # Salva i tensori
    torch.save(item.ref_code, voice_dir / "ref_code.pt")
    torch.save(item.ref_spk_embedding, voice_dir / "ref_spk_embedding.pt")

    # Salva i metadati
    meta = {
        "name": name,
        "ref_audio_original": os.path.abspath(ref_audio),
        "ref_text": ref_text,
        "x_vector_only_mode": item.x_vector_only_mode,
        "icl_mode": item.icl_mode,
        "ref_code_shape": list(item.ref_code.shape) if item.ref_code is not None else None,
        "ref_spk_embedding_shape": list(item.ref_spk_embedding.shape),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_creation_time_s": round(elapsed, 1),
    }
    with open(voice_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Calcola dimensione salvata
    total_bytes = sum(f.stat().st_size for f in voice_dir.iterdir())
    print(f"Profilo '{name}' salvato in {voice_dir}/")
    print(f"Dimensione su disco: {total_bytes / 1024:.1f} KB")
    print(f"Tempo risparmiato per ogni generazione futura: ~{elapsed:.1f}s")


def load_prompt(name: str) -> VoiceClonePromptItem:
    """Carica un profilo voce pre-calcolato dal disco."""
    voice_dir = CACHE_DIR / name
    if not voice_dir.exists():
        print(f"Errore: profilo '{name}' non trovato in {CACHE_DIR}/")
        sys.exit(1)

    with open(voice_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    ref_code = torch.load(voice_dir / "ref_code.pt", weights_only=True)
    ref_spk_embedding = torch.load(voice_dir / "ref_spk_embedding.pt", weights_only=True)

    return VoiceClonePromptItem(
        ref_code=ref_code,
        ref_spk_embedding=ref_spk_embedding,
        x_vector_only_mode=meta["x_vector_only_mode"],
        icl_mode=meta["icl_mode"],
        ref_text=meta["ref_text"],
    )


def list_voices():
    """Lista tutti i profili voce salvati."""
    voices = [d for d in CACHE_DIR.iterdir() if d.is_dir() and (d / "metadata.json").exists()]

    if not voices:
        print("Nessun profilo voce salvato.")
        print(f"Usa: python {sys.argv[0]} save <nome> <audio.wav> \"trascrizione\"")
        return

    print(f"Profili voce salvati ({len(voices)}):")
    print("-" * 60)
    for v in sorted(voices):
        with open(v / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        total_kb = sum(f.stat().st_size for f in v.iterdir()) / 1024
        print(f"  {meta['name']:<20} {total_kb:>8.1f} KB   creato: {meta.get('created_at', '?')}")
    print("-" * 60)


def show_info(name: str):
    """Mostra informazioni dettagliate su un profilo voce."""
    voice_dir = CACHE_DIR / name
    if not voice_dir.exists():
        print(f"Errore: profilo '{name}' non trovato.")
        return

    with open(voice_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    total_kb = sum(f.stat().st_size for f in voice_dir.iterdir()) / 1024

    print(f"Profilo: {meta['name']}")
    print(f"  Trascrizione ref:     {meta['ref_text']}")
    print(f"  Audio originale:      {meta['ref_audio_original']}")
    print(f"  ICL mode:             {meta['icl_mode']}")
    print(f"  x_vector_only:        {meta['x_vector_only_mode']}")
    print(f"  ref_code shape:       {meta.get('ref_code_shape', '?')}")
    print(f"  spk_embedding shape:  {meta['ref_spk_embedding_shape']}")
    print(f"  Dimensione disco:     {total_kb:.1f} KB")
    print(f"  Creato:               {meta.get('created_at', '?')}")
    print(f"  Tempo prompt orig.:   {meta.get('prompt_creation_time_s', '?')}s")


def generate(name: str, text: str, language: str = "Italian"):
    """Genera audio usando un profilo voce salvato."""
    print(f"Caricamento profilo '{name}' da disco...")
    t0 = time.time()
    prompt_item = load_prompt(name)
    load_time = time.time() - t0
    print(f"Profilo caricato in {load_time:.2f}s (vs ~minuti per ri-calcolarlo)")

    m = get_model()

    print(f"Generazione audio...")
    print(f"  Testo: \"{text}\"")
    t0 = time.time()
    wavs, sr = m.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=[prompt_item],
    )
    gen_time = time.time() - t0

    output_file = f"output_{name}_{int(time.time())}.wav"
    sf.write(output_file, wavs[0], sr)

    print(f"Audio generato in {gen_time:.1f}s")
    print(f"Durata audio: {len(wavs[0]) / sr:.1f}s")
    print(f"Salvato: {output_file}")
    print()
    print(f"Tempo caricamento prompt da disco: {load_time:.2f}s")
    print(f"Tempo generazione:                 {gen_time:.1f}s")


def print_usage():
    print("Qwen3-TTS Voice Cache - Gestione profili voce pre-calcolati")
    print()
    print("Comandi:")
    print(f"  python {sys.argv[0]} save <nome> <audio.wav> \"trascrizione\"")
    print(f"  python {sys.argv[0]} list")
    print(f"  python {sys.argv[0]} info <nome>")
    print(f"  python {sys.argv[0]} generate <nome> \"testo da sintetizzare\"")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "save":
        if len(sys.argv) < 5:
            print("Uso: python test_voice_cache.py save <nome> <audio.wav> \"trascrizione\"")
            sys.exit(1)
        save_prompt(sys.argv[2], sys.argv[3], sys.argv[4])

    elif cmd == "list":
        list_voices()

    elif cmd == "info":
        if len(sys.argv) < 3:
            print("Uso: python test_voice_cache.py info <nome>")
            sys.exit(1)
        show_info(sys.argv[2])

    elif cmd == "generate":
        if len(sys.argv) < 4:
            print("Uso: python test_voice_cache.py generate <nome> \"testo\"")
            sys.exit(1)
        generate(sys.argv[2], sys.argv[3])

    else:
        print(f"Comando sconosciuto: {cmd}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
