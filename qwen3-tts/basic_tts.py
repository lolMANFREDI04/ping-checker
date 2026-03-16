"""
Qwen3-TTS Voice Clone - Script Base
====================================
Segue l'esempio ufficiale di HuggingFace per il modello Qwen3-TTS-12Hz-1.7B-Base.
Configurato per girare in modalità CPU (senza CUDA/ROCm).

Il modello verrà scaricato automaticamente al primo avvio (~3.4 GB).
La generazione su CPU richiederà diversi minuti.

Uso:
    python basic_tts.py
    python basic_tts.py mio_audio.wav "trascrizione del mio audio"
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

print("=" * 50)
print("  Qwen3-TTS Voice Clone - Modalità CPU")
print("=" * 50)
print()

# Caricamento del modello in modalità CPU
# - device_map="cpu" perché la GPU non supporta CUDA/ROCm
# - dtype=torch.float32 per compatibilità CPU (bfloat16 richiede hardware specifico)
# - Nessun flash_attention_2 (richiede GPU)
print("[1/4] Caricamento modello (scaricamento automatico al primo avvio)...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cpu",
    dtype=torch.float32,
)
print("       Modello caricato con successo!")

# Audio di riferimento: da argomento o generato localmente come placeholder
ref_audio_path = "ref_sample.wav"

if len(sys.argv) >= 3:
    # Usa audio e trascrizione forniti dall'utente
    ref_audio_path = sys.argv[1]
    ref_text = sys.argv[2]
    print(f"       Uso audio di riferimento: {ref_audio_path}")
elif os.path.exists(ref_audio_path):
    ref_text = input("Inserisci la trascrizione dell'audio ref_sample.wav: ").strip()
    if not ref_text:
        print("Nessuna trascrizione. Uso modalita' x_vector_only (qualita' ridotta).")
        ref_text = None
else:
    # Nessun audio fornito: genera un tono di esempio per test
    print("       Nessun audio di riferimento trovato.")
    print("       Genero un tono di test (per risultati reali, fornisci un audio vero).")
    print()
    print("  Uso: python basic_tts.py <file_audio.wav> \"trascrizione dell'audio\"")
    print()
    sr_ref = 24000
    duration = 3.0
    t = np.linspace(0, duration, int(sr_ref * duration), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    sf.write(ref_audio_path, tone, sr_ref)
    ref_text = None  # x_vector_only_mode

# Testo da sintetizzare con la voce clonata (in italiano)
text = "Oltre noi. Lo squadrone, a quanto ho capito, c'è eee... Ci saro pure, tipo Martina, Matilde. C'è pure Giulia Loiacono. Cioè pure Alberto, forse Calandrino eemm... Pure Alessandra. Eee... siamo un po' comunque, siamo un po."

print("[2/4] Preparazione prompt voce...")
clone_kwargs = {
    "ref_audio": ref_audio_path,
}
if ref_text:
    clone_kwargs["ref_text"] = ref_text
    clone_kwargs["x_vector_only_mode"] = False
else:
    clone_kwargs["x_vector_only_mode"] = True

print("[3/4] Generazione audio con voce clonata (potrebbe richiedere diversi minuti su CPU)...")
wavs, sr = model.generate_voice_clone(
    text=text,
    language="Italian",
    **clone_kwargs,
)

output_file = "output_voice_clone.wav"
sf.write(output_file, wavs[0], sr)
print(f"[4/4] Audio salvato in: {output_file}")
print()
print("Fatto! Puoi ascoltare il file generato.")
