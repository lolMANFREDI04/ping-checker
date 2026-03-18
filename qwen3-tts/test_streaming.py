"""
Qwen3-TTS - Test Modalità Streaming vs Non-Streaming
=====================================================
Confronta le due modalità di generazione:
  - non_streaming_mode=True  (tutto il testo processato prima, poi genera)
  - non_streaming_mode=False (simula streaming: il testo viene passato in modo incrementale)

Il parametro non_streaming_mode nella libreria qwen-tts NON è vero streaming output
(non restituisce chunk audio in tempo reale), ma cambia il modo interno in cui il
testo viene alimentato al modello:
  - True:  il modello vede TUTTO il testo prima di iniziare a generare i codec audio
  - False: simula l'input incrementale del testo (Dual-Track), il modello inizia a
           generare audio anche prima di aver visto tutto il testo

Su CPU la differenza di latenza è minima perché il collo di bottiglia è la decodifica.
Su GPU la modalità streaming (False) permette di ottenere il primo audio chunk molto prima.

Uso:
    python test_streaming.py audio_riferimento.wav "trascrizione dell'audio"
"""

import os
import sys
import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# --- Configurazione ---
TEXT_IT = "Buongiorno a tutti, questo è un test di generazione vocale in modalità streaming e non streaming."

def main():
    if len(sys.argv) < 3:
        print("Uso: python test_streaming.py <audio_riferimento.wav> \"trascrizione audio\"")
        print()
        print("L'audio di riferimento serve per clonare la voce.")
        sys.exit(1)

    ref_audio_path = sys.argv[1]
    ref_text = sys.argv[2]

    if not os.path.exists(ref_audio_path):
        print(f"Errore: file '{ref_audio_path}' non trovato.")
        sys.exit(1)

    print("=" * 60)
    print("  Qwen3-TTS - Test Streaming vs Non-Streaming")
    print("=" * 60)
    print()

    # Carica modello
    print("[1/5] Caricamento modello (CPU, float32)...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    print(f"       Modello caricato in {time.time() - t0:.1f}s")

    # Pre-calcola il prompt voce (una sola volta)
    print("[2/5] Creazione prompt voce (una sola volta)...")
    t0 = time.time()
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    prompt_time = time.time() - t0
    print(f"       Prompt creato in {prompt_time:.1f}s")
    print()

    # --- Test 1: Non-Streaming (default nei nostri script) ---
    print("[3/5] Generazione NON-STREAMING (non_streaming_mode=True)...")
    print(f"       Testo: \"{TEXT_IT}\"")
    t0 = time.time()
    wavs_ns, sr = model.generate_voice_clone(
        text=TEXT_IT,
        language="Italian",
        voice_clone_prompt=voice_prompt,
        non_streaming_mode=True,
    )
    time_non_streaming = time.time() - t0
    sf.write("output_non_streaming.wav", wavs_ns[0], sr)
    print(f"       Tempo: {time_non_streaming:.1f}s")
    print(f"       Durata audio: {len(wavs_ns[0]) / sr:.1f}s")
    print(f"       Salvato: output_non_streaming.wav")
    print()

    # --- Test 2: Streaming simulato ---
    print("[4/5] Generazione STREAMING simulato (non_streaming_mode=False)...")
    print(f"       Testo: \"{TEXT_IT}\"")
    t0 = time.time()
    wavs_s, sr = model.generate_voice_clone(
        text=TEXT_IT,
        language="Italian",
        voice_clone_prompt=voice_prompt,
        non_streaming_mode=False,
    )
    time_streaming = time.time() - t0
    sf.write("output_streaming.wav", wavs_s[0], sr)
    print(f"       Tempo: {time_streaming:.1f}s")
    print(f"       Durata audio: {len(wavs_s[0]) / sr:.1f}s")
    print(f"       Salvato: output_streaming.wav")
    print()

    # --- Riepilogo ---
    print("[5/5] Riepilogo")
    print("=" * 60)
    print(f"  Creazione prompt voce:     {prompt_time:.1f}s (una tantum)")
    print(f"  Non-Streaming (True):      {time_non_streaming:.1f}s")
    print(f"  Streaming simulato (False): {time_streaming:.1f}s")
    diff = abs(time_non_streaming - time_streaming)
    faster = "Non-Streaming" if time_non_streaming < time_streaming else "Streaming"
    print(f"  Differenza:                {diff:.1f}s (più veloce: {faster})")
    print()
    print("  NOTA: Su CPU le prestazioni sono simili. La vera differenza")
    print("  si nota su GPU dove lo streaming permette di ottenere il primo")
    print("  pacchetto audio molto prima (latenza ~97ms da specifica).")
    print("=" * 60)


if __name__ == "__main__":
    main()
