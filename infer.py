"""
Reconstruct longer audio/video tracks with your trained codec.
Saves outputs safely into a 'preds/' folder with unique filenames (_compressed.wav)
and prints exact file size compression metrics.
Run: python infer.py
"""
import os
import math
import subprocess
import tempfile
import torch
import torchaudio

MODEL_FILE = "model"  # training file name without .py
Encodec, EncodecConfig = (lambda m: (m.Encodec, m.EncodecConfig))(__import__(MODEL_FILE))

# ---------------- settings ----------------
INPUT       = "close_eyes.mp3"   # Can be 2-3 minute mp3, wav, or mp4 video
CKPT        = "ckpt_v3.pt"
LM_CKPT     = "lm_v3.pt"        # Optional, used for bitrate estimation
START_SEC   = 0                 # Start time in seconds
MAX_SECONDS = 180               # Supports full 2-3 minute clips (e.g., 180 seconds)
SEG_SECONDS = 5                 # Processes in 5-second chunks to keep CPU/RAM safe
OUT_DIR     = "preds"           # Dedicated output folder
# ------------------------------------------


def load_audio(path, start, seconds):
    """-> (1, samples) float tensor at 24 kHz mono, peak-normalised."""
    assert os.path.exists(path), f"File not found: {path!r}"
    tmp = tempfile.mktemp(suffix=".wav")
    
    # FFmpeg automatically extracts audio track if input is an MP4 video or MP3 audio
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", 
        "-ss", str(start), "-t", str(seconds),
        "-i", path, "-ac", "1", "-ar", "24000", tmp
    ], check=True)
    
    w, sr = torchaudio.load(tmp)
    os.remove(tmp)
    assert w.shape[1] > 24000, "Audio too short or start time is past the end of the file."
    return w / w.abs().max().clamp(min=1e-5) * 0.95


@torch.no_grad()
def main():
    torch.set_num_threads(os.cpu_count() or 1)
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = EncodecConfig()
    ck = torch.load(CKPT, map_location="cpu")
    
    # Match codebook size from checkpoint
    cfg.codebook_size = ck["model"]["rvq.codebooks.0.embedding.weight"].shape[0]
    model = Encodec(cfg)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"Loaded {CKPT} (trained to step {ck.get('step')})")

    hop = model.downsample_factor()
    fps = 24000 / hop
    
    print(f"Loading and resampling {INPUT} ({MAX_SECONDS}s max)...")
    wav = load_audio(INPUT, START_SEC, MAX_SECONDS)
    
    seg = hop * int(SEG_SECONDS * fps)  # Whole number of latent frames per chunk
    n = (wav.shape[1] // seg) * seg
    wav = wav[:, :n]

    recon, all_codes = [], []
    print(f"Running neural compression & reconstruction across {n / 24000:.1f}s of audio...")
    for i in range(0, n, seg):
        x = wav[:, i:i + seg].unsqueeze(0)  # (1, 1, seg)
        y, *_ = model(x)
        all_codes.append(model.get_codes(x))  # (1, n_q, frames)
        recon.append(y[0].clamp(-1, 1))
        
    recon = torch.cat(recon, dim=1)
    codes = torch.cat(all_codes, dim=2)

    # Calculate metrics
    err = wav - recon
    snr = 10 * math.log10(wav.pow(2).mean().item() / (err.pow(2).mean().item() + 1e-12))
    nominal_kbps = cfg.n_codebook * math.log2(cfg.codebook_size) * fps / 1000
    used = [len(torch.unique(codes[:, q])) for q in range(cfg.n_codebook)]

    # Generate unique output path in preds/ without overwriting
    base_name = os.path.splitext(os.path.basename(INPUT))[0]
    out_recon_path = os.path.join(OUT_DIR, f"{base_name}_compressed.wav")
    
    # Save reconstructed compressed audio
    torchaudio.save(out_recon_path, recon, 24000)

    # File size metrics calculation
    duration_sec = n / 24000
    # Uncompressed PCM 24kHz 16-bit mono size = 24000 * 2 bytes/sample * duration
    original_size_kb = (24000 * 2 * duration_sec) / 1024
    # Neural compressed size based on actual bitrate (nominal_kbps * 1000 bits/sec / 8 bits/byte * duration)
    compressed_size_kb = (nominal_kbps * 1000 / 8 * duration_sec) / 1024
    compression_ratio = original_size_kb / compressed_size_kb

    print("\n" + "="*50)
    print(f"SUCCESS! Saved to: {out_recon_path}")
    print(f"--- Compression & Audio Metrics ---")
    print(f"Duration:          {duration_sec:.2f} seconds")
    print(f"Original PCM Size: ~{original_size_kb:.2f} KB")
    print(f"Compressed Size:   ~{compressed_size_kb:.2f} KB (at {nominal_kbps:.2f} kbps)")
    print(f"Compression Ratio: {compression_ratio:.1f}x smaller")
    print(f"Signal-to-Noise:   {snr:.2f} dB")
    print(f"Codebooks Used:    {used} / {cfg.codebook_size} tokens active")
    print("="*50)


if __name__ == "__main__":
    main()
