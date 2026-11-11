"""
Small, fast Encodec for a 4 GB RAM CPU laptop. No CLI: edit the settings below, then run

    python encodec_all.py

It does everything in order: prep (if needed) -> tests (optional) -> train codec -> train LM.
Needs in the same folder: rvq.py, encoder.py, decoder.py, transformer.py, descriminator.py
and a Dataset/ folder with mp3s (or a ready wav24/ folder).
"""
import os, glob, math, time, random, subprocess, warnings
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

from rvq import ResidualVectorQuantization
from encoder import Encoder
from decoder import Decoder
from transformer import CodeBookTransformer
from descriminator import MultiSTFTDiscriminator, STFTDiscriminator
warnings.filterwarnings("ignore", message=".*mel filterbank has all zero values.*")

# ========================================================================================
# SETTINGS  (everything you may want to change is here)
# ========================================================================================
RUN_TESTS     = False     # True = run sanity tests first
TRAIN_CODEC   = True
TRAIN_LM      = True
RESUME        = False     # True = continue from ckpt_last.pt instead of starting fresh

# --- data (kept small so 4 GB RAM is enough) ---
MAX_FILES     = 40        # use at most this many wavs
MAX_SECONDS   = 60        # only load the first N seconds of each wav

# --- codec training ---
STEPS         = 800
WARMUP        = 600       # GAN starts here; set >= STEPS for reconstruction only
BATCH         = 8        # frames/step (BATCH*CROP_FRAMES) should be well above codebook_size
CROP_FRAMES   = 40        # crop length in latent frames
LR            = 3e-4
MAX_MINUTES   = 30        # time cap for codec
MEL_SCALES    = (7, 9)    # fewer = faster
LOSS_W        = dict(t=0.1, f=1.0, g=3.0, feat=3.0, w=0.25)  # w = commitment weight (summed over codebooks)
LOG_EVERY     = 50
SAVE_EVERY    = 250

# --- LM training ---
LM_STEPS      = 800
LM_BATCH      = 8
LM_FRAMES     = 40        # keep <= max_offset below
LM_LR         = 3e-4
LM_MAX_MINUTES = 8
LM_VAL_EVERY  = 200


class EncodecConfig:       # small version
    encoder_in_channels = 1
    encoder_hidden_channels = 16
    encoder_out_channels = 64
    encoder_n_conv_blocK = 4
    codebook_size = 128
    latent_dim = 64             # must equal encoder_out_channels and decoder_latent_dim
    n_codebook = 4
    decay = 0.99
    dead_code_threshold = 0.1   # code used < ~1 frame per 10 steps counts as dead
    decoder_latent_dim = 64
    decoder_hidden_channel = 16
    decoder_out_channel = 1
    decoder_conv_blocks = 4
    n_head = 4                  # must divide latent_dim
    n_layer = 2
    max_offset = 40
    feedf_dim = 128
    window_size = 3
    n_ffts = (256, 128)


# ========================================================================================
# Losses
# ========================================================================================
class MelLoss(nn.Module):
    def __init__(self, sr=24000, scales=(6, 7, 8, 9, 10, 11)):
        super().__init__()
        self.mels = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2 ** i, hop_length=2 ** i // 4,
                                                 n_mels=64, normalized=True) for i in scales])

    def forward(self, x, x_hat):  # (B, samples)
        loss = 0
        for m in self.mels:
            s, sh = m(x), m(x_hat)
            loss = loss + (s - sh).abs().mean() + ((s - sh).pow(2).sum(-2) + 1e-8).sqrt().mean()
        return loss / len(self.mels)

def d_loss_fn(real_outs, fake_outs):
    return torch.stack([F.relu(1 - lr).mean() + F.relu(1 + lf).mean()
                        for (lr, _), (lf, _) in zip(real_outs, fake_outs)]).mean()

def g_adv_loss_fn(fake_outs):
    return torch.stack([F.relu(1 - lf).mean() for lf, _ in fake_outs]).mean()

def feat_match_loss_fn(real_outs, fake_outs, eps=1e-8):
    return torch.stack([(a.detach() - b).abs().mean() / (a.detach().abs().mean() + eps)
                        for (_, fr), (_, ff) in zip(real_outs, fake_outs)
                        for a, b in zip(fr, ff)]).mean()


# ========================================================================================
# Model
# ========================================================================================
class Encodec(nn.Module):
    def __init__(self, config: EncodecConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.encoder_in_channels, config.encoder_hidden_channels,
                               config.encoder_out_channels, config.encoder_n_conv_blocK)
        self.rvq = ResidualVectorQuantization(config.codebook_size, config.latent_dim, config.n_codebook,
                                              config.decay, config.dead_code_threshold)
        self.decoder = Decoder(config.decoder_latent_dim, config.decoder_hidden_channel,
                               config.decoder_out_channel, config.decoder_conv_blocks)
        self.codebook_transformer = CodeBookTransformer(
            config.n_codebook, config.codebook_size, config.latent_dim, config.n_head,
            config.n_layer, config.max_offset, config.feedf_dim, config.window_size)
        self.descriminator = MultiSTFTDiscriminator(n_ffts=config.n_ffts)

    def forward(self, x):  # x: (B, 1, T)
        latent = self.encoder(x)
        quantized, residual, perplexity, codes, commit = self.rvq(latent)
        x_hat = self.decoder(quantized)
        T = x.shape[-1]
        if x_hat.shape[-1] > T:
            x_hat = x_hat[..., :T]
        elif x_hat.shape[-1] < T:
            x_hat = F.pad(x_hat, (0, T - x_hat.shape[-1]))
        return x_hat, codes, commit, perplexity

    @torch.no_grad()
    def get_codes(self, x):  # -> (B, n_q, T') long
        codes = self.rvq(self.encoder(x))[3]
        n = self.config.n_codebook
        if codes.shape[0] == n and codes.shape[1] != n:
            codes = codes.permute(1, 0, 2)
        return codes.long()

    @torch.no_grad()
    def downsample_factor(self, n=24000):
        was = self.training
        self.eval()
        t = self.encoder(torch.zeros(1, 1, n)).shape[-1]
        self.train(was)
        return round(n / t)

    def gen_parameters(self):
        return [p for m in (self.encoder, self.rvq, self.decoder) for p in m.parameters()]


# ========================================================================================
# Data
# ========================================================================================
def prep(src="Dataset", dst="wav24"):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for root, _, files in os.walk(src):
        for f in sorted(files):
            if f.lower().endswith(".mp3"):
                out = os.path.join(dst, os.path.splitext(f)[0] + ".wav")
                if not os.path.exists(out):
                    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i",
                                    os.path.join(root, f), "-ac", "1", "-ar", "24000", out], check=True)
                n += 1
    print(f"{n} mp3 files -> {dst}/ ({len(glob.glob(dst + '/*.wav'))} wavs)")


class Crops(Dataset):
    def __init__(self, files, crop, n):
        self.crop, self.n, self.wavs = crop, n, []
        for f in files:
            w, _ = torchaudio.load(f, num_frames=MAX_SECONDS * 24000)  # only first N sec -> saves RAM
            if w.shape[1] > crop:
                self.wavs.append(w / w.abs().max().clamp(min=1e-5) * 0.95)
        assert self.wavs, f"No usable wavs (need > {crop} samples each). cwd={os.getcwd()}"

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = None
        for _ in range(10):  # skip near-silent crops
            w = random.choice(self.wavs)
            s = random.randint(0, w.shape[1] - self.crop)
            x = w[:, s:s + self.crop]
            if x.pow(2).mean() > 1e-5:
                break
        return x  # (1, crop)


def scalar(v):
    return (v if torch.is_tensor(v) else torch.as_tensor(v)).float().mean().item()

def split_files(folder="wav24"):
    files = sorted(glob.glob(f"{folder}/*.wav"))
    if len(files) > MAX_FILES:
        files = random.Random(0).sample(files, MAX_FILES)
        files.sort()
    val = set(files[::8])
    return [f for f in files if f not in val], sorted(val)


@torch.no_grad()
def init_codebooks(model, x):
    """Start every codebook on real encoder outputs. Works with the ORIGINAL rvq.py (uses its buffers only)."""
    z = model.encoder(x)
    flat0 = z.reshape(-1, model.config.latent_dim)
    print(f"[init] latent mean={flat0.mean().item():.4f}  std across frames={flat0.std(0).mean().item():.4f}")
    residual = z
    for cb in model.rvq.codebooks:
        flat = residual.reshape(-1, cb.latent_dim)
        N, K = flat.shape[0], cb.codebook_size
        idx = torch.randperm(N)[:K] if N >= K else torch.randint(0, N, (K,))
        vecs = flat[idx]
        cb.embedding.weight.data.copy_(vecs); cb.ema_emb.copy_(vecs); cb.em_cluster_size.fill_(1.0)
        q = cb.embedding(torch.cdist(flat, cb.embedding.weight).argmin(-1)).view(residual.shape)
        residual = residual - q


# ========================================================================================
# Train codec
# ========================================================================================
def train_codec():
    model = Encodec(EncodecConfig())
    disc = model.descriminator
    hop = model.downsample_factor()
    crop = hop * CROP_FRAMES
    W = LOSS_W
    print(f"[codec] downsample={hop} ({24000 / hop:.0f} frames/s), crop={crop}, steps={STEPS}, "
          f"GAN from {WARMUP if WARMUP < STEPS else 'never'}")

    train_files, val_files = split_files()
    opt_g = torch.optim.Adam(model.gen_parameters(), lr=LR, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.9))
    start = 0
    if RESUME and os.path.exists("ckpt_last.pt"):
        ck = torch.load("ckpt_last.pt", map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        start = ck["step"] + 1
        print("[codec] resumed from step", start)
    if start >= STEPS:
        print("[codec] checkpoint already past STEPS, skipping"); return

    dl = DataLoader(Crops(train_files, crop, (STEPS - start) * BATCH), batch_size=BATCH,
                    num_workers=0, drop_last=True)
    val_set = Crops(val_files or train_files, crop * 3, 4)
    mel_loss = MelLoss(24000, scales=MEL_SCALES)

    def save(step):
        sd = {k: v for k, v in model.state_dict().items()
              if not k.startswith(("descriminator", "codebook_transformer"))}
        torch.save({"model": sd, "disc": disc.state_dict(), "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(), "step": step}, "ckpt_last.pt")

    if start == 0:
        init_codebooks(model, next(iter(dl)))
    model.train()
    t0, last = time.time(), start
    for step, x in zip(range(start, STEPS), dl):
        last = step
        use_gan = step >= WARMUP
        x_hat, codes, commit, ppl = model(x)

        if use_gan and random.random() < 2 / 3:
            loss_d = d_loss_fn(disc(x), disc(x_hat.detach()))
            opt_d.zero_grad(set_to_none=True); loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0); opt_d.step()

        l_t = (x - x_hat).abs().mean()
        l_f = mel_loss(x.squeeze(1), x_hat.squeeze(1))
        loss_g = W["t"] * l_t + W["f"] * l_f + W["w"] * commit.mean()
        if use_gan:
            with torch.no_grad():
                real_outs = disc(x)
            fake_outs = disc(x_hat)
            loss_g = loss_g + W["g"] * g_adv_loss_fn(fake_outs) + W["feat"] * feat_match_loss_fn(real_outs, fake_outs)
        opt_g.zero_grad(set_to_none=True); loss_g.backward()
        torch.nn.utils.clip_grad_norm_(model.gen_parameters(), 1.0); opt_g.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                lat_std = model.encoder(x[:2]).std().item()  # ~0 means the encoder collapsed
            print(f"latent_std={lat_std:.4f}", end="  ")
            print(f"{step} t={l_t.item():.4f} mel={l_f.item():.3f} commit={scalar(commit):.4f} "
                  f"ppl={scalar(ppl):.0f} gan={use_gan} {(time.time() - t0) / (step - start + 1):.2f}s/step", flush=True)
        if step % SAVE_EVERY == 0 and step > start:
            model.eval()
            with torch.no_grad():
                v = val_set[0].unsqueeze(0)
                y, *_ = model(v)
                print(f"   val l1={(v - y).abs().mean().item():.4f}")
                torchaudio.save(f"val_{step}_orig.wav", v[0], 24000)
                torchaudio.save(f"val_{step}_recon.wav", y[0].clamp(-1, 1), 24000)
            model.train(); save(step)
        if (time.time() - t0) / 60 > MAX_MINUTES:
            print(f"[codec] time cap hit at step {step}"); break
    save(last)
    print("[codec] done -> ckpt_last.pt")


# ========================================================================================
# Train codebook transformer on frozen codec
# ========================================================================================
def train_lm():
    cfg = EncodecConfig()
    model = Encodec(cfg)
    model.load_state_dict(torch.load("ckpt_last.pt", map_location="cpu")["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hop = model.downsample_factor()
    fps = 24000 / hop
    lm = model.codebook_transformer
    for p in lm.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(lm.parameters(), lr=LM_LR)

    crop = hop * LM_FRAMES
    train_files, val_files = split_files()
    dl = DataLoader(Crops(train_files, crop, LM_STEPS * LM_BATCH), batch_size=LM_BATCH,
                    num_workers=0, drop_last=True)
    val = DataLoader(Crops(val_files or train_files, crop, 32), batch_size=LM_BATCH)

    def kbps(ce):
        return ce / math.log(2) * cfg.n_codebook * fps / 1000

    print(f"[lm] steps={LM_STEPS}, chance ce={math.log(cfg.codebook_size):.3f}")
    t0 = time.time()
    for step, x in zip(range(LM_STEPS), dl):
        lm.train()
        loss = lm.loss(model.get_codes(x))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            print(step, f"ce={loss.item():.3f} est_kbps={kbps(loss.item()):.2f}", flush=True)
        if step % LM_VAL_EVERY == 0 and step > 0:
            lm.eval()
            with torch.no_grad():
                ce = sum(lm.loss(model.get_codes(v)).item() for v in val) / len(val)
            print(f"   VAL ce={ce:.3f} est_kbps={kbps(ce):.2f} nominal={cfg.n_codebook * 8 * fps / 1000:.2f}")
            torch.save(lm.state_dict(), "lm.pt")
        if (time.time() - t0) / 60 > LM_MAX_MINUTES:
            print(f"[lm] time cap hit at step {step}"); break
    torch.save(lm.state_dict(), "lm.pt")
    print("[lm] done -> lm.pt")


# ========================================================================================
# Tests
# ========================================================================================
def run_tests():
    torch.manual_seed(0)
    cfg = EncodecConfig()
    model = Encodec(cfg)
    HOP = model.downsample_factor()
    X = torch.randn(2, 1, HOP * 40).clamp(-1, 1) * 0.1

    def t_disc_shapes():
        logits, feats = STFTDiscriminator(256)(torch.randn(2, 12800))
        T = feats[0].shape[2]
        assert all(f.shape[2] == T for f in feats) and logits.shape[2] == T, [f.shape[2] for f in feats]

    def t_multi_disc():
        assert len(MultiSTFTDiscriminator(cfg.n_ffts)(X)) == len(cfg.n_ffts)

    def t_roundtrip():
        x_hat, codes, commit, ppl = model(X)
        assert x_hat.shape == X.shape, (x_hat.shape, X.shape)
        c = model.get_codes(X)
        assert c.shape == (2, cfg.n_codebook, 40), c.shape
        assert 0 <= c.min() and c.max() < cfg.codebook_size

    def t_backward():
        model.zero_grad()
        x_hat, _, commit, _ = model(X)
        loss = (X - x_hat).abs().mean() + MelLoss(scales=MEL_SCALES)(X.squeeze(1), x_hat.squeeze(1)) + commit.mean()
        assert torch.isfinite(loss); loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
            "encoder gets no gradient -> straight-through broken"

    def t_gan():
        x_hat, *_ = model(X)
        d = model.descriminator
        ld = d_loss_fn(d(X), d(x_hat.detach())); assert torch.isfinite(ld); ld.backward()
        fake = d(x_hat)
        lg = g_adv_loss_fn(fake) + feat_match_loss_fn(d(X), fake); assert torch.isfinite(lg); lg.backward()

    def t_lm():
        loss = model.codebook_transformer.loss(model.get_codes(X)); loss.backward()
        chance = math.log(cfg.codebook_size)
        print(f"    lm ce={loss.item():.3f}, chance={chance:.3f}")
        assert 0.7 * chance < loss.item() < 1.5 * chance

    def t_overfit():
        t = torch.arange(HOP * 40) / 24000
        X2 = (0.3 * torch.sin(2 * math.pi * 220 * t) + 0.2 * torch.sin(2 * math.pi * 660 * t)).repeat(2, 1, 1)
        m = Encodec(cfg); m.train()
        opt = torch.optim.Adam(m.gen_parameters(), lr=1e-3)
        mel, hist = MelLoss(scales=MEL_SCALES), []
        for _ in range(60):
            x_hat, _, commit, _ = m(X2)
            loss = (X2 - x_hat).abs().mean() + mel(X2.squeeze(1), x_hat.squeeze(1)) + commit.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            hist.append(loss.item())
        first, last = sum(hist[:10]) / 10, sum(hist[-10:]) / 10
        print(f"    overfit {first:.3f} -> {last:.3f}")
        assert last < first

    failed = 0
    for name, fn in [("disc_shapes", t_disc_shapes), ("multi_disc", t_multi_disc), ("roundtrip", t_roundtrip),
                     ("backward", t_backward), ("gan", t_gan), ("lm", t_lm), ("overfit", t_overfit)]:
        try:
            fn(); print(f"PASS {name}")
        except Exception as e:
            failed += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all tests passed" if not failed else f"{failed} test(s) failed")
    return failed == 0


# ========================================================================================
if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 1)
    if not glob.glob("wav24/*.wav"):
        prep()
    if RUN_TESTS and not run_tests():
        raise SystemExit("fix failing tests first")
    if TRAIN_CODEC:
        train_codec()
    if TRAIN_LM:
        train_lm()
