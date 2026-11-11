"""
Resume training of the codec from its saved checkpoint (NOT from scratch).
Run it:    python resume.py
"""
import os, shutil, importlib
import torch

# ------------------------------- settings -------------------------------------------------------
MODEL_FILE      = "model"   # your training file name without .py (matches model.py)
EXTRA_STEPS     = 4000      # how many MORE steps on top of what is already saved
LR              = 2e-4      # starting learning rate for this resumed run
MAX_MINUTES     = 150       # time cap for this run
GAN_LAST_STEPS  = 0         # >0: GAN only for last N steps; 0 = reconstruction only
GAN_WEIGHTS     = dict(t=0.1, f=1.0, g=1.0, feat=2.0, w=0.1)
TRAIN_LM_AFTER  = False     # Set True on the final run if you want to update LM after codec
LM_STEPS        = 2000
LM_MAX_MINUTES  = 25
# ------------------------------------------------------------------------------------------------

m = importlib.import_module(MODEL_FILE)
torch.set_num_threads(os.cpu_count() or 1)

assert os.path.exists(m.CKPT_FILE), f"{m.CKPT_FILE} not found (make sure you ran it at least once to create a checkpoint)"

ck = torch.load(m.CKPT_FILE, map_location="cpu")
done = ck["step"] + 1
backup = m.CKPT_FILE.replace(".pt", f"_backup_step{done}.pt")
shutil.copy(m.CKPT_FILE, backup)
print(f"Checkpoint is at step {done}; backup created -> {backup}")

# Safely update learning rates inside the saved optimizer states
for key in ("opt_g", "opt_d"):
    if key in ck:
        for g in ck[key]["param_groups"]:
            g["lr"] = LR
torch.save(ck, m.CKPT_FILE)

# Safely inject/override attributes in the imported module to prevent AttributeError
setattr(m, "RESUME", True)
setattr(m, "STEPS", done + EXTRA_STEPS)
setattr(m, "WARMUP", m.STEPS - GAN_LAST_STEPS if GAN_LAST_STEPS > 0 else m.STEPS)
setattr(m, "MAX_MINUTES", MAX_MINUTES)
setattr(m, "LR", LR)
setattr(m, "LM_STEPS", LM_STEPS)
setattr(m, "LM_MAX_MINUTES", LM_MAX_MINUTES)
if GAN_LAST_STEPS > 0:
    setattr(m, "LOSS_W", GAN_WEIGHTS)

print(f"Resuming training: steps {done} -> {m.STEPS}, lr={LR}, time cap {MAX_MINUTES} min")

# Run training
m.train_codec()
if TRAIN_LM_AFTER:
    m.train_lm()
