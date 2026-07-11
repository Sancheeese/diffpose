from pathlib import Path

models = sorted(Path("checkpoints/").glob(f"specimen_01_epoch*.ckpt"))
for model_name in models:
    print(model_name)

