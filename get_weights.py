from pathlib import Path
import shutil

for model_path in Path("./models").iterdir():
    print(f"Extracting weights for: {model_path.stem}")
    ckpt_path = model_path / "weights" / "best.pt"
    ckpt_path.rename(f"./models/{model_path.stem}.pt")
    shutil.rmtree(model_path)
