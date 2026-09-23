import argparse
from pathlib import Path

from PIL import Image

from models import RESOLUTION
from prompts import TRIGGER


def main(src, dst):
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(src).glob("*.png")):
        image = Image.open(path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.NEAREST)
        image.save(dst_path / path.name)
        (dst_path / f"{path.stem}.txt").write_text(TRIGGER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/sprites")
    parser.add_argument("--dst", default="data/train")
    args = parser.parse_args()
    main(args.src, args.dst)
