from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..config import DataConfig


def load_paths(path: Path) -> list[Path]:
    """Load all the paths in a directory."""
    return list(path.glob("*.jpg"))


def scale_img(img: Image.Image) -> Image.Image:
    return img.resize((DataConfig.compressed_width, DataConfig.compressed_height))


def main() -> None:
    in_dogs = Path(DataConfig.processed_dogs)
    in_cats = Path(DataConfig.processed_cats)

    out_dogs = Path(DataConfig.scaled_dogs)
    out_dogs.mkdir(exist_ok=True, parents=True)
    out_cats = Path(DataConfig.scaled_cats)
    out_cats.mkdir(exist_ok=True, parents=True)

    for path in tqdm(load_paths(in_dogs), desc="Scaling dogs"):
        img = Image.open(path)
        scaled = scale_img(img)
        scaled.save(out_dogs / path.name)

    for path in tqdm(load_paths(in_cats), desc="Scaling cats"):
        img = Image.open(path)
        scaled = scale_img(img)
        scaled.save(out_cats / path.name)


if __name__ == "__main__":
    main()
