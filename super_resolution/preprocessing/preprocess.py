import pickle as pkl
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..config import DataConfig


def fit_image(img: Image.Image) -> Image.Image:
    """Scale an image to the target size."""
    height, width = img.size
    if height > width:
        new_height = DataConfig.target_height
        new_width = int(width * (new_height / height))
    else:
        new_width = DataConfig.target_width
        new_height = int(height * (new_width / width))

    resized = img.resize((new_width, new_height))
    # crop the image to the target size (center crop)
    left = (new_width - DataConfig.target_width) // 2
    top = (new_height - DataConfig.target_height) // 2
    right = left + DataConfig.target_width
    bottom = top + DataConfig.target_height
    cropped = resized.crop((left, top, right, bottom))

    return cropped


def load_paths(path: Path) -> list[Path]:
    """Load all the paths in a directory."""
    return list(path.glob("*.jpg"))


def is_too_small(img: Image.Image) -> bool:
    """Check if an image is too small."""
    height, width = img.size
    return height < DataConfig.target_height or width < DataConfig.target_width


def equalize_sets(*data_sets: list[np.ndarray]) -> list[list[np.ndarray]]:
    """Equalize the number of images in each set."""
    min_len = min(len(data_set) for data_set in data_sets)
    return [data_set[:min_len] for data_set in data_sets]


def main() -> None:
    """Load images, filter out too small images, scale and save rest."""
    cat_paths = load_paths(Path("data/unpacked/Cat"))
    dog_paths = load_paths(Path("data/unpacked/Dog"))
    print(f"Got {len(cat_paths)} cat paths and {len(dog_paths)} dog paths.")

    cat_images = []
    dog_images = []
    for path in tqdm(cat_paths):
        try:
            img = Image.open(path)
            if is_too_small(img):
                continue
            cat_images.append(np.array(fit_image(img)))
        except Exception as e:
            print(f"Path: {path} | Error: {e}")

    for path in tqdm(dog_paths):
        try:
            img = Image.open(path)
            if is_too_small(img):
                continue
            dog_images.append(np.array(fit_image(img)))
        except Exception as e:
            print(f"Path: {path} | Error: {e}")

    print(
        f"Got {len(cat_images)} cat images and {len(dog_images)} dog images after filtering."
    )
    equalized_cats, equalized_dogs = equalize_sets(cat_images, dog_images)
    print(
        f"Equalized to {len(equalized_cats)} cat images and {len(equalized_dogs)} dog images."
    )

    with open(DataConfig.processed_cats, "wb") as f:
        pkl.dump(equalized_cats, f)
    with open(DataConfig.processed_dogs, "wb") as f:
        pkl.dump(equalized_dogs, f)


if __name__ == "__main__":
    main()
