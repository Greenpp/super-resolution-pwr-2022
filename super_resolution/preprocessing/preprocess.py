from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..config import DataConfig


def fit_image(img: Image.Image) -> Image.Image:
    """Scale an image to the target size."""
    width, height = img.size
    if width > height:
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
    width, height = img.size
    return height < DataConfig.target_height or width < DataConfig.target_width


def is_gray(img: Image.Image) -> bool:
    """Check if an image is grayscale."""
    return len(img.getbands()) == 1


def equalize_sets(*data_sets: list[np.ndarray]) -> list[list[np.ndarray]]:
    """Equalize the number of images in each set."""
    min_len = min(len(data_set) for data_set in data_sets)
    return [data_set[:min_len] for data_set in data_sets]


def save_images(images: list[np.ndarray], path: Path) -> None:
    """Save images to a file."""
    for i, img in tqdm(enumerate(images), desc="Saving images", total=len(images)):
        Image.fromarray(img.transpose(1, 2, 0)).save(path / f"{i}.jpg")


def process_paths(paths: list[Path]) -> list[np.ndarray]:
    """Load, filter and process images."""
    images = []
    for path in tqdm(paths):
        try:
            img = Image.open(path)
            if is_too_small(img):
                continue
            if is_gray(img):
                continue
            img = np.array(fit_image(img))
            if img.shape != (DataConfig.target_height, DataConfig.target_width, 3):
                raise ValueError(f"Image {path} has shape {img.shape}")
            images.append(img.transpose(2, 0, 1))
        except Exception as e:
            print(f"Path: {path} | Error: {e}")
    return images


def main() -> None:
    """Load images, filter out too small images, scale and save rest."""
    cat_paths = load_paths(Path(DataConfig.unpacked_cats))
    dog_paths = load_paths(Path(DataConfig.unpacked_dogs))
    print(f"Got {len(cat_paths)} cat paths and {len(dog_paths)} dog paths.")

    print("Processing cat images...")
    cat_images = process_paths(cat_paths)
    print(f"Got {len(cat_images)} cat images.")
    print("Processing dog images...")
    dog_images = process_paths(dog_paths)
    print(f"Got {len(dog_images)} dog images.")

    equalized_cats, equalized_dogs = equalize_sets(cat_images, dog_images)
    print(
        f"Equalized to {len(equalized_cats)} cat images and {len(equalized_dogs)} dog images."
    )

    output_cats = Path(DataConfig.processed_cats)
    output_cats.mkdir(parents=True, exist_ok=True)
    output_dogs = Path(DataConfig.processed_dogs)
    output_dogs.mkdir(parents=True, exist_ok=True)

    save_images(equalized_cats, output_cats)
    save_images(equalized_dogs, output_dogs)


if __name__ == "__main__":
    main()
