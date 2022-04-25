import pickle as pkl
from pathlib import Path

from sklearn.model_selection import train_test_split

from ..config import RANDOM_SEED, DataConfig


def load_names(path: Path) -> list[str]:
    files = path.iterdir()
    return [f.name for f in files]


def save_names(path: Path, names: list[str]) -> None:
    with open(path, "wb") as f:
        pkl.dump(names, f)


def main() -> None:
    original_dogs = Path(DataConfig.processed_dogs)
    original_cats = Path(DataConfig.processed_cats)

    print("Loading names...")
    dog_names = load_names(original_dogs)
    cat_names = load_names(original_cats)

    print("Splitting data...")
    dog_train, dog_val = train_test_split(
        dog_names, test_size=DataConfig.split_ratio, random_state=RANDOM_SEED
    )

    cat_train, cat_val = train_test_split(
        cat_names, test_size=DataConfig.split_ratio, random_state=RANDOM_SEED
    )

    print("Saving data...")
    save_names(Path(DataConfig.train_dogs), dog_train)
    save_names(Path(DataConfig.train_cats), cat_train)
    save_names(Path(DataConfig.val_dogs), dog_val)
    save_names(Path(DataConfig.val_cats), cat_val)


if __name__ == "__main__":
    main()
