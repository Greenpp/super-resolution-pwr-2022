import tarfile

from tqdm import tqdm

from ..config import DataConfig


def unpack_file(file_path: str, destination_path: str) -> None:
    """Unpack a file to a destination path."""
    with tarfile.open(file_path, "r") as tar:
        members = tar.getmembers()
        for m in tqdm(members):
            tar.extract(m, destination_path)


def main() -> None:
    """Unpack the raw data."""
    unpack_file(DataConfig.raw_data, DataConfig.unpacked_data)


if __name__ == "__main__":
    main()
