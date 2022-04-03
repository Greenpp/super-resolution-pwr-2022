import tarfile

from ..config import DataConfig


def unpack_file(file_path: str, destination_path: str) -> None:
    """Unpack a file to a destination path."""
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(destination_path)
    return None


def main() -> None:
    """Unpack the raw data."""
    unpack_file(DataConfig.raw_data, DataConfig.unpacked_data)
    return None


if __name__ == "__main__":
    main()
