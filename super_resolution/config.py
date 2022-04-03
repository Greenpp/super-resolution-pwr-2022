RANDOM_SEED = 42


class DataConfig:
    raw_data = "data/raw.tar.gz"
    unpacked_data = "data/unpacked"
    processed_dogs = "data/processed_dogs.pkl"
    processed_cats = "data/processed_cats.pkl"

    # values based on data_analysis.py
    target_width = 240
    target_height = 320

    split_ration = 0.8
    compressed_width = 30
    compressed_height = 40


class TrainingConfig:
    batch_size = 32
    num_workers = 4
    max_epochs = 10
    lr = 0.001
