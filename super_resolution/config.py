RANDOM_SEED = 42


class DataConfig:
    raw_data = "data/raw.tar.gz"

    unpacked_data = "data/unpacked"
    unpacked_dogs = "data/unpacked/Dog"
    unpacked_cats = "data/unpacked/Cat"

    processed_dogs = "data/processed/Dog"
    processed_cats = "data/processed/Cat"

    scaled_dogs = "data/scaled/Dog"
    scaled_cats = "data/scaled/Cat"

    train_dogs = "data/dog_train.pkl"
    train_cats = "data/cat_train.pkl"
    val_dogs = "data/dog_val.pkl"
    val_cats = "data/cat_val.pkl"

    # values based on data_analysis.py
    target_width = 240
    target_height = 320

    split_ratio = 0.2
    compressed_width = 30
    compressed_height = 40


class TrainingConfig:
    batch_size = 128
    num_workers = 4
    max_epochs = 50
    lr = 0.0001
