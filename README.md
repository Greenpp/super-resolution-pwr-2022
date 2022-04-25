# Super-resolution

## How to run

1. Install requirements

   ```sh
   poetry install
   ```

2. Pull data

   ```sh
   dvc pull data/raw.tar.gz.dvc
   ```

3. Preprocess data

   ```sh
   dvc repro
   ```

4. Login to wandb

   ```sh
   wandb login
   ```

5. Run training

   ```sh
   python main.py --help
   ```

## How to contribute

### Tools

[Poetry](https://python-poetry.org/)

[Pre-commit](https://pre-commit.com/)

[Pyenv](https://github.com/pyenv/pyenv)

### Steps

1. Clone repository

2. Install pre-commit hooks

   ```sh
   pre-commit install
   ```

3. Install dependencies

   ```sh
   poetry install
   ```
