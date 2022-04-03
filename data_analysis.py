# %%
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm

sns.set()


def get_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def get_img_paths(path: Path) -> list[Path]:
    return list(path.glob("*.jpg"))


def get_sizes(paths: list[Path]) -> list[tuple[int, int]]:
    sizes = []
    for path in tqdm(paths):
        try:
            sizes.append(get_image_size(path))
        except Exception as e:
            print(f"Path: {path} | Error: {e}")
    return sizes


# %%
cat_paths = get_img_paths(Path("data/unpacked/Cat"))
dog_paths = get_img_paths(Path("data/unpacked/Dog"))
print(f"Got {len(cat_paths)} cat images and {len(dog_paths)} dog paths.")

cat_sizes = get_sizes(cat_paths)
dog_sizes = get_sizes(dog_paths)
all_sizes = cat_sizes + dog_sizes

sizes_arr = np.array(all_sizes)
print(f"Loaded total of {len(all_sizes)} images.")
# %%
# plot heights and widths
bins = 25
sns.histplot(sizes_arr[:, 0], bins=bins)
plt.title("Distribution of Image Heights")
plt.show()
sns.histplot(sizes_arr[:, 1], bins=bins)
plt.title("Distribution of Image Widths")
plt.show()

sizes_count = Counter(all_sizes)
most_common_num = 10
most_common_sizes = sizes_count.most_common(most_common_num)
min_most_common_width = min(size[0][1] for size in most_common_sizes)
min_most_common_height = min(size[0][0] for size in most_common_sizes)
print(f"Min width of {most_common_num} most common sizes: {min_most_common_width}")
print(f"Min height of {most_common_num} most common sizes: {min_most_common_height}")

imgs_smaller_than_min = 0
for height, width in all_sizes:
    if height < min_most_common_height or width < min_most_common_width:
        imgs_smaller_than_min += 1

print(
    f"Percentage of images smaller than min: {round((imgs_smaller_than_min / len(all_sizes)) * 100, 2)}%"
)
cat_smaller_than_min = 0
for height, width in cat_sizes:
    if height < min_most_common_height or width < min_most_common_width:
        cat_smaller_than_min += 1
dog_smaller_than_min = 0
for height, width in dog_sizes:
    if height < min_most_common_height or width < min_most_common_width:
        dog_smaller_than_min += 1
print(
    f"Percentage of cats smaller than min: {round((cat_smaller_than_min / len(cat_sizes)) * 100, 2)}%"
)
print(
    f"Percentage of dogs smaller than min: {round((dog_smaller_than_min / len(dog_sizes)) * 100, 2)}%"
)

# %%
