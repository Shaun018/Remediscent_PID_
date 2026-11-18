
from pathlib import Path
import pandas as pd
import hashlib
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

SOURCE = Path("dataset/processed")
OUTPUT = Path("dataset/split")

df = pd.read_csv(SOURCE / "metadata.csv")

valid_records = []
seen = set()

corrupt = 0
duplicates = 0

print("Checking images...")

for _, row in df.iterrows():
    path = Path(row["local_path"])

    try:
        with Image.open(path) as img:
            img.verify()

        # Remove exact duplicates within each class
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = (row["class_name"], digest)

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        valid_records.append(row)

    except Exception:
        corrupt += 1

clean_df = pd.DataFrame(valid_records)

print("Valid images:", len(clean_df))
print("Corrupt images:", corrupt)
print("Exact duplicates:", duplicates)

# Split each class independently
splits = {"train": [], "validation": [], "test": []}

for class_name, group in clean_df.groupby("class_name"):

    train, remaining = train_test_split(
        group,
        test_size=0.30,
        random_state=42
    )

    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=42
    )

    splits["train"].append(train)
    splits["validation"].append(validation)
    splits["test"].append(test)

for split_name, groups in splits.items():

    split_df = pd.concat(groups, ignore_index=True)

    for _, row in split_df.iterrows():
        source = Path(row["local_path"])
        destination = (
            OUTPUT / split_name /
            row["class_name"] / source.name
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    split_df.to_csv(
        SOURCE / f"{split_name}_metadata.csv",
        index=False
    )

    print(f"{split_name}: {len(split_df)} images")

print("\n========== FINAL CLASS DISTRIBUTION ==========")

for split_name in splits:
    print(f"\n{split_name.upper()}")

    split_df = pd.concat(splits[split_name])
    print(split_df.groupby("class_name").size())

print("\nDataset splitting completed!")
