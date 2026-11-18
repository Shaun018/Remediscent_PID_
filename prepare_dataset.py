
from pathlib import Path
import pandas as pd
import shutil

ROOT = Path("dataset/raw/sampleData")
OUTPUT = Path("dataset/processed")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

consumer_excel = ROOT / "consumer/sample_images.xlsx"
reference_excel = ROOT / "reference/reference_sample.xlsx"

# Index all available image files by filename
image_index = {}

for file in ROOT.rglob("*"):
    if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
        image_index[file.name.lower()] = file

# Read consumer metadata
df = pd.read_excel(consumer_excel)

df["filename"] = df["Image"].apply(
    lambda x: Path(str(x).replace("\\", "/")).name
)

df["source_path"] = df["filename"].str.lower().map(image_index)

# Keep only records with matching image files
matched = df.dropna(subset=["source_path"]).copy()

# Remove duplicate image records
matched = matched.drop_duplicates(subset=["source_path"])

# Create class folder
class_folder = OUTPUT / "consumer" / "fluoxetine_10mg_capsule"
class_folder.mkdir(parents=True, exist_ok=True)

# Copy images and record their processed paths
processed_paths = []

for index, row in matched.iterrows():
    source = Path(row["source_path"])
    destination = class_folder / f"fluoxetine_{index}{source.suffix.lower()}"

    shutil.copy2(source, destination)
    processed_paths.append(str(destination))

matched["processed_path"] = processed_paths
matched["class_name"] = "fluoxetine_10mg_capsule"

# Save metadata
metadata_path = OUTPUT / "consumer_metadata.csv"
matched.to_csv(metadata_path, index=False)

# Save reference metadata separately
reference_df = pd.read_excel(reference_excel)
reference_df.to_csv(OUTPUT / "reference_metadata.csv", index=False)

print("\n========== DATASET PREPARATION ==========")
print("Consumer metadata records:", len(df))
print("Matched images:", len(matched))
print("Unmatched records:", len(df) - len(matched))
print("Unique image files:", len(image_index))
print("Medicine classes:", matched["class_name"].nunique())
print("Processed images:", class_folder)
print("Metadata saved:", metadata_path)
print("Reference metadata saved.")

print("\nMatched image filenames:")
for path in matched["processed_path"]:
    print(Path(path).name)
