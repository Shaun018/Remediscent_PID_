
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import re
import time

ROOT_URL = "https://data.lhncbc.nlm.nih.gov/public/Pills/"
EXCEL = "dataset/raw/directory_consumer_grade_images.xlsx"
OUTPUT = Path("dataset/processed/consumer")
MAX_IMAGES = 100

df = pd.read_excel(EXCEL)

# Select consumer test images only
df = df[
    df["Layout"].astype(str).str.strip().str.upper() == "C3PI_TEST"
].copy()

# Find the 10 products with the most unique image paths
summary = (
    df.groupby(["NDC11", "Part", "Name"])["Image"]
    .nunique()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(10)
)

print("\nSELECTED 10 CLASSES")
print(summary.to_string(index=False))

session = requests.Session()
session.headers.update({"User-Agent": "MedRemind-Research/1.0"})

def download_image(item):
    url, destination = item

    if destination.exists():
        return str(destination), "EXISTS"

    try:
        response = session.get(url, timeout=40)
        response.raise_for_status()

        if not response.headers.get("Content-Type", "").lower().startswith("image/"):
            return str(destination), "NOT_IMAGE"

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)

        return str(destination), "DOWNLOADED"

    except Exception as e:
        return str(destination), f"FAILED: {e}"

tasks = []
class_records = []
class_map = []

for class_number, row in enumerate(summary.itertuples(index=False), 1):

    ndc = str(int(row.NDC11)).zfill(11)
    part = str(row.Part)
    name = str(row.Name)

    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    class_name = f"class_{class_number:02d}_{safe_name}"

    class_map.append({
        "class_id": class_number - 1,
        "class_name": class_name,
        "medicine_name": name,
        "ndc11": ndc,
        "part": part
    })

    subset = df[
        (df["NDC11"] == row.NDC11) &
        (df["Part"] == row.Part) &
        (df["Name"] == row.Name)
    ].drop_duplicates(subset=["Image"]).head(MAX_IMAGES)

    for image_path in subset["Image"]:
        relative_path = str(image_path).replace("\\", "/")
        filename = Path(relative_path).name

        destination = OUTPUT / class_name / filename
        url = ROOT_URL + relative_path

        tasks.append((url, destination))
        class_records.append({
            "class_name": class_name,
            "medicine_name": name,
            "ndc11": ndc,
            "part": part,
            "image_path": relative_path,
            "local_path": str(destination)
        })

print(f"\nImages queued: {len(tasks)}")
print("Starting downloads...\n")

results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(download_image, task) for task in tasks]

    for i, future in enumerate(as_completed(futures), 1):
        results.append(future.result())

        if i % 25 == 0 or i == len(tasks):
            print(f"Progress: {i}/{len(tasks)}")

# Keep only successful downloads
successful_paths = {
    path for path, status in results
    if status in ("DOWNLOADED", "EXISTS")
}

manifest = pd.DataFrame(class_records)
manifest = manifest[
    manifest["local_path"].isin(successful_paths)
]
manifest.to_csv("dataset/processed/metadata.csv", index=False)

pd.DataFrame(class_map).to_csv(
    "dataset/processed/class_mapping.csv", index=False
)

print("\n========== DOWNLOAD SUMMARY ==========")
print("Total requested:", len(tasks))
print("Successful:", len(successful_paths))
print("Failed:", len(tasks) - len(successful_paths))
print("\nImages per class:")
print(manifest.groupby("class_name").size().to_string())

print("\nSaved metadata.csv and class_mapping.csv")
