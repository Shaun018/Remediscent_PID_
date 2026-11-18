
import pandas as pd

file = "dataset/raw/directory_consumer_grade_images.xlsx"
df = pd.read_excel(file)

print("\nIMAGE LAYOUT DISTRIBUTION")
print(df["Layout"].value_counts().to_string())

# Select consumer-style images only
consumer = df[
    df["Layout"].astype(str).str.strip().str.upper() == "C3PI_TEST"
].copy()

# Count images per distinct product and pill part
summary = (
    consumer.groupby(["NDC11", "Part", "Name"])
    .agg(image_count=("Image", "nunique"))
    .reset_index()
    .sort_values("image_count", ascending=False)
)

print("\nTOTAL CONSUMER IMAGES:", len(consumer))
print("DISTINCT PRODUCT/PART CLASSES:", len(summary))

print("\nTOP 30 CONSUMER CLASSES")
print(summary.head(30).to_string(index=False))

summary.to_csv(
    "dataset/processed/consumer_class_candidates.csv",
    index=False
)

print("\nSaved consumer_class_candidates.csv")
