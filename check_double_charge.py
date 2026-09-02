import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

from src.utils.config import config

df = pd.read_csv("data/raw/bitext/Bitext_augmented.csv")
df = df[["instruction", "category"]].dropna()

keywords = [
    "twice",
    "double",
    "duplicate",
    "two times",
    "2 times",
    "billed twice",
]

pattern = "|".join(keywords)

mask = df["instruction"].str.lower().str.contains(
    pattern,
    na=False
)

matches = df[mask]

print(
    f"Found {len(matches)} training examples "
    "matching double-charge phrasing\n"
)

print("Categories:")
print(matches["category"].value_counts())

print("\nSample examples:")

for _, row in matches.head(20).iterrows():
    print(
        f"  [{row['category']}] "
        f"{row['instruction']}"
    )
