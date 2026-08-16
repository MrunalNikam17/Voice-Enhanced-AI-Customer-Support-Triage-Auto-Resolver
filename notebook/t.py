import pandas as pd

# Change this to your actual CSV filename
CSV_PATH = "../data/raw/bitext/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

df = pd.read_csv(CSV_PATH)

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== CATEGORY COUNT =====")
print(df["category"].value_counts())

print("\n===== INTENT COUNT =====")
print(df["intent"].value_counts())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())