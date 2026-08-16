import pandas as pd

CSV_PATH = "../data/raw/bitext/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

df = pd.read_csv(CSV_PATH)

# -----------------------------------------
# Basic response statistics
# -----------------------------------------

print("===== RESPONSE LENGTH =====")

df["response_length"] = df["response"].str.len()

print(df["response_length"].describe())


# -----------------------------------------
# Number of unique responses
# -----------------------------------------

print("\n===== UNIQUE RESPONSES =====")

print(
    "Total responses:",
    len(df)
)

print(
    "Unique responses:",
    df["response"].nunique()
)

print(
    "Duplicate responses:",
    len(df) - df["response"].nunique()
)


# -----------------------------------------
# Unique responses per intent
# -----------------------------------------

print("\n===== UNIQUE RESPONSES PER INTENT =====")

intent_stats = (
    df.groupby("intent")["response"]
    .nunique()
    .sort_values()
)

print(intent_stats)


# -----------------------------------------
# Show examples from each intent
# -----------------------------------------

print("\n===== SAMPLE DATA PER INTENT =====")

for intent in df["intent"].unique():

    print("\n" + "=" * 70)
    print("INTENT:", intent)
    print("=" * 70)

    sample = df[df["intent"] == intent].head(3)

    for _, row in sample.iterrows():

        print("\nQuestion:")
        print(row["instruction"])

        print("\nResponse:")
        print(row["response"])