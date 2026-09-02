import pandas as pd
from src.utils.config import config

csv_path = config.BITEXT_CSV_PATH

df = pd.read_csv(csv_path)

extra_examples = [
    ("I was charged twice for the same order", "REFUND"),
    ("I got billed twice for one purchase", "REFUND"),
    ("There are two identical charges on my card", "REFUND"),
    ("I see a duplicate transaction on my account", "REFUND"),
    ("Why was I charged two times for the same order?", "REFUND"),
    ("My card was charged twice for one purchase", "REFUND"),
    ("I was billed twice this month for the same thing", "REFUND"),
    ("I have a duplicate charge on my bank statement", "REFUND"),
    ("The same payment appears twice on my statement", "REFUND"),
    ("I paid once but my card was charged twice", "REFUND"),
    ("There is a duplicate payment on my account", "REFUND"),
    ("I was charged two times for a single transaction", "REFUND"),
    ("My credit card shows the same charge twice", "REFUND"),
    ("I made one purchase but see two charges", "REFUND"),
    ("Why did I get charged twice for my purchase?", "REFUND"),
    ("I have two charges for the same order", "REFUND"),
    ("The payment was duplicated and I want one charge refunded", "REFUND"),
    ("I was charged twice for the same item", "REFUND"),
    ("My account shows a duplicate card transaction", "REFUND"),
    ("Please refund the extra charge, I was billed twice", "REFUND"),
]

extra_df = pd.DataFrame(
    extra_examples,
    columns=["instruction", "category"]
)

# Preserve the original dataset exactly.
augmented_df = pd.concat(
    [df, extra_df],
    ignore_index=True
)

output_path = "data/raw/bitext/Bitext_augmented.csv"

augmented_df.to_csv(
    output_path,
    index=False
)

print(f"Original rows:  {len(df)}")
print(f"Added rows:     {len(extra_df)}")
print(f"New total rows: {len(augmented_df)}")
print(f"Saved to:       {output_path}")
