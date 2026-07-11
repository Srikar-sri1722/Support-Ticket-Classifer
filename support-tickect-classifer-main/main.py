import pandas as pd
import numpy as np

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("support_tickets_6k.csv")

print("Original Shape:", df.shape)

# =========================================================
# KEEP ONLY ENGLISH ROWS
# =========================================================

if "language" in df.columns:
    df = df[df["language"] == "en"].copy()

# =========================================================
# CREATE CATEGORY COLUMN
# =========================================================

def map_category(row):

    queue = str(row.get("queue", "")).lower()

    tags = " ".join([
        str(row.get("tag_1", "")),
        str(row.get("tag_2", "")),
        str(row.get("tag_3", "")),
        str(row.get("tag_4", "")),
        str(row.get("tag_5", "")),
        str(row.get("tag_6", ""))
    ]).lower()

    combined = queue + " " + tags

    # ACCOUNT ACCESS
    if any(word in combined for word in [
        "account",
        "login",
        "password",
        "authentication",
        "access",
        "mfa"
    ]):
        return "Account Access"

    # BILLING
    elif any(word in combined for word in [
        "billing",
        "invoice",
        "payment",
        "refund",
        "returns",
        "exchange",
        "subscription"
    ]):
        return "Billing"

    # FEATURE REQUEST
    elif any(word in combined for word in [
        "feature",
        "enhancement",
        "integration",
        "request",
        "product"
    ]):
        return "Feature Request"

    # DEFAULT
    else:
        return "Technical Support"

# Apply category mapping
df["category"] = df.apply(map_category, axis=1)

# =========================================================
# CREATE RANDOM URGENCY VALUES (1–3)
# =========================================================

np.random.seed(42)

df["urgency"] = np.random.randint(
    1,
    4,
    size=len(df)
)

# =========================================================
# CREATE RANDOM IMPACT VALUES (1–3)
# =========================================================

df["impact"] = np.random.randint(
    1,
    4,
    size=len(df)
)

# =========================================================
# KEEP ONLY REQUIRED COLUMNS
# =========================================================

df_final = df[[
    "body",
    "category",
    "urgency",
    "impact"
]]

# =========================================================
# CLEAN DATA
# =========================================================

df_final = df_final.dropna()

df_final = df_final.drop_duplicates()

# Remove very short tickets
df_final = df_final[
    df_final["body"].str.len() > 50
]

# Reset index
df_final = df_final.reset_index(drop=True)

# =========================================================
# CHECK RESULTS
# =========================================================

print("\nFinal Shape:", df_final.shape)

print("\nCategory Distribution:")
print(df_final["category"].value_counts())

print("\nUrgency Distribution:")
print(df_final["urgency"].value_counts())

print("\nImpact Distribution:")
print(df_final["impact"].value_counts())

# =========================================================
# SAVE FINAL DATASET
# =========================================================

df_final.to_csv(
    "support_tickets_final.csv",
    index=False
)

print("\nSaved:")
print("support_tickets_final.csv")

# =========================================================
# SHOW SAMPLE ROWS
# =========================================================

print("\nSample Rows:\n")

print(df_final.head())