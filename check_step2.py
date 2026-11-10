import pandas as pd
from app.preprocessing import validate_data, split_data

df = pd.read_csv("data/creditcard.csv")
validate_data(df)
print("Données valides.")

train, val, test = split_data(df)
for name, part in [("train", train), ("validation", val), ("test", test)]:
    print(name, len(part), "lignes,", int(part["Class"].sum()), "fraudes")