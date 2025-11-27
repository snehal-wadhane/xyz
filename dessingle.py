import pandas as pd
import math

data = {
    "Age": ["Young","Young","Middle","Old","Old","Old","Middle","Young","Young","Old",
            "Young","Middle","Middle","Old"],
    "Income": ["High","High","High","Medium","Low","Low","Low","Medium","Low","Medium",
               "Medium","Medium","High","Medium"],
    "Married": ["No","No","No","No","Yes","Yes","Yes","No","Yes","Yes",
                "Yes","No","Yes","No"],
    "Health": ["Fair","Good","Fair","Fair","Fair","Good","Good","Fair","Fair","Good",
               "Good","Good","Fair","Good"],
    "Class": ["No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes",
              "Yes","Yes","Yes","No"]
}

df = pd.DataFrame(data)
print(df)

freq_age = df["Age"].value_counts()
print("Frequency Table for Age:")
print(freq_age)

def entropy(series):
    counts = series.value_counts()
    total = len(series)
    ent = 0
    for c in counts:
        p = c / total
        ent -= p * math.log2(p)
    return ent

overall_entropy = entropy(df["Class"])
print("Overall Entropy:", overall_entropy)

age_groups = df.groupby("Age")

entropy_age = {}
for age, group in age_groups:
    entropy_age[age] = entropy(group["Class"])

print("Entropy for each Age group:", entropy_age)

weighted_entropy = 0
total = len(df)

for age, group in age_groups:
    weighted_entropy += (len(group) / total) * entropy(group["Class"])

print("Weighted Entropy (Age Split):", weighted_entropy)

info_gain_age = overall_entropy - weighted_entropy
print("Information Gain (Age):", info_gain_age)
