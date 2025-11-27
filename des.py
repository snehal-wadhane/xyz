import math
def entropy(column):
    values = column.value_counts()
    total = len(column)

    ent = 0
    for count in values:
        p = count / total
        ent -= p * math.log2(p)
    return ent
def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values = df[attribute].unique()

    weighted_entropy = 0
    for v in values:
        subset = df[df[attribute] == v]
        weighted_entropy += (len(subset)/len(df)) * entropy(subset[target])

    gain = total_entropy - weighted_entropy
    return gain
target = "Buys"     # target column name

for col in df.columns:
    if col != target:
        print(col, " → Information Gain = ", information_gain(df, col, target))

gains = {}
for col in df.columns:
    if col != target:
        gains[col] = information_gain(df, col, target)

root_node = max(gains, key=gains.get)
print("\nROOT NODE OF DECISION TREE =", root_node)
