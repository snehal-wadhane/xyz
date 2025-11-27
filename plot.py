import matplotlib.pyplot as plt
num=df.select_dtypes(include=['int64','float64'])
for col in num:
    plt.hist(num[col].dropna())
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()



#11
import matplotlib.pyplot as plt
num=df.select_dtypes(include='number')
for col in num:
    plt.hist(num[col].dropna(),color='red',edgecolor='black',orientation='vertical',alpha=0.7,bins=5,histtype='stepfilled')
    plt.title(f"histogram of {col}")
    plt.xlabel("count")
    plt.ylabel(f"{col}")
    plt.show()


plt.hist([df['sr'], df['NumberFloorsofBuilding']], bins=20, stacked=True)
plt.title("Stacked Histogram")
plt.legend(['sr', 'Floors'])
plt.show()


import matplotlib.pyplot as plt

num = df.select_dtypes('number')

plt.figure(figsize=(12, 8))

for i, col in enumerate(num.columns, 1):
    plt.subplot(2, 3, i)   
    plt.boxplot(num[col])
    plt.title(f'Box plot of {col}')
    
plt.show()

#15
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')
print("Titanic Dataset Loaded:\n")
print(titanic.head())

# a) Countplot: Number of passengers survived vs not survived
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='survived')
plt.title("Survival Count (0=No, 1=Yes)")
plt.show()

# b) Countplot: Survival by Sex
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='survived', hue='sex')
plt.title("Survival by Sex")
plt.show()

# c) Countplot: Survival by Class
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='survived', hue='class')
plt.title("Survival by Passenger Class")
plt.show()

# d) Boxplot: Age distribution by Survival
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='survived', y='age')
plt.title("Age Distribution by Survival")
plt.show()

# e) Heatmap: Correlation of numeric features
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Titanic Features")
plt.show()

#curve distribution
sns.histplot(titanic['age'], kde=True)
plt.title("Age Distribution of Passengers")
plt.show()

#16
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')
print("Titanic Dataset Loaded:\n")
print(titanic.head())
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
titanic = sns.load_dataset('titanic')

# Plot histogram of fare
plt.hist(titanic['fare'].dropna(), bins=30)
plt.title("Distribution of Ticket Fare")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()


##confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion Matrix Values
TP = 1
FP = 1
FN = 8
TN = 90

# Create confusion matrix array
conf_matrix = np.array([[TP, FP],
                        [FN, TN]])

# Labels
labels = ['Positive', 'Negative']

# Plot the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = 1 - accuracy
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Display metrics
print(f"Accuracy: {accuracy:.3f}")
print(f"Error Rate: {error_rate:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1_score:.3f}")
