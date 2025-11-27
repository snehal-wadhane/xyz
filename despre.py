from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
df=pd.read_csv('Datasets/Lipstick.csv',index_col='Id')
encoder = {}
cat = df.select_dtypes(include='object')

for col in cat.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoder[col] = le  

x = df.drop('Buys', axis=1)
y = df['Buys']

d = DecisionTreeClassifier(criterion='entropy')
d.fit(x, y)

data = pd.DataFrame({
    'Age': ['<21'],        
    'Income': ['Low'],     
    'Gender': ['Female'],  
    'Ms': ['Married']      
})
le.inverse_transform
for col in data.columns:
    data[col] = encoder[col].transform(data[col])

pred = d.predict(data)[0]              # prediction → 0 or 1
p = encoder['Buys'].inverse_transform([pred])[0]
print(p)

# if p == 1:
#     print("Buys")
# else:
#     print("Does NOT Buy")
