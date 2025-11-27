df['price_clean'] = (
    df['price']
    .str.replace('TL', '')
    .str.replace(',', '')
    .str.replace('arrow_downward%\d', '',regex=True)
    .astype('float64')
)

def fix_encoding(text):
    try:
        return text.encode("latin1", errors="ignore").decode("utf8", errors="ignore")
    except:
        return text

df["address"] = df["address"].astype(str).apply(fix_encoding)


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
num=df['age']
num_df_imputed = imputer.fit_transform(df[['age']])
df['age']=num_df_imputed

import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression

df = sns.load_dataset("titanic")

# Split data
train = df[df['age'].notnull()]
test = df[df['age'].isnull()]

# Features to use for prediction
features = ['fare', 'pclass', 'sibsp', 'parch']

# Train regression
model = LinearRegression()
model.fit(train[features], train['age'])

# Predict missing ages
df.loc[df['age'].isnull(), 'age'] = model.predict(test[features])


import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression

df = sns.load_dataset("titanic")

# Split data
train = df[df['age'].notnull()]
test = df[df['age'].isnull()]

# Features to use for prediction
features = ['fare', 'pclass', 'sibsp', 'parch']

# Train regression
model = LinearRegression()
model.fit(train[features], train['age'])

# Predict missing ages
df.loc[df['age'].isnull(), 'age'] = model.predict(test[features])


df = df.drop_duplicates()

df = df.rename(columns={'sibsp': 'siblings_spouses'})

# Step 8: Handle inconsistent values
df['sex'] = df['sex'].replace({'male':'male', 'female':'female'})  # example

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
le=LabelEncoder()
df['sex']=le.fit_transform(df['sex'])

scaler = MinMaxScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

# Step 14: Standardization
std = StandardScaler()
df[['age', 'fare']] = std.fit_transform(df[['age', 'fare']])

from sklearn.linear_model import LinearRegression
import pandas as pd
df = sns.load_dataset("titanic")
df['age_group'] = pd.cut(df['age'],
                         bins=[0, 18, 40, 100],
                         labels=['Low', 'Medium', 'High'])

pd.cut(df['age'], bins=3)
pd.qcut(df['age'], q=4)
