#prac1
survived_passengers = df[df['Survived'] == 1]
sorted_by_age = df.sort_values(by="Age")
d = df.loc[:3].query("Age == 22")
df.loc[(df['Pclass'] == 3) & (df.index.isin(range(893, 896)))]
df.sort_values(by=['Pclass','PassengerId'],ascending=[False,True],kind='heapsort',na_position='first',ignore_index=True)

#prac2
stats = pd.DataFrame({
    'min'    : num.min(),
    'max'    : num.max(),
    'var'    : num.var(),
    'std'    : num.std(),
    'q1'     : num.quantile(0.25),
    'median' : num.quantile(0.50)
})
num.quantile([0.25, 0.50, 0.75])
print(df.var(numeric_only=True))

num = df.select_dtypes(include=['number'])
print(num.max() - num.min())


#13
import pandas as pd

df = pd.read_csv("Datasets/Covid Vaccine Statewise.csv")

# Select only the required columns
result = df[['State', 'First Dose Administered']]

# Group state-wise (if same state appears multiple times)
result = result.groupby('State')['First Dose Administered'].sum()

print(result)




n=num.groupby('State')['First Dose Administered'].sum().where(num['State']!='India')
n=num.groupby('State')['Second Dose Administered'].sum().where(num['State']!='India')
df[df['State'] == 'India']['Male (Doses Administered)'].sum()


cat=['district','catogory']
num=['NumberFloorsofBuilding','price_clean','clean_GrossSquareMeters']
summary=df.groupby(cat)[num].agg(['count','mean', 'median', 'min', 'max', 'std'])


species_list = ['Iris-setosa', 'Iris-versicolor']

df_filtered = df[df['species'].isin(species_list)]

df_filtered.groupby('species').agg(
    ['sum', 'count', 'std', 'var', 'mean', 'median',
     ('Q1', lambda x: x.quantile(0.25)),
     ('Q3', lambda x: x.quantile(0.75))]
)
