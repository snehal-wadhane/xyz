survived_passengers = df[df['Survived'] == 1]
sorted_by_age = df.sort_values(by="Age")
d = df.loc[:3].query("Age == 22")

num = df.select_dtypes(include=['int64', 'float64'])
num.quantile([0.25, 0.50, 0.75])

print(df.var(numeric_only=True))

num = df.select_dtypes(include=['number'])
print(num.max() - num.min())

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
