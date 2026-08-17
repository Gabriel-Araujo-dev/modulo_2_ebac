import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

df = pd.read_csv('ecommerce_preparados.csv')
print(df.head())

# limpeza e tratamento de dados
print('Tipagem dos dados: \n', df.dtypes)
print('Quantidade de valores nulos:', df.isnull().sum())

df['Material'] = df['Material'].fillna('não informado')
df['Gênero'] = df['Gênero'].fillna('não informado')
df = df.fillna(0)
print('Quantidade de valores nulos:', df.isnull().sum())

df = df.drop(columns=['Review1', 'Review2', 'Review3'])
print(df.head())

df['Marca'] = df['Marca'].str.title()
df['Material'] = df['Material'].str.title()
df['Temporada'] = df['Temporada'].str.title()
df = df[df['Temporada'] != '2021']

# dicionário para padronizar os valores de temporada
mapeamento_temporadas = {
    'Não Definido': 'Indefinido',
    'Outono/Inverno': 'Outono-Inverno',
    'Outono-Inverno': 'Outono-Inverno',
    'Primavera/Verão': 'Primavera-Verão',
    'Primavera-Verão': 'Primavera-Verão',
    'Primavera/Verão/Outono/Inverno': 'Todas as Estações',
    'Primavera/Verão Outono/Inverno': 'Todas as Estações',
    'Primavera-Verão Outono-Inverno': 'Todas as Estações',
    'Primavera-Verão - Outono-Inverno': 'Todas as Estações'
}
df['Temporada'] = df['Temporada'].replace(mapeamento_temporadas)

# Gráfico de Histograma - Preço separado por Gênero
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Preço', hue='Gênero', multiple='layer', bins=30, alpha=0.6)
plt.title('Distribuição de Preço por Gênero')
plt.xlabel('Preço (R$)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# Gráfico de Dispersão - Preço vs Nota
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Preço', y='Nota')
plt.title('Preço vs Nota do Produto')
plt.xlabel('Preço (R$)')
plt.ylabel('Nota')
plt.tight_layout()
plt.show()

# Gráfico de Mapa de Calor - correlação entre variáveis numéricas
cols = ['Nota', 'N_Avaliações', 'Desconto', 'Preço', 'Qtd_Vendidos_Cod']
corr = df[cols].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Mapa de Calor de Correlações')
plt.tight_layout()
plt.show()

# Gráfico de Barras - Quantidade de produtos por Temporada
x = df['Temporada'].value_counts().index
y = df['Temporada'].value_counts().values
plt.figure(figsize=(10, 6))
plt.bar(x, y, color='#2E86AB')
plt.title('Quantidade de Produtos por Temporada')
plt.xlabel('Temporada')
plt.ylabel('Quantidade')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Gráfico de Pizza - Distribuição por Temporada
plt.figure(figsize=(10, 8))
plt.pie(y, labels=x, autopct='%.2f%%', startangle=90)
plt.title('Distribuição de Produtos por Temporada')
plt.tight_layout()
plt.show()

# Gráfico de Densidade - Nota dos produtos
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Nota'], fill=True, color='#863E9C')
plt.title('Densidade das Notas dos Produtos')
plt.xlabel('Nota')
plt.tight_layout()
plt.show()

# Gráfico de Regressão - Número de avaliações vs Quantidade vendida
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='N_Avaliações_MinMax', y='Qtd_Vendidos_Cod')
plt.title('Quantidade Vendida vs Número de Avaliações')
plt.xlabel('Número de avaliações (normalizado)')
plt.ylabel('Quantidade vendida (codificada)')
plt.tight_layout()
plt.show()
