# coding: utf-8

# ----------------------------------------------------------------------------

# Disciplina Tópicos Especiais II (C318) 

# Curso: Fundamentos de Machine Learning 

# Professor: Ricardo Augusto

# ----------------------------------------------------------------------------

# Projeto de Machine Learning

# Tema: Predição de despesas médicas por meio de Regressão Linear

# Grupo: Emanuel Massafera Magalhães, Pedro Henrique de Almeida e Thiago Santos da Costa

#%% Formulação e Definição do Problema de ML

# Frame the problem and look at the big picture.

# i) Enquadramento do problema de ML - Aprendizagem Supervisionada

# ii) Trata-se de um problema de regressão (variável de saída: despesas médicas de pacientes dos Estados Unidos)

# iii) Regressão Múltipla: uma vez que o sistema de ML precisa lidar com várias características (features) para gerar uma predição

# iv) Regressão Univariada: uma vez que estamos gerando a predição de um único valor para a variável de saída (um valor por paciente)

# v)  Se estamos fazendo a predição de múltiplos valores para a variável de saída temos uma regressão multivariada

# vi) Não há fluxo contínuo de dados entrando no sistema de ML a ser desenvolvido - com isso não há necessidade de ajuste-rápido aos dados (online learning))

# vii) A quantidade de dados pode ser acomodada na memória (batch learning)

#%% Estrutura do Projeto de ML

# - Estrutura do projeto   
#   - Importação de bibiliotecas utilizadas
#   - Importação da base de dados
#   - Manipulações iniciais nos dados
#   - Criação de conjuntos de dados de treino e teste
#   - Investigando Correlações
#   - Preparação dos dados para Modelagem 
#   - Limpeza dos dados
#   - Manipulando features categóricas
#   - Feature Scaling - Pipeline de transformação
#   - Criando o modelo de regressão linear
#   - Avaliação de Desempenho - Evaluation
#   - Validação Cruzada
#   - Usando o modelo nos dados de teste

#%% Bibliotecas utilizadas no projeto

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Importação da base de dados

# Importando dados (arquivo .csv) a partir da URL (fonte)
url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
df  = pd.read_csv(url)

# Salvando no diretório do projeto
df.to_csv('insurance.csv')

# Informações sobre o dataframe (atributo info)
df.info()

# ---------------------------
# Informações sobre o dataset

# 1. age: Número inteiro que indica a idade do beneficiário principal (excluindo acima de 64 anos, uma vez que são geralmente cobertos pelo governo).

# 2. sex: Gênero do segurado, podendo ser homem ou mulher.

# 3. bmi: Numero que indica o Índice de Massa Corporal (IMC) de segurado. É uma medida internacional utilizada para determinar se uma pessoa está no peso ideal. O IMC é calculado como sendo peso (em quilogramas) dividido pela altura (em metros) ao quadrado. Um IMC ideal está dentro do faixa de 18,5 a 24,9.

# 4. children: Número inteiro que indica o número de filhos ou dependentes cobertos pelo plano de seguro do segurado.

# 5. smoker: Indica se o segurado regularmente fuma tabaco ou não.

# 6. region: Indica o local de residência do beneficiário nos EUA, dividido em quatro regiões geográficas: nordeste, sudeste, sudoeste ou noroeste.

# 7. charges: Variável dependente que representa o valor de despesas médicas.

#%% Manipulações iniciais nos dados

# Descrição estatística do dataframe Pandas
df_stats = df.describe()
df_stats

# Extração de informações de estatística descritiva do dataframe:
# Ex: 25% dos pacientes tem idade (mediana) até 27 anos
# Ex: 75% dos pacientes tem idade (mediana) até 51 anos
# Ex: A idade média dos pacientes é de 39 anos
# Ex: O valor médio do bmi é de 30.66 kg/m²
# Ex: O valor médio dos gastos médicos medianos é de 13270 dólares

# ----------------------------------------------------------------------------
# Acessando e Manipulando o dataframe

# Indexação por rótulo - selecionando uma série Pandas a partir do dataframe
df['age']

# Indexação por rótulo - selecionando mais de uma série Pandas a partir do dataframe
df[['age', 'bmi']]

# Usando a notação ponto (.) para acesso à series de um dataframe
df.age

# Verificando a variável categórica - sex
gêneros = df['sex'].value_counts()
gêneros

# Verificando a variável categórica - smoker
fumantes = df['smoker'].value_counts()
fumantes

# Verificando a variável categórica - smoker
regiões = df['region'].value_counts()
regiões

# Visualizando histogramas relacionados com todas as variáveis do dataframe
df.hist(bins=50)
plt.show()

# Visualizando histogramas das variáveis - características - features
df['age'].hist(bins = 50)
plt.xlabel('age')
plt.ylabel('count')

df['bmi'].hist(bins = 50)
plt.xlabel('bmi')
plt.ylabel('count')

df['children'].hist(bins = 50)
plt.xlabel('children')
plt.ylabel('count')

df['charges'].hist(bins=50)
plt.xlabel('charges')
plt.ylabel('count')

# Biblioteca para análise de dados (visualizações estatísticas)
import klib
klib.dist_plot(df['age'])
klib.dist_plot(df['charges'])
klib.dist_plot(df['bmi'])
klib.dist_plot(df['children'])

# ----------------------------------------------------------------------------
# Análise conduzida a partir dos histogramas 

# i) A grande maioria dos indivíduos em nossos dados tem despesas médicas 
# anuais entre zero e US $ 15.000, embora a cauda da distribuição se estenda
# muito além desses picos. Como a regressão linear assume uma distribuição 
# normal para a variável dependente, essa distribuição não é ideal.

# ii) Outro problema em mãos é que os modelos de regressão exigem que todos
# os recursos sejam numéricos, mas temos três features (sex, region e smoker) 
# que não são numéricas.

# ----------------------------------------------------------------------------

#%% Criação de conjuntos de dados de treino e teste 

# Método de Amostragem Aleatória Simples (Sklearn)
from sklearn.model_selection import train_test_split

# Função do scikit-learn train_test_split
train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

#%% Investigando Correlações

# Criando uma cópia do dataset de treino
insurance = train_set.copy()

# Fazendo a estimativa da matriz de correlação das variáveis do dataframe
correlation_matrix = insurance.corr()
correlation_matrix

klib.corr_plot(insurance)
klib.corr_plot(insurance, target='charges')

# Gráfico usando plt.scatter
x = insurance['age']
y = insurance['charges']
plt.scatter(x,y, alpha = 0.05)
plt.xlabel('age')
plt.ylabel('charges')
plt.title('ScatterPlot - age vs charges')
plt.grid()

x = insurance['bmi']
y = insurance['charges']
plt.scatter(x,y, alpha = 0.05)
plt.xlabel('bmi')
plt.ylabel('charges')
plt.title('ScatterPlot - bmi vs charges')
plt.grid()

# Checando qual é o valor do coeficiente de correlação (Person)
# computado entre cada feature (variável) e a característica/variável charges
correlation_matrix['charges']

# Colocando as correlações em ordem ascendente
correlation_matrix['charges'].sort_values(ascending = False)

# Função scatter_matrix (Pandas): forma alternativa de checagem da matriz de correlação 
from pandas.plotting import scatter_matrix

# Selecionando atributos (features) de interesse para uso da scatter_matrix
attributes = ['charges', 
              'age', 
              'bmi',
              'children']
# Função scatter_matrix: plot na forma de matriz contendo gráficos de dispersão
scatter_matrix(insurance[attributes], figsize=(12, 8))

# Análise: aparentemente, a feature age tem relação estatística
# significativa (alta correlação) com os gastos médicos

#%% Preparação dos Dados para Modelagem

# Criando uma cópia do dataset de treino e removendo a variável target charges
insurance = train_set.drop('charges', axis = 1)

# Criando uma série para a variável de saída (target) 
insurance_labels = train_set['charges'].copy()

#%% Limpeza dos Dados (Data Cleaning)

klib.missingval_plot(insurance)

# Não há valores faltantes no dataset

#%% Manipulando features categóricas

# Checando informações do dataset
insurance.info()

insurance_num = insurance.drop(['sex', 'smoker', 'region'], axis=1)
insurance_cat = insurance[['sex', 'smoker', 'region']]
insurance_cat.head(10)

# Aplicando OneHotEncoder para fazer a codificação (encoding) das
# variáveis sex, smoker e region
from sklearn.preprocessing import OneHotEncoder

# Criando encoder
cat_encoder = OneHotEncoder()

# Fazendo o fit (estimação de parâmetros) e transform (transformando os dados)
insurance_cat_1hot = cat_encoder.fit_transform(insurance_cat)
insurance_cat_1hot

# A saída é uma matriz esparsa (sparse matrix) - sparse matrix only stores 
# the location of the nonzero elements
# Podemos converter para um array NumPy
insurance_cat_1hot.toarray()

# Lista de atributos das categorias
cat_encoder.categories_

#%% Feature Scaling - Pipeline de transformação

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
    ])

from sklearn.compose import ColumnTransformer
num_attribs = list(insurance_num)
cat_attribs = ['sex', 'smoker', 'region']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs),
    ])

insurance_prepared = full_pipeline.fit_transform(insurance)

#%% Modelagem

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(insurance_prepared, insurance_labels)

#%% Avaliação de Desempenho - Evaluation

# ----------------------------------------------------------------------------
# Avaliação de desempenho com todos os dados (sem separação de treino e teste)

from sklearn.metrics import mean_squared_error
insurance_predictions = lin_reg.predict(insurance_prepared)
lin_mse = mean_squared_error(insurance_labels, insurance_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# ----------------------------------------------------------------------------
# Avaliação de desempenho com separação de treino e teste
from sklearn.model_selection import train_test_split

# Matriz de Features X
X = insurance_prepared

# Variáveç Target
y = insurance_labels

# Divisão de treino e teste (sklearn) - train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=42)

lin_mse = mean_squared_error(insurance_labels, insurance_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

#%% Validação Cruzada

from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())
    
lin_scores = cross_val_score(lin_reg, insurance_prepared, insurance_labels,
scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

r2_train = lin_reg.score(insurance_prepared, insurance_labels)
print('R2 no set de treino: %.2f' % r2_train)

#%% Usando o modelo nos dados de teste

insurance_test = test_set.drop('charges', axis = 1)
insurance_test_labels = test_set['charges'].copy()

insurance_test_prepared = full_pipeline.fit_transform(insurance_test)

lin_scores = cross_val_score(lin_reg, insurance_test_prepared, insurance_test_labels,
scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

r2_test = lin_reg.score(insurance_test_prepared, insurance_test_labels)
print('R2 no set de teste: %.2f' % r2_test)
