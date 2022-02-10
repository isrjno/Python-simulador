#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


import panel as pn
import hvplot.pandas
pn.extension()


# In[3]:


dados = pd.read_csv('base_ff.csv')


# In[4]:


y = dados[['Clientes']]
X = dados.drop(['Clientes', 'Loja'], axis=1)


# In[5]:


X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)


# In[6]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# In[7]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


from sklearn.neighbors import KNeighborsRegressor


# In[10]:


import numpy as np
def cria_modelo(Modelo, variaveis):
    if Modelo == 'Regressão Linear':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
       
    if Modelo == "Árvore de Decisão":
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(max_depth=20)
    if Modelo == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(max_depth=20, n_estimators=25)
    if Modelo == "KNN":
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor()
    dados = X_teste.copy()
    dados = dados[variaveis]
    print('Executando: ', variaveis)
    model.fit(X_treino[variaveis], y_treino)
    
    dados['Clientes_Predito'] = model.predict(dados)
    dados['Clientes_Real'] = y_teste
    res = np.abs(dados['Clientes_Real'] - dados['Clientes_Predito'])
    metricas = pd.DataFrame({'Dados': ['Treino', 'Teste'], 
              'MAE' : [mean_absolute_error(y_treino, model.predict(X_treino[variaveis])), mean_absolute_error(y_teste, model.predict(X_teste[variaveis]))],
              'MAPE' : [mean_absolute_percentage_error(y_treino, model.predict(X_treino[variaveis])), mean_absolute_percentage_error(y_teste, model.predict(X_teste[variaveis]))],
              'RMSE' : [np.sqrt(mean_squared_error(y_treino, model.predict(X_treino[variaveis]))), np.sqrt(mean_squared_error(y_teste, model.predict(X_teste[variaveis])))],
              'R2' : [r2_score(y_treino, model.predict(X_treino[variaveis])), r2_score(y_teste, model.predict(X_teste[variaveis]))],
    }).set_index('Dados')
   
    
    return pn.Column(        
                  '### Desempenho do Modelo', metricas,
                  dados.hvplot.scatter(x='Clientes_Real', 
                  y='Clientes_Predito', 
                  title='Real vs Predito - Conjunto de teste',
                  xlabel='Clientes - Real',
                  ylabel='Clientes - Predito'), res.hvplot.kde(title='Distribuição do erro (valor absoluto)', color='red'))         


# In[11]:


multi_choice = pn.widgets.MultiChoice(name='Variáveis preditoras', value=['Fim de Semana ou Feriado', 'Media Historica', 'Investimento em MKT',
       'Chuva', 'Shopping'],
    options=['Fim de Semana ou Feriado', 'Media Historica', 'Investimento em MKT',
       'Chuva', 'Shopping'])


# In[12]:


kw = dict(Modelo=['Regressão Linear', 'Árvore de Decisão', 'Random Forest', 'KNN'], variaveis=multi_choice)
res = pn.interact(cria_modelo, **kw)


# In[13]:


res.pprint()


# In[14]:


dash = pn.Row(pn.Column('# Simulador - Emeritus', res[0]), '',  res[1])


# In[15]:


dash.servable(title='Simulador')

