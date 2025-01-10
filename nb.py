# Montar o Google Drive para acessar a planilha
from google.colab import auth
from google.colab import drive
import gspread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ace_tools as tools  # Para exibir DataFrame no Google Colab

# Autenticar no Google Drive
auth.authenticate_user()
drive.mount('/content/drive')

# Conectar à Google Sheets
gc = gspread.authorize(auth)

# Abrir a planilha a partir da pasta raiz do Google Drive
SHEET_NAME = "Nome_da_Sua_Planilha"  # Substitua pelo nome correto da sua planilha
spreadsheet = gc.open(SHEET_NAME)
sheet = spreadsheet.sheet1

# Carregar os dados da planilha para um DataFrame Pandas
data = pd.DataFrame(sheet.get_all_records())

# Exibir os primeiros dados para verificação
print("Primeiras linhas dos dados carregados:")
print(data.head())

# **PASSO 1: Definir as variáveis preditoras (X) e a variável resposta (y)**
X = data[['Largura da Boca (cm)', 'Altura da Boca (cm)', 'Velocidade (km/h)']]
y = data['Força Gerada (N)']

# **PASSO 2: Dividir os dados em treino e teste (80% treino, 20% teste)**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **PASSO 3: Criar e treinar o modelo de regressão linear**
model = LinearRegression()
model.fit(X_train, y_train)

# **PASSO 4: Fazer previsões**
y_pred = model.predict(X_test)

# **PASSO 5: Avaliar o modelo**
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n🔍 **Métricas do Modelo:**")
print(f"📉 Erro Absoluto Médio (MAE): {mae:.2f}")
print(f"📊 Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")

# **PASSO 6: Visualizar os resultados**
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='black')
plt.xlabel("Força Real (N)")
plt.ylabel("Força Predita (N)")
plt.title("Comparação entre Força Real e Predita")
plt.grid(True)
plt.show()

# **PASSO 7: Exibir os resultados em formato de tabela**
predictions_df = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
tools.display_dataframe_to_user(name="Previsões de Força", dataframe=predictions_df)
