# HDO-predi-o-Seletividade-2NH
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Carregar dados
df = pd.read_csv("data.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Normalização
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Modelo
model = Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Avaliação
loss = model.evaluate(X_test, y_test)
print("Erro:", loss)
