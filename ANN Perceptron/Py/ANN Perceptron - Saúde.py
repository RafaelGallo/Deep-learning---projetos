import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


dados = pd.read_csv("Dados/insurance.csv")

dados.columns = ["Idade", "Sexo", "Peso", "Criança", "Fumante", "Região","Preço"]


print("Média das idades", dados.Idade.mean())
print("Média das Preço", dados.Preço.mean())
print("Mediana das idades", dados.Idade.median())
print("Mediana das Preço", dados.Preço.median())
print("Mediana das idades", dados.Idade.mode())
print("Mediana das Preço", dados.Preço.mode())


train = dados.iloc[:, 0:8].values
test = dados.iloc[:, 1].values


dados.drop(["Sexo", "Região", "Fumante"], axis = 1, inplace = True)

X=dados.drop('Preço', axis=1)
y=dados['Preço']


from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 1, input_shape = [3]))
model.add(tf.keras.layers.Dense(units = 1, input_shape = [3]))
model.add(tf.keras.layers.Dense(units = 1, input_shape = [3]))
model.compile(optimizer = tf.keras.optimizers.Adam(0.1), loss = "mean_squared_error")
model_json = model.to_json()
model.summary()

model.get_weights()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_predict = model.predict(x_test)

RMSE = mean_squared_error(y_test, y_predict)
MSE = np.sqrt(mean_squared_error(y_test, y_predict))
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2) 


y_predict = model.predict(x_test)

plt.figure(figsize=(11,7))
plt.plot(y_test, y_predict, "o", color = 'b')
plt.xlabel("Model Predictions")
plt.ylabel("True Value (ground Truth)")
plt.title('Linear Regression Predictions')
plt.show()


y_predict = model.predict(x_test)

model_json = model.to_json()

with open('model_ANN.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_ANN.h5')