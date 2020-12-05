import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

dados = pd.read_csv("Dados/Celsius-to-Fahrenheit.csv")
dados.reset_index(drop = True, inplace = True)

dados.describe()

print("Média das idades", dados.Celsius.mean())
print("Média das Preço", dados.Fahrenheit.mean())
print("Mediana das idades", dados.Celsius.median())
print("Mediana das Preço", dados.Fahrenheit.median())


x = dados["Celsius"]
y = dados["Fahrenheit"]


from sklearn.model_selection import train_test_split

X_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x.shape
y.shape

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 1, input_shape = [1]))
model.add(tf.keras.layers.Dense(units = 1, input_shape = [2]))
model.add(tf.keras.layers.Dense(units = 1, input_shape = [3]))
model.compile(optimizer = tf.keras.optimizers.Adam(0.1), loss = "mean_squared_error")
model_json = model.to_json()
model.summary()

hist = model.fit(X_train, y_train, epochs = 1000)

hist.history.keys()

plt.plot(hist.history["loss"])
plt.title("Model Loss Progress During Training")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend(["Training Loss"])


model.get_weights()

pred = model.predict(x_test)
pred

temperatura_c = 25
temperatura_f = model.predict([temperatura_c])
temperatura_f

temperatura_f = 9/5 * temperatura_c + 20
temperatura_f


#Salvando o modelo
model_json = model.to_json()

with open('ANN Perceptron.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('ANN Perceptron.h5')