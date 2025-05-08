import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt

N = 100
K = 10
dt = 0.005
M = 40000
T = M*dt
Nbatch = 1
validation_split_ratio = 0.2
EPOCHS = np.int64(M / (N * (1 - validation_split_ratio) / Nbatch))

"""
np.random.seed(123)
tf.random.set_seed(124)
"""
x_training = np.random.uniform(-4, 4, size=(N, 1)) 
x_test = np.random.uniform(-4, 4, size=(100, 1))

def f(x):
    return np.square(np.abs(x - 0.5))

y_training = f(x_training) 
y_test = f(x_test)

# - - - - - - - - - -Neural Network Layers - - - - - - - - - - 
Input_layer = tf.keras.Input(shape=(1,)) 
Hidden_layer = keras.layers.Dense(units=K,
                                  activation='sigmoid', 
                                  use_bias=True, 
                                  kernel_initializer='random_normal', 
                                  bias_initializer='zeros', 
                                  name="hidden_layer_1")
Output_layer = keras.layers.Dense(units=1, 
                                  use_bias=False, 
                                  name="output_layer")

model = keras.Sequential([Input_layer, Hidden_layer, Output_layer])
optimizer = tf.keras.optimizers.SGD(learning_rate=dt) 
model.compile(optimizer=optimizer, loss='mean_squared_error')
history = model.fit(x=x_training , y=y_training , batch_size=Nbatch, epochs=EPOCHS, validation_split=validation_split_ratio , verbose =1)


model.evaluate(x=x_test, y=y_test, verbose=1)
pts = np.linspace(-4, 4, 300).reshape(-1, 1) 
target_fcn_vals = f(pts)
alpha_vals = model(pts)

# figure 1
plt.figure('Learned function', figsize=(15, 10)) 
plt.plot(x_training, y_training, 'o', label='Training data') 
plt.plot(pts, target_fcn_vals, label='Target function') 
plt.plot(pts, alpha_vals, label='alpha')
plt.legend()
plt.show()

# figure 2
plt.figure('Error', figsize=(15, 10)) 
plt.semilogy(history.history['loss'], label='Training error')
plt.semilogy(history.history['val_loss'], label='Validation error') 
plt.xlabel('Epoch')
plt.ylabel('Mean squared error')
plt.legend()
plt.grid(True)
plt.show()
 