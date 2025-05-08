import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt

N = 100
K = 10
M = 40000 
d = 10 #dimension
dt = 0.005
T = M*dt
Nbatch = 1
validation_split_ratio = 0.2
EPOCHS = np.int64(M/(N*(1-validation_split_ratio)/Nbatch))
x_training = np.random.uniform(-4,4,size =(N,d))
x_test = np.random.uniform(-4,4,size =(100,d))

def f(x):
  d = len(x)
  ones = 1/2*np.ones((d,))
  norm = np.linalg.norm(x - ones)
  return norm
y_training = np.zeros(N)
y_test = np.zeros(N)
for i in range(N):
  y_training[i] = f(x_training[i])
  y_test[i] = f(x_test[i])
Input_layer = tf.keras.Input(shape =(d,))
Hidden_layer = keras.layers.Dense(units=K, 
                                  activation ="sigmoid", 
                                  use_bias =True, 
                                  kernel_initializer ='random_normal', 
                                  bias_initializer ='zeros', 
                                  name= "hidden_layer_1")

Output_layer = keras.layers.Dense(units=1,
                                  use_bias =False, 
                                 kernel_initializer='random_normal',
                                  name="output_layer")

model = keras.Sequential([Input_layer, Hidden_layer, Output_layer])

optimizer = tf.keras.optimizers.SGD(learning_rate=dt / d)

model.compile(optimizer = optimizer,loss='mean_squared_error')

history = model.fit(x=x_training, 
                    y=y_training, 
                    batch_size=Nbatch, 
                    epochs = EPOCHS, 
                    validation_split = validation_split_ratio, 
                    verbose =1)

model.evaluate (x=x_test, y=y_test, verbose=1)

pts = np.meshgrid(*[np.linspace(-4, 4, 300) for _ in range(d)]) # generate meshgrid of points
pts = np.column_stack([pts[i].ravel() for i in range(d)]) # stack meshgrid points into a single (300^d, d) array

target_fcn_vals = f(pts)
alpha_vals = model(pts)

