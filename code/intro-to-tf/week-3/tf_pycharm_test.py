import tensorflow as tf
import numpy as np

x_train = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y_train = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=([1]))
])

model.compile(optimizer='sgd', loss='mean_squared_error')

history = model.fit(x_train, y_train, epochs=500)

# Prediction
prediction = model.predict([7.0])
print("Prediction", prediction)
