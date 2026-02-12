import numpy as np
import matplotlib.pyplot as plt
import os
# Remove TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Force TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf


filename = "test_ssh"

# Download data
(X_train, t_train), (X_test, t_test) = tf.keras.datasets.mnist.load_data()

# Normalize inputs
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# One-hot output vectors
T_train = tf.keras.utils.to_categorical(t_train, 10)
T_test = tf.keras.utils.to_categorical(t_test, 10)

def create_cnn():
    
    inputs = tf.keras.Input(shape = (28, 28, 1))
    x = tf.keras.layers.Conv2D(16, kernel_size=(5,5),activation='relu',padding = 'same',use_bias=False)(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64,kernel_size=(5,5),activation='relu',padding = 'same',use_bias=False)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64,kernel_size=(5,5),activation='relu',padding = 'same',use_bias=False)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10,activation='softmax',use_bias=False)(x)

    # Create functional model
    model= tf.keras.Model(inputs, x)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # Loss function
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    return model

# Create model
model = create_cnn()

# Train model
history = model.fit(X_train, T_train, batch_size=128, epochs=5, validation_split=0.1,  )

model.save(f"models/{filename}_model.keras")

# Test model
predictions_keras = model.predict(X_test, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, T_test, verbose=0)
print(f"Test accuracy: {test_accuracy}")