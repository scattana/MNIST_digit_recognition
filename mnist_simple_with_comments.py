# -------------------------
# LAST UPDATED:
# 21 June 2020 04:26 PM CDT
# Seth Cattanach
# -------------------------
# NOTES:
# Format changes and fork
# testing for AcademyNEXT
# -------------------------

# -------------------------
# PACKAGE IMPORTS
# -------------------------
# Importing the required Keras modules containing model and layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------
# DATA COLLECTION
# -------------------------
# Load data and split into train/test sets
# using the MNIST Digit dataset (black & white)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# -------------------------
# DATA PRE-PROCESSING
# -------------------------
# Reshaping the array to
# 4-dims (match Keras API)
num_pixel_height = 28
num_pixel_width = 28
num_color_channels = 1
input_shape = (num_pixel_height, num_pixel_width, num_color_channels)
x_train = x_train.reshape(x_train.shape[0], num_pixel_height, num_pixel_width, num_color_channels)
x_test = x_test.reshape(x_test.shape[0], num_pixel_height, num_pixel_width, num_color_channels)


# -------------------------
# DATA PRE-PROCESSING
# -------------------------
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# -------------------------
# DATA PRE-PROCESSING
# -------------------------
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train = x_train / 255
x_test = x_test / 255


# -------------------------
# MODEL CREATION
# -------------------------
# Create a Sequential model and add neural network layers
# NOTE: final Dense layer MUST have 10 neurons b/c 10 classes (digits 0-9)
model = keras.Sequential()
model.add(layers.Conv2D(28, kernel_size=(5,5), input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(layers.Dense(128, activation=tf.nn.relu))
model.add(layers.Dense(10,activation=tf.nn.softmax))


# -------------------------
# MODEL COMPILATION
# -------------------------
# Compile the model
#       NOTE: "adam" optimizer (gradient descent - 1st & 2nd order)
#       "loss" specifies which loss metric(s) we'd like to use
#       ...[sparse_]cat_crossentropy = rough measure of std dev
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# -------------------------
# MODEL FITTING/TRAINING
# -------------------------
# Fit the training data to our compiled model
model.fit(x=x_train,y=y_train, epochs=10)


# -------------------------
# MODEL EVALUATION
# -------------------------
# Evaluate our model using default metric (accuracy)
#       available metrics:
#       ..."TopK" TopKCategoricalAccuracy(k=N)
#       ...other options to discuss: [Root]MeanSquaredError
#       ...kullback_leibler_divergence ("Relative Entropy")
results = model.evaluate(x_test, y_test)
print(dict(zip(model.metrics_names, results)))


