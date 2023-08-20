import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
'''
# dataset will handwritten digits
mnist = tf.keras.datasets.mnist

# split the dataset into train and test dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# scaling down the data like from 0-255 to 0-1 so it would be easily and faster
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
#not scaling down the y because these are labels from 0-9

# creating a simpel model
model = tf.keras.models.Sequential()
# adding layers to the model
#flatten layer for converting the grid matrix into 1D array
model.add(tf.keras.layers.Flatten())
#dense layer so the neurons are connected to each other
model.add(tf.keras.layers.Dense(units=128, activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation= tf.nn.relu))
#output layer
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
#compiling model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3)

#getting accuracy and loss of the model
loss, accuracy = model.evaluate(x_test, y_test)
print("accuracy", accuracy)
print("loss", loss)

#save model
model.save("digit_reconization.model")
'''
# ===After running above code one time the model will be saved so dont have to run again and again====

#loading the model after it has been trained
model = tf.keras.models.load_model("digit_reconization.model")
#reading images using cv2
for i in range(1,5):
    img =cv2.imread(f"Hand Written Digits/{i}.png")[:, :, 0]

    # convert the digit into black and the bg to white
    img =np.invert(np.array([img]))
    predection = model.predict(img)
    print(f"The digit is likely to be {np.argmax(predection)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()