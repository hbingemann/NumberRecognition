
import mnist
import tensorflow as tf
import numpy as np
from display import show_training_history


def create_and_save_model():

    # set the random state for reproducibility
    tf.random.set_seed(43)

    # get training and testing labels
    X_train_raw, y_train_raw = mnist.train_images(), mnist.train_labels()
    X_test_raw, y_test_raw = mnist.test_images(), mnist.test_labels()

    # make values 0-1 in the image
    X_train = X_train_raw / 255
    X_test = X_test_raw / 255

    # make the images a one dimensional array
    _, rows, cols = X_train.shape
    X_train = np.reshape(X_train, newshape=(len(X_train), rows*cols))
    X_test = np.reshape(X_test, newshape=(len(X_test), rows*cols))

    # one hot encode the y values
    y_train = tf.one_hot(y_train_raw, depth=10)
    y_test = tf.one_hot(y_test_raw, depth=10)

    # create the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(20, activation="relu"))
    model.add(tf.keras.layers.Dense(20, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    # compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.optimizers.Adam(),
        metrics="accuracy"
    )

    # train the model
    training_history = model.fit(X_train, y_train, epochs=8)

    # tensorflow evaluation
    model.evaluate(X_test, y_test)

    # show training history
    show_training_history(training_history)

    # save the model
    model.save("models/first_model")
