import keras
import pygame
from drawing import DrawingGrid, draw_texts_top_right
from constants import FPS, SCREEN_SIZE, BACKGROUND_COLOR
from number_prediction import predict_number
import mnist
import tensorflow as tf
import numpy as np
import pandas as pd
from display import show_img, set_title, show_training_history


def main():
    # initialize pygame
    pygame.init()

    # load model
    model = keras.models.load_model("models/first_model")

    # set the random state for reproducibility
    tf.random.set_seed(43)

    # # get training and testing labels
    # X_train_raw, y_train_raw = mnist.train_images(), mnist.train_labels()
    # X_test_raw, y_test_raw = mnist.test_images(), mnist.test_labels()
    #
    # # make values 0-1 in the image
    # X_train = X_train_raw / 255
    # X_test = X_test_raw / 255
    #
    # # make the images a one dimensional array
    # _, rows, cols = X_train.shape
    # X_train = np.reshape(X_train, newshape=(len(X_train), rows*cols))
    # X_test = np.reshape(X_test, newshape=(len(X_test), rows*cols))
    #
    # # one hot encode the y values
    # y_train = tf.one_hot(y_train_raw, depth=10)
    # y_test = tf.one_hot(y_test_raw, depth=10)

    # # see the computers predictions
    # predictions = model.predict(X_test)
    # for i, img in enumerate(X_test_raw):
    #     set_title(np.argmax(predictions[i]))
    #     show_img(img)

    # do number drawing and predicting
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Number Recognition")
    grid = DrawingGrid(28, 28)

    run = True
    clock = pygame.time.Clock()
    while run:
        clock.tick(FPS)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                run = False

            else:
                grid.handle_event(event)

        screen.fill(BACKGROUND_COLOR)
        grid.display(screen)
        draw_texts_top_right(screen,
                             f"Computer prediction: {predict_number(grid.get_grid_for_predicting(), model)}",
                             "Press C to clear")
        pygame.display.update()

    pygame.quit()


if __name__ == '__main__':
    main()
