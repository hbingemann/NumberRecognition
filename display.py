import matplotlib.pyplot as plt
from matplotlib import pylab
import pandas as pd


def set_title(title):
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title(title)


def show_img(img):
    plt.imshow(img)
    plt.show()


def show_training_history(training_history):
    pd.DataFrame(training_history.history).plot()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
