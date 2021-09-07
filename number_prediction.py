import numpy as np


def predict_number(img, model):
    img = img.flatten()
    prediction = model.predict(np.expand_dims(img, axis=0)).squeeze()
    return np.argmax(prediction)
