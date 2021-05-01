import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
from constants import *

class GestureClassifier():
    def __init__(self):
        self._interpreter = Interpreter(str(GESTURE_CLASSIFICATION_PATH))
        self._interpreter.allocate_tensors()

    def set_input_tensor(self, image):
        tensor_index = self._interpreter.get_input_details()[0]['index']
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def softmax(self, z):
        z -= np.max(z, keepdims=True)
        return np.exp(z) / np.sum(np.exp(z), keepdims=True)

    def classify(self, image):
        """Returns predicted gesture given an image."""
        image = cv2.resize(image, (256, 256))

        image = np.asarray(image).reshape(256, 256, 1)
        self.set_input_tensor(image)
        self._interpreter.invoke()
        output_details = self._interpreter.get_output_details()[0]
        output = np.squeeze(self._interpreter.get_tensor(output_details['index']))
        s = self.softmax(output)

        return LABELS[np.argmax(s)], np.max(s)
