import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
from PIL import Image

class GestureClassifier():
    def __init__(self):
        self._interpreter = Interpreter('model/model.tflite')
        self._interpreter.allocate_tensors()
        self.labels = ['C', 'L', 'fist', 'okay', 'palm', 'peace']

    def set_input_tensor(self, image):
        tensor_index = self._interpreter.get_input_details()[0]['index']
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def softmax(self, z):
        z -= np.max(z, keepdims=True)
        return np.exp(z) / np.sum(np.exp(z), keepdims=True)

    def classify(self, image):
        """Returns predicted gesture given an image."""
        # image = image.resize((256, 256), Image.BILINEAR)
        image = cv2.resize(image, (256, 256))

        image = np.asarray(image).reshape(256, 256, 1)
        self.set_input_tensor(image)
        self._interpreter.invoke()
        output_details = self._interpreter.get_output_details()[0]
        output = np.squeeze(self._interpreter.get_tensor(output_details['index']))
        s = self.softmax(output)
        return self.labels[np.argmax(s)], np.max(s)
