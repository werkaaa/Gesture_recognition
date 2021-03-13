from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from constants import *
import numpy as np
import cv2
from typing import List

from tflite_runtime.interpreter import Interpreter


class ObjectDetector():
    def __init__(self):
        self._interpreter = Interpreter(str(OBJECT_DETECTION_PATH))
        self._interpreter.allocate_tensors()
        _, _input_height, _input_width, _ = self._interpreter.get_input_details()[0]['shape']
        self._input_height = _input_height
        self._input_width = _input_width

    def process_image(self, image):
        """detects objects on image of any size"""
        resized_image = cv2.resize(image, (self._input_height, self._input_width))
        results = self.detect_objects(resized_image)
        return results

    def detect_objects(self, image) -> List[GestureData]:
        """Returns a list of detection results"""
        
        self._set_input_tensor(image)
        self._interpreter.invoke()

        # Get all output details
        boxes = self._get_output_tensor(0)
        classes = self._get_output_tensor(1)
        scores = self._get_output_tensor(2)
        count = int(self._get_output_tensor(3))

        results = []
        for i in range(count):
            if scores[i] >= OBJECT_MIN_CONFIDENCE:
                result = GestureData(boxes[i], int(classes[i]), scores[i])
                results.append(result)
        return results

    def _set_input_tensor(self, image):
        """Sets the input tensor."""
        tensor_index = self._interpreter.get_input_details()[0]['index']
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self, index):
        """Returns the output tensor at the given index."""
        output_details = self._interpreter.get_output_details()[index]
        tensor = np.squeeze(
            self._interpreter.get_tensor(output_details['index']))
        return tensor
