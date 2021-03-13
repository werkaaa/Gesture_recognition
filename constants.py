import pathlib

OBJECT_DETECTION_PATH = pathlib.Path('/', 'tmp', 'detect.tflite')
OBJECT_MIN_CONFIDENCE = 0.4
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 400
SEARCH_FOR_HAND = True

class GestureData():
    def __init__(self, box, object_id: int, object_score: float):
        self.box = box
        self.object_score = object_score
        self.object_id = object_id
        self.gesture_label = None
        self.gesture_score = None
