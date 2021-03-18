import pathlib

OBJECT_DETECTION_PATH = pathlib.Path('/', 'tmp', 'detect.tflite')
OBJECT_MIN_CONFIDENCE = 0.5
OBJECT_MIN_COVER = 0.15
EROSION_SIZE = 5
MIN_WINDOW_SIZE = 40
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 400
SEARCH_FOR_HAND = True

BACKGROUND_DIFF_PIX = 25  # defines how much the background pixels can change
BACKGROUND_DIFF = 0.05  # maximum allowed change of background (% of all pixels)
BACKGROUND_TIMER = 20 # defines frequency of background update attempts

class GestureData():
    def __init__(self, box, object_id: int, object_score: float):
        self.box = box
        self.object_score = object_score
        self.object_id = object_id
        self.gesture_label = None
        self.gesture_score = None
