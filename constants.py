import pathlib

HEADLESS = True

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

GESTURE_CLASSIFICATION_PATH = pathlib.Path(__file__).parent.joinpath('model/model.tflite')
LABELS = ['C', 'L', 'fist', 'okay', 'palm', 'peace']
COLORS = {'C': (255, 0, 0),
          'L': (0, 255, 0),
          'fist': (0, 0, 255),
          'okay': (255, 255, 0),
          'palm': (255, 0, 255),
          'peace': (0, 255, 255)}
DIODE_NUMBERS = {'C': (1, 2),
                 'L': (3, 4),
                 'fist': (5, 6),
                 'okay': (9, 10),
                 'palm': (11, 12),
                 'peace': (13, 14)}

class GestureData():
    def __init__(self, box, object_id: int, object_score: float):
        self.box = box
        self.object_score = object_score
        self.object_id = object_id
        self.gesture_label = 'None'
        self.gesture_score = 0
