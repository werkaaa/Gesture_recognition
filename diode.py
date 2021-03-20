try:
  import board
  import neopixel
except:
  pass

from constants import *

class Diode():
  def __init__(self):
    try:
      self.pixels = neopixel.NeoPixel(board.D18, 16, brightness=0.1, auto_write=False)
    except:
      self.pixels = [()]*16

  def shine_all(self, gestures):
    try:
      self.pixels.fill((0, 0, 0))
    except:
      self.pixels = [()] * 16
    for gesture in gestures:
      if gesture in gestures:
        to_shine = DIODE_NUMBERS[gesture.gesture_label]
        self.pixels[to_shine[0]] = COLORS[gesture.gesture_label]
        self.pixels[to_shine[1]] = COLORS[gesture.gesture_label]
    try:
      self.pixels.show()
    except:
      print(self.pixels)

