# Rozpoznawanie Gestów
[Weronika Ormaniec](https://github.com/werkaaa), [Adam Kania](https://github.com/remilvus)

Aplikacja na Raspberry Pi. Może oznaczać gesty na obrazie z kamery oraz sygnalizować klasyfikację diodami.

## Przykłady działania:
Wszystkie dostępne gesty pokazane w kolejności: C, L, pięść, ok, cała dłoń, victoria  
![All gestures showed one by one](https://github.com/werkaaa/Gesture_recognition/blob/raspberry-pi/examples/all_gestures.gif)  

Dwa gesty (C, ok) pokazane naraz  
![Two gestures at once (C and okay)](https://github.com/werkaaa/Gesture_recognition/blob/raspberry-pi/examples/multiple_gestures_at_once.gif)  

## Notebook do trenowania modelu:
[Notebook](https://colab.research.google.com/drive/1MmNZ0twx4wT_GgKj8PIRguof5-ERyw1A?usp=sharing)

## Biblioteki, z których korzystaliśmy:
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Tensorflow](https://www.tensorflow.org/lite)
* [Neopixel](https://learn.adafruit.com/neopixels-on-raspberry-pi/python-usage)
* inne podstawowe

## Źródło danych do trenowania modelu:
[Dane](https://github.com/athena15/project_kojak)

