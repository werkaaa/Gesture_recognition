# Python_project
[Weronika Ormaniec](https://github.com/werkaaa), [Adam Kania](https://github.com/remilvus)
# Rozpoznawanie Gestów
## Instrukcja obsługi:
Zanim program zacznie klasyfikować gesty, należy wykonać krótką konfigurację, która obejmuje ustawienie tła oraz pobranie koloru z dłoni (rękawiczki).

Jeżeli komputer nie ma dostępnej karty graficznej, rekomendujemy ubranie kolorowej rękawiczki, gdyż w takim wypadku program nie korzysta z modułu Face Recognition.

### Opcje konfiguracji:
* B (środkowy przycisk myszy) - Dodaj tło
* P - Pobierz próbkę koloru, który powinien być wyszukany (czerwony krzyżyk)
* Q - Zamknij program
* D - Wyświetl okna do konfiguracji
* C - Wyczyść tło i próbki koloru
* A - Zmień sposób wyszukiwania dłoni
* R - Ustaw, wykrywanie czerwonej rękawiczki
* M - Niespodzianka!

* lewy przycisk myszy (jedno kliknięcie) - przesuń punkt pobierania koloru
* lewy przycisk myszy (dwa kliknięcia) - pobierz kolor z miejsca kliknięcia

### Parametry możliwe do ustawienia:
-granice tolerancji dla kolorów(HSV) (dla skóry, dla tła):
* skin_Sat, back_Sat - saturacja 
* skin_H_up, back_H_up - odcień (górna granica)
* skin_H_do, back_H_do - odcień (dolna granica)
* skin_V_up, back_V_up - moc światła białego (górna granica)
* skin_V_do, back_V_do - moc światła białego (dolna granica)
-inne:
* kernel_size - wielkość obszaru usuwania szumu
* alpha - mnożnik wygładzania maski

## Notebook do trenowania modelu:
https://colab.research.google.com/drive/189vYRl-LEkticxFXlKF3IC8P1YliIxU4

## Źródło danych do trenowania modelu:
https://github.com/athena15/project_kojak

# Gesture Recognition
## Manual:
Before the program starts classifying gestures, some short configuration is needed. You should set the background and sample the color from the image of your hand (glove).

If your computer does not have access to the GPU, please use some vividly colored glove, since in this case, the program does not use the Face Recognition module.

### Configuration:
* B (middle  mouse button)  - Add background
* P  - Sample the color (red cross)
* Q - Quit
* D - Show more configuration windows
* C - clear the background and color samples
* A - Change the way of hand localization (alternatively top-right corner)
* R - Set red glove recognition helper
* M - Suprise!

Left mouse button (single-click) - Move the color sampling point
Left mouse button (double-click) - Sample color from the clicked point

## Notebook where the model was trained:
https://colab.research.google.com/drive/189vYRl-LEkticxFXlKF3IC8P1YliIxU4

## Dataset:
https://github.com/athena15/project_kojak
