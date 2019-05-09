import numpy as np
import cv2 as cv
import skin_finder as SkinFinder
import hand_finder


if __name__=="__main__":
    cam = cv.VideoCapture(0)

    width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

    finder = SkinFinder(width, height)

    while True:
        ret, frame = cam.read()
        key_input = cv.waitKey(1)
        if key_input == ord('b'):
            finder.add_background(frame)

        frame = finder.repair_brightness(frame)

        if key_input == ord('p'):
            finder.get_skin_color(frame)

        if key_input == ord('q'):
            break

        if key_input == ord('a'):
            x = int(input("x: "))
            y = int(input("y: "))
            finder.add_probing_point((x, y))

        masks_merged = finder.get_important_area(frame)

        frame = finder.place_marker(frame)
        #       frame = cv.drawMarker(frame, find_hand(masks_merged)[:2], (0, 255, 0))

        skin_mask = finder.get_skin_mask(frame)
        foreground = finder.get_foreground_mask(frame)
        # cv.imshow("skin_mask", skin_mask)
        # cv.imshow("foreground_no_noise", foreground)
        cv.imshow("masks_merged", masks_merged)
        cv.imshow("frame", frame)