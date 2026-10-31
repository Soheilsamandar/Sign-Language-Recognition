import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)


offset = 20
imgSize = 300
folder = "Data/C"
counter = 0

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # Reset white canvas every frame
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = int(math.ceil(imgSize - wCal) / 2)
            imgWhite[:, wGap : wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = int(math.ceil(imgSize - hCal) / 2)
            imgWhite[hGap : hCal + hGap, :] = imgResize
        cv.imshow("imgCrop", imgCrop)
        cv.imshow("imgWhite", imgWhite)
    cv.imshow("frame", img)
    key = cv.waitKey(1)
    if key == 27:
        break
    if key == ord("s"):
        counter += 1
        cv.imwrite(f"{folder}/image_{time.time()}.jpg", imgWhite)
        print(counter)
