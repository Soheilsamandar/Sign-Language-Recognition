import cv2 as cv
import numpy as np
import tensorflow as tf
import math

from pathlib import Path
from cvzone.HandTrackingModule import HandDetector

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "output" / "keras_model.h5"
LABELS_PATH = BASE_DIR / "output" / "labels.txt"


# -----------------------------
# Load Model
# -----------------------------

model = tf.keras.models.load_model(MODEL_PATH)

label_map = {}

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    for line in file:
        index, label = line.strip().split(maxsplit=1)
        label_map[int(index)] = label

labels = [label_map[i] for i in range(len(label_map))]

model_height = model.input_shape[1]
model_width = model.input_shape[2]

print("Model input:", model.input_shape)
print("Model output:", model.output_shape)
print("Labels:", labels)


# -----------------------------
# Camera
# -----------------------------

cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300


# -----------------------------
# Main Loop
# -----------------------------

while True:

    success, img = cap.read()

    if not success:
        print("Camera error!")
        break

    imgOutput = img.copy()

    hands, imgOutput = detector.findHands(imgOutput)

    if hands:

        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # IMPORTANT:
        # Crop from imgOutput, not img
        frame_h, frame_w = imgOutput.shape[:2]

        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(frame_w, x + w + offset)
        y2 = min(frame_h, y + h + offset)

        imgCrop = imgOutput[y1:y2, x1:x2]

        if imgCrop.size != 0:

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspectRatio = h / w

            if aspectRatio > 1:

                k = imgSize / h
                wCal = math.ceil(k * w)

                imgResize = cv.resize(imgCrop, (wCal, imgSize))

                wGap = math.ceil((imgSize - wCal) / 2)

                imgWhite[:, wGap : wGap + wCal] = imgResize

            else:

                k = imgSize / w
                hCal = math.ceil(k * h)

                imgResize = cv.resize(imgCrop, (imgSize, hCal))

                hGap = math.ceil((imgSize - hCal) / 2)

                imgWhite[hGap : hGap + hCal, :] = imgResize

            # BGR to RGB
            imgRGB = cv.cvtColor(imgWhite, cv.COLOR_BGR2RGB)

            # Resize
            imgModel = cv.resize(imgRGB, (model_width, model_height))

            inputImage = np.expand_dims(imgModel.astype(np.float32), axis=0)

            # Prediction
            prediction = model.predict(inputImage, verbose=0)[0]

            index = int(np.argmax(prediction))
            confidence = float(prediction[index]) * 100

            label = labels[index]

            # Display prediction
            cv.putText(
                imgOutput,
                f"{label} ({confidence:.1f}%)",
                (x, max(30, y - 20)),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2,
            )

            # Show processed image
            cv.imshow("White Image", imgWhite)

            print(
                {
                    labels[i]: round(float(prediction[i]) * 100, 2)
                    for i in range(len(labels))
                }
            )

    cv.imshow("Recognition", imgOutput)

    key = cv.waitKey(1)

    if key == 27:
        break


cap.release()
cv.destroyAllWindows()
