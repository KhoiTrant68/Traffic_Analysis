import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import imutils

import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv

# Define colors
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
black = (0, 0, 0)
white = (255, 255, 255)


thickness = 2
font_scale = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

# Define dicts for coordinate
up = {}
down = {}
left = {}
right = {}

# Define zones
polygon_up = np.array([[592, 246], [709, 231], [891, 324], [706, 340]], np.int32)

polygon_down = np.array([[535, 539], [882, 508], [964, 585], [565, 587]], np.int32)

polygon_left = np.array([[220, 378], [414, 367], [431, 422], [228, 437]], np.int32)

polygon_right = np.array([[1057, 382], [1275, 367], [1279, 452], [1228, 480]], np.int32)

# Define model
model = YOLO("yolov8x.pt")

# Define id classes for observed vehicles
print(model.names)
vehicles = [2, 3, 5, 7]

# Define track history
track_history = defaultdict(lambda: [])

# Define output
width = 1280
height = 720


path = "data/4CornersCrossroad.mp4"

cap = cv2.VideoCapture(path)


fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("video.avi", fourcc, 20.0, (width, height))


while True:
    ret, frame = cap.read()
    if ret == False:
        break

    frame = imutils.resize(frame, width=1280)

    cv2.polylines(frame, [polygon_up], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(
        frame, [polygon_down], isClosed=True, color=green, thickness=thickness
    )
    cv2.polylines(
        frame, [polygon_left], isClosed=True, color=green, thickness=thickness
    )
    cv2.polylines(
        frame, [polygon_right], isClosed=True, color=green, thickness=thickness
    )

    results = model.track(frame, persist=True, verbose=False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")

    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        # Define center of bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if class_id in vehicles:
            cv2.circle(frame, (cx, cy), 3, blue, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), blue, thickness=1)

            class_name = results.names[class_id].upper()
            text = "ID:{} {}".format(track_id, class_name)
            cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, blue, thickness)

            track = track_history[track_id]
            track.append((cx, cy))

            if len(track) > 30:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame, [points], isClosed=False, color=blue, thickness=thickness
            )

            up_result = cv2.pointPolygonTest(polygon_up, (cx, cy), measureDist=False)
            down_result = cv2.pointPolygonTest(
                polygon_down, (cx, cy), measureDist=False
            )
            left_result = cv2.pointPolygonTest(
                polygon_left, (cx, cy), measureDist=False
            )
            right_result = cv2.pointPolygonTest(
                polygon_right, (cx, cy), measureDist=False
            )

            if up_result > 0:
                up[track_id] = x1, y1, x2, y2

            if down_result > 0:
                down[track_id] = x1, y1, x2, y2

            if left_result > 0:
                left[track_id] = x1, y1, x2, y2

            if right_result > 0:
                right[track_id] = x1, y1, x2, y2
    up_counter_text = "Up Direction Counter: {}".format(str(len(list(up.keys()))))
    down_counter_text = "Down Direction Counter: {}".format(str(len(list(down.keys()))))
    left_counter_text = "Left Direction Counter: {}".format(str(len(list(left.keys()))))
    right_counter_text = "Right Direction Counter: {}".format(
        str(len(list(right.keys())))
    )

    cv2.rectangle(frame, (0, 0), (350, 150), white, -1)
    cv2.putText(frame, up_counter_text, (10, 25), font, 0.8, black, thickness)
    cv2.putText(frame, down_counter_text, (10, 65), font, 0.8, black, thickness)
    cv2.putText(frame, left_counter_text, (10, 105), font, 0.8, black, thickness)
    cv2.putText(frame, right_counter_text, (10, 145), font, 0.8, black, thickness)

    writer.write(frame)
    cv2.imshow("Direction Counter", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
cap.release()
writer.release()
cv2.destroyAllWindows()
print("[INFO]...Processing is finished!")
