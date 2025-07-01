import numpy as np
import cv2
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Color points storage
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]
blue_index, green_index, red_index, yellow_index = 0, 0, 0, 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Create a white canvas
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
buttons = [
    ((40, 1), (140, 65), "CLEAR", (0, 0, 0)),
    ((160, 1), (255, 65), "BLUE", colors[0]),
    ((275, 1), (370, 65), "GREEN", colors[1]),
    ((390, 1), (485, 65), "RED", colors[2]),
    ((505, 1), (600, 65), "YELLOW", colors[3]),
]

for (start, end, text, color) in buttons:
    cv2.rectangle(paintWindow, start, end, color, -1)
    cv2.putText(paintWindow, text, (start[0] + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    center = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            x1, y1 = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            x2, y2 = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])

            center = (x1, y1)

            # Fix distance calculation
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Pinch threshold
            PINCH_THRESHOLD = 40

            if distance < PINCH_THRESHOLD:
                center = None  # Hide cursor when pinching (stops drawing)

            # Draw fingertip cursor if not pinching
            if center:
                cv2.circle(frame, center, 8, (0, 255, 0), -1)

            # If touching the color selection area
            if center and y1 <= 65:
                if 40 <= x1 <= 140:  # Clear
                    bpoints.clear()
                    gpoints.clear()
                    rpoints.clear()
                    ypoints.clear()
                    bpoints.append(deque(maxlen=512))
                    gpoints.append(deque(maxlen=512))
                    rpoints.append(deque(maxlen=512))
                    ypoints.append(deque(maxlen=512))
                    blue_index = green_index = red_index = yellow_index = 0
                    paintWindow[67:, :, :] = 255
                elif 160 <= x1 <= 255:
                    colorIndex = 0  # Blue
                elif 275 <= x1 <= 370:
                    colorIndex = 1  # Green
                elif 390 <= x1 <= 485:
                    colorIndex = 2  # Red
                elif 505 <= x1 <= 600:
                    colorIndex = 3  # Yellow

            # Draw only when not pinching
            elif center:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)

    else:
        # Append new deque if no finger is detected
        bpoints.append(deque(maxlen=512))
        gpoints.append(deque(maxlen=512))
        rpoints.append(deque(maxlen=512))
        ypoints.append(deque(maxlen=512))
        blue_index += 1
        green_index += 1
        red_index += 1
        yellow_index += 1
        
        # Display message when no hand is detected
        cv2.putText(frame, "Hand Not Detected", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw the paint strokes
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Draw buttons on the frame
    for (start, end, text, color) in buttons:
        cv2.rectangle(frame, start, end, color, -1)
        cv2.putText(frame, text, (start[0] + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
