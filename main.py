import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import font as tkFont

thres = 0.45  # Threshold to detect object
nms_threshold = 0.2
cap = None  # Declare a global variable for the video capture

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# HSV range values for color detection
lower_range_red = np.array([0, 120, 70])
upper_range_red = np.array([10, 255, 255])

lower_range_green = np.array([50, 100, 100])
upper_range_green = np.array([70, 255, 255])

lower_range_blue = np.array([90, 100, 100])
upper_range_blue = np.array([120, 255, 255])

lower_range_yellow = np.array([20, 100, 100])
upper_range_yellow = np.array([30, 255, 255])

lower_range_black = np.array([0, 0, 0])
upper_range_black = np.array([180, 255, 30])

def detect_objects(frame):
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        color_roi = frame[y:y+h, x:x+w]

        # Detect color within the bounding box
        color = detect_color(color_roi)

        # Draw bounding box and display color rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), color=lower_range_dict[color], thickness=cv2.FILLED)
        cv2.putText(frame, f"{classNames[classIds[i] - 1].upper()} - {color}", (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def detect_color(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask_red = cv2.inRange(hsv_roi, lower_range_red, upper_range_red)
    mask_green = cv2.inRange(hsv_roi, lower_range_green, upper_range_green)
    mask_blue = cv2.inRange(hsv_roi, lower_range_blue, upper_range_blue)
    mask_yellow = cv2.inRange(hsv_roi, lower_range_yellow, upper_range_yellow)
    mask_black = cv2.inRange(hsv_roi, lower_range_black, upper_range_black)

    colors = {
        "Red": cv2.countNonZero(mask_red),
        "Green": cv2.countNonZero(mask_green),
        "Blue": cv2.countNonZero(mask_blue),
        "Yellow": cv2.countNonZero(mask_yellow),
        "Black": cv2.countNonZero(mask_black),
    }

    predominant_color = max(colors, key=colors.get)
    return predominant_color

# Dictionary to map color names to corresponding lower range for rectangle color
lower_range_dict = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Black": (0, 0, 0)
    
}

# GUI Functions
def start_detection():
    global cap
    cap = cv2.VideoCapture(0)
    show_frame()

def show_frame():
    global cap
    if cap is not None:
        success, img = cap.read()
        img = cv2.resize(img, (640, 480))

        img = detect_objects(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        panel.img = img
        panel.config(image=img)
        panel.image = img

        panel.after(10, show_frame)

def close_window():
    global cap
    if cap is not None:
        cap.release()
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Modern Object Detection")

# Styling
backgroundColor = "#222831"
buttonColor = "#00adb5"
textColor = "#eeeeee"
fontStyle = tkFont.Font(family="Lucida Grande", size=12)

root.configure(bg=backgroundColor)

# Image panel
panel = tk.Label(root, bg=backgroundColor)
panel.pack(padx=10, pady=10)

# Start button
start_button = tk.Button(root, text="Start Detection", command=start_detection, bg=buttonColor, fg=textColor, font=fontStyle)
start_button.pack(pady=10)

# Exit button
exit_button = tk.Button(root, text="Exit", command=close_window, bg=buttonColor, fg=textColor, font=fontStyle)
exit_button.pack(pady=10)

# Main loop
root.mainloop()