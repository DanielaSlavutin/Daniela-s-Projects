import numpy as np
import cv2
from PIL import Image

#  Function to get HSV limits for each color
def get_limits_by_color(color_name):
    color_ranges={
        "Red": ([0,100,100], [10,255,255]),
        "Orange": ([11,100,100], [22,255,255]),
        "Yellow": ([23,100,100],[33,255,255]),
        "Green": ([34,100,100],[78,255,255]),
        "Blue": ([79,100,100],[131,255,255]),
        "Violet": ([132,100,100],[170,255,255])
    }
    lower, upper = color_ranges[color_name]
    return np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)

# Function to draw rectangles around detected objects
def draw_color_borders(frame, hsv_frame, color_name):
    lowerLim, upperLim = get_limits_by_color(color_name)
    mask = cv2.inRange(hsv_frame, lowerLim, upperLim)

    # Find contours for the mask
    contours,_ =  cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500: # Filter out small objects
            x, y, w, h = cv2.boundingRect(contour)
            # Draw rectangle and label for each color
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
            cv2.putText(frame, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# List of colors and their initial color index
colors=["Red", "Orange", "Yellow", "Green", "Blue", "Violet"]
current_color_index=0

print(f"Press 'n' to switch to the next color. Current color: {colors[current_color_index]}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Draw borders for the currently selected color
    draw_color_borders(frame, hsv_frame, colors[current_color_index])

    # Show the frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Quit
        break
    elif key == ord('n'): # Switch to the next color
        current_color_index = (current_color_index +1 ) % len(colors)
        print(f"Switched to {colors[current_color_index]}")


cap.release()
cv2.destroyAllWindows()