"""
CAUTION !!!

The following code is used to detect common circular red laser points.
The parameters involved have been tested in an indoor scene with white lighting.
The maximum output power of the laser is less than 5mW, and the illuminated plane is a non-red diffuse reflection plane.

"""

import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create trackbar window
cv2.namedWindow('Adjustments')

# Create a trackbar to adjust the brightness threshold
def on_threshold_trackbar(val):
    pass  # This function does not need to do anything, as the trackbar will directly change the threshold variable

# Create a trackbar to adjust the lower threshold of red hue
def on_red_lower_trackbar(val):
    pass  # Same as above

# Create a trackbar to adjust the upper threshold of red hue
def on_red_upper_trackbar(val):
    pass  # Same as above

# Trackbar range and default values
brightness_threshold_default = 210
red_lower_threshold_default = 10
red_upper_threshold_default = 20

# Create trackbars
cv2.createTrackbar('Brightness Threshold', 'Adjustments', brightness_threshold_default, 255, on_threshold_trackbar)
cv2.createTrackbar('Red Lower Threshold', 'Adjustments', red_lower_threshold_default, 180, on_red_lower_trackbar)
cv2.createTrackbar('Red Upper Threshold', 'Adjustments', red_upper_threshold_default, 180, on_red_upper_trackbar)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera, exiting program.")
        break

    # Convert the BGR image to HSV format
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Separate the H, S, V channels of the HSV image
    h, s, v = cv2.split(hsv)

    # Get the current value of the trackbar
    brightness_threshold = cv2.getTrackbarPos('Brightness Threshold', 'Adjustments')
    red_lower_threshold = cv2.getTrackbarPos('Red Lower Threshold', 'Adjustments')
    red_upper_threshold = cv2.getTrackbarPos('Red Upper Threshold', 'Adjustments')

    # Use Canny edge detection
    edges = cv2.Canny(v, 50, 150)  # Adjust the thresholds to suit different scenes

    # Find contours in the edges
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter circular contours
    circular_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter != 0 else 0
        if circularity > 0.8:  # Adjust the threshold to filter more circular contours
            circular_contours.append(contour)

    # Create a brightness threshold to find brighter areas
    _, v_thresholded = cv2.threshold(v, brightness_threshold, 255, cv2.THRESH_BINARY)

    # Create a red hue mask
    red_mask = cv2.inRange(h, red_lower_threshold, red_upper_threshold) + cv2.inRange(h, red_upper_threshold + 160, 180)

    # Combine the brightness mask and the red hue mask
    mask = cv2.bitwise_and(v_thresholded, red_mask)

    # Remove noise using morphological operations
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    # Draw circles in the filtered circular contours
    for contour in circular_contours:
        # Calculate the bounding circle of the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        # Draw a circle at the center point
        cv2.circle(frame, center, radius, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Canny Edges', edges)
    cv2.imshow('Red Laser Detection', frame)

    # Exit loop if 'q' key is pressed or if any window is closed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Red Laser Detection', cv2.WND_PROP_VISIBLE) < 1 or \
       cv2.getWindowProperty('Canny Edges', cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty('Adjustments', cv2.WND_PROP_VISIBLE) < 1:
        break
        
# Release the camera resource
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
