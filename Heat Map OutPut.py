import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
print('helloz')
# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Get depth scale
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # Convert depth to meters

# Camera intrinsic parameters (adjust for accuracy)
fx, fy, cx, cy = 611.1494140625,609.7687377929688, 318.608642578125, 233.38153076171875  # Camera intrinsics (adjust as needed)

# Global variables for ROI selection
roi_selected = False
roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0
def bilinear_interpolation(img, x, y):
    """Perform bilinear interpolation for more accurate depth values"""
    x1, y1 = int(x), int(y)
    x2, y2 = min(x1 + 1, img.shape[1] - 1), min(y1 + 1, img.shape[0] - 1)

    Q11 = img[y1, x1]
    Q21 = img[y1, x2]
    Q12 = img[y2, x1]
    Q22 = img[y2, x2]

    R1 = (x2 - x) * Q11 + (x - x1) * Q21
    R2 = (x2 - x) * Q12 + (x - x1) * Q22

    return (y2 - y) * R1 + (y - y1) * R2
def pixel_to_world(depth, x, y):
    """Convert pixel coordinates to real-world size"""
    Z = depth * depth_scale  # Depth in meters
    X = (x - cx) * Z / fx  # Convert X from pixels to meters
    Y = (y - cy) * Z / fy  # Convert Y from pixels to meters
    return X, Y, Z  # Real-world coordinates

def select_roi(event, x, y, flags, param):
    """Mouse callback function to select the region of interest"""
    global roi_x, roi_y, roi_w, roi_h, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_x, roi_y = x, y  # Set start point
    elif event == cv2.EVENT_LBUTTONUP:
        roi_w, roi_h = x - roi_x, y - roi_y  # Set width and height
        roi_selected = True
        print(f"ROI Selected: X={roi_x}, Y={roi_y}, W={roi_w}, H={roi_h}")

def show_message():
    """Show a messagebox using Tkinter"""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    messagebox.showinfo("ROI Selection", "Draw a rectangle by clicking and dragging on the image.")
    root.destroy()

cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", select_roi)
show_message()  # Show instruction popup

try:
    while True:
        # Capture frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Show selection window
        if not roi_selected:
            temp_image = color_image.copy()
            cv2.putText(temp_image, "Select ROI (Click & Drag)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Select ROI", temp_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Crop to the selected ROI
        roi = color_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        roi_depth = depth_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Inside your detection loop (after obtaining ROI and contours)

        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Ignore small objects
                continue

            # Get the rotated bounding box
            rect = cv2.minAreaRect(contour)
            # rect returns ((center_x, center_y), (w, h), angle)
            (center_x, center_y), (w, h), angle = rect
            # Compute key depth points (ensure they are within bounds)
            center_px, center_py = int(center_x), int(center_y)
            top_px, top_py = max(center_px, 0), max(center_py - int(h / 2), 0)
            bottom_px, bottom_py = max(center_px, 0), min(center_py + int(h / 2), depth_image.shape[0] - 1)
            left_px, left_py = max(center_px - int(w / 2), 0), max(center_py, 0)
            right_px, right_py = min(center_px + int(w / 2), depth_image.shape[1] - 1), max(center_py, 0)

            # Get depth values at these points
            depth_vals = [
                roi_depth[center_py, center_px],
                roi_depth[top_py, top_px],
                roi_depth[bottom_py, bottom_px],
                roi_depth[left_py, left_px],
                roi_depth[right_py, right_px]
            ]

            # Convert to meters using RealSense depth scale
            depth_vals_meters = [d * depth_scale for d in depth_vals if d > 0]

            # Use the minimum depth (closest point) to improve accuracy
            if depth_vals_meters:
                depth_meters = min(depth_vals_meters)
            else:
                depth_meters = roi_depth[center_py, center_px] * depth_scale
            # Get the rotated bounding box
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (w, h), angle = rect

            # Get the box points to draw the rotated rectangle
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)

            # Adjust for ROI offset
            box_offset = box + np.array([roi_x, roi_y])
            cv2.polylines(color_image, [box_offset], isClosed=True, color=(0, 255, 0), thickness=2)

            # Identify top and bottom points (highest and lowest Y-coordinates)
            top_idx = np.argmin(box[:, 1])  # Point with min Y (top of object)
            bottom_idx = np.argmax(box[:, 1])  # Point with max Y (bottom of object)

            top_x, top_y = box[top_idx]
            bottom_x, bottom_y = box[bottom_idx]

            # Convert to depth frame coordinates
            top_depth = roi_depth[top_y - roi_y, top_x - roi_x]
            bottom_depth = roi_depth[bottom_y - roi_y, bottom_x - roi_x]

            # Convert to real-world coordinates
            _, _, top_Z = pixel_to_world(top_depth, top_x, top_y)
            _, _, bottom_Z = pixel_to_world(bottom_depth, bottom_x, bottom_y)

            # Calculate object height in meters and inches
            object_height_m = abs(bottom_Z - top_Z)
            object_height_in = object_height_m * 39.37

            # Display object height on image
            cv2.putText(color_image, f"Height: {object_height_in:.2f} in",
                        (roi_x + center_x, roi_y + center_y - 10),  # Position near the object
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Get the box points to draw the rotated rectangle
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            # If you are working in an ROI, add the ROI offset:
            box_offset = box + np.array([roi_x, roi_y])
            cv2.polylines(color_image, [box_offset], isClosed=True, color=(0, 255, 0), thickness=2)

            # Use the center of the rotated rectangle to get depth.
            # If you're in an ROI, note that center_x and center_y are relative to the ROI.
            center_px = int(center_x)
            center_py = int(center_y)
            depth_val = bilinear_interpolation(roi_depth, center_px, center_py)
            depth_meters = depth_val * depth_scale

            # Calculate the real-world dimensions.
            # Using the camera intrinsics: (fx, fy)
            width_meters = w * depth_meters / fx
            height_meters = h * depth_meters / fy

            # Convert to inches (1 meter = 39.37 inches)
            width_inches = width_meters * 39.37
            height_inches = height_meters * 39.37

            # Display measurements near the center of the object.
            cv2.putText(color_image, f"{width_inches:.2f} in x {height_inches:.2f} in",
                        (roi_x + center_px, roi_y + center_py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw ROI selection box
        cv2.rectangle(color_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        # Show the image with detected objects
        cv2.imshow("Object Detection", color_image)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
