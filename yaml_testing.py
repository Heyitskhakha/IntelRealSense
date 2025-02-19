import pyrealsense2 as rs
import numpy as np
import cv2

def capture_and_detect_objects(background_image_file):
    # Load the background image
    background_image = cv2.imread(background_image_file)

    # Check if the background image is loaded correctly
    if background_image is None:
        print(f"Error: Unable to load background image from {background_image_file}")
        return

    # Set up RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable RGB and depth streams
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # Start the pipeline
    pipeline.start(config)

    # Hardcoded focal length and sensor width
    focal_length_x = 895.39
    focal_length_y = 895.39
    sensor_width = 6.4  # mm

    # Define crop area (x, y, width, height)
    crop_x, crop_y, crop_width, crop_height = 300, 200, 800, 300

    # Depth-to-color alignment
    align = rs.align(rs.stream.color)

    # Create OpenCV window to display images
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

    while True:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get the aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Convert depth and color frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Crop the region of interest (ROI)
        cropped_color_image = color_image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        cropped_background_image = background_image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

        # Subtract the background image (cropped)
        fg_mask = cv2.absdiff(cropped_background_image, cropped_color_image)
        fg_mask_gray = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(fg_mask_gray, (5, 5), 0)

        # Apply Canny Edge Detection
        edges = cv2.Canny(blurred, 50, 150)  # Adjust thresholds as needed

        # Find contours from the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour
        for contour in contours:
            # Skip small contours
            if cv2.contourArea(contour) < 100:
                continue

            # Get the rotated bounding box of the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Draw the rotated bounding box
            cv2.drawContours(cropped_color_image, [box], 0, (0, 255, 0), 2)

            # Extract dimensions and angle
            (width, height) = rect[1]
            angle = rect[2]

            # Convert dimensions from pixels to millimeters
            width_mm = (width * sensor_width * (397-7.5)) / (640 * focal_length_x) * 100
            height_mm = (height * sensor_width * (397-7.5)) / (640 * focal_length_x) * 100

            # Optional offset for improving accuracy
            offset = 0  # mm offset, can be adjusted based on empirical measurements
            width_mm -= offset
            height_mm -= offset

            # Display the dimensions and angle
            cv2.putText(cropped_color_image, f'Width: {width_mm:.2f} mm',
                        (int(rect[0][0] - width / 2), int(rect[0][1] - height / 2 - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(cropped_color_image, f'Height: {height_mm:.2f} mm',
                        (int(rect[0][0] - width / 2), int(rect[0][1] - height / 2 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(cropped_color_image, f'Angle: {angle:.2f}°',
                        (int(rect[0][0] - width / 2), int(rect[0][1] - height / 2 - 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Overlay the edges on the cropped color image
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to 3-channel
        overlay_image = cv2.addWeighted(cropped_color_image, 0.8, edges_colored, 0.2, 0)

        # Display the processed image with detected edges and dimensions
        cv2.imshow('Detected Objects', overlay_image)

        # Wait for key press to capture a new frame or exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'Esc' to exit
            break

    # Stop the pipeline after capturing the image
    pipeline.stop()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Provide the background image path
    background_image_file = 'C:/Users/Sky/PycharmProjects/IntelRealSense/background_image.png'  # Adjust path
    capture_and_detect_objects(background_image_file)
