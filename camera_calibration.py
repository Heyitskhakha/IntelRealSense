import pyrealsense2 as rs
import numpy as np
import cv2
import yaml
import os


def calibrate_camera():
    # Set up RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable RGB and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the pipeline
    pipeline.start(config)

    # Define checkerboard dimensions
    board_size = (8, 6)  # Adjust this according to your checkerboard
    square_size = 0.025  # in meters (e.g., 16mm)

    # Prepare object points in 3D world coordinates (with Z = 0)
    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.indices(board_size).T.reshape(-1, 2) * square_size

    # Create list to hold object points and image points
    obj_points = []
    img_points = []

    # Folder to save the captured images
    image_folder = 'calibration_images'
    os.makedirs(image_folder, exist_ok=True)

    # Capture images of the checkerboard from the RealSense camera
    print("Move the checkerboard to different poses and press 'q' to capture each image.")
    capture_count = 0

    while capture_count < 10:  # Capture 10 different poses
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert to grayscale for corner detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, board_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        # Visualize the detected corners
        if ret:
            cv2.drawChessboardCorners(color_image, board_size, corners, ret)
            cv2.imshow("Checkerboard Corners", color_image)  # Display the checkerboard with detected corners
            print(f"Pose {capture_count + 1}: Checkerboard detected!")

            # Refine corner locations to subpixel accuracy
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

            # If corners are detected, add them to the list
            img_points.append(corners_subpix)
            obj_points.append(objp)

            # Save the captured color and depth images
            color_image_path = os.path.join(image_folder, f"color_{capture_count + 1}.png")
            depth_image_path = os.path.join(image_folder, f"depth_{capture_count + 1}.png")

            cv2.imwrite(color_image_path, color_image)
            cv2.imwrite(depth_image_path, depth_image)

            print(f"Captured pose {capture_count + 1}/10. Move the checkerboard to another position.")

            # Increment the capture count
            capture_count += 1

        # Display the images
        cv2.imshow("Captured Color Image", color_image)
        cv2.imshow("Captured Depth Image", depth_image)

        # Wait for key press to capture the next pose
        key = cv2.waitKey(1)
        if key == 27:  # Press Esc to exit early
            print("Exiting calibration...")
            break
        elif key == ord('q'):  # Press 'q' to capture an image
            print("Capture key 'q' pressed.")
            continue  # Continue to capture the image

    # Perform camera calibration for the RGB camera using OpenCV
    print("Performing camera calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # Display the intrinsic parameters for the RGB camera
    print("RGB Camera Intrinsic Matrix:\n", mtx)
    print("RGB Camera Distortion Coefficients:\n", dist)

    # Save the calibration data to a .yaml file
    calibration_file = r'C:/Users/Sky/PycharmProjects/IntelRealSense/camera_calibration.yaml'

    calibration_data = {
        "camera_matrix": {
            "rows": mtx.shape[0],
            "cols": mtx.shape[1],
            "data": mtx.flatten().tolist(),
        },
        "distortion_coefficients": {
            "rows": dist.shape[0],
            "cols": dist.shape[1],
            "data": dist.flatten().tolist(),
        },
    }

    with open(calibration_file, "w") as f:
        yaml.dump(calibration_data, f, default_flow_style=False)

    print(f"Calibration data saved to: {calibration_file}")

    # Optionally, you can undistort and visualize results here

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Stop the pipeline
    pipeline.stop()


def main():
    calibrate_camera()


if __name__ == "__main__":
    main()
