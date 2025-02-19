import pyrealsense2 as rs
import numpy as np
import cv2

def get_depth():
    # Set up RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the pipeline
    pipeline.start(config)

    try:
        # Capture a single frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Show color and depth images
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_image)

        # Get the depth at a specific pixel (for example, at pixel (x=320, y=240))
        depth_in_mm = depth_image[240, 320]  # Depth in millimeters at the center pixel
        print(f"Depth at pixel (320, 240): {depth_in_mm} mm")

        # Optionally, you can apply a colormap to the depth image for better visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("Depth Colormap", depth_colormap)

        cv2.waitKey(0)

    finally:
        # Stop the pipeline
        pipeline.stop()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_depth()
