import pyrealsense2 as rs

# Start a pipeline
pipeline = rs.pipeline()
config = rs.config()

# Set resolution to 640x480
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)

# Get intrinsics of the depth stream
depth_stream = profile.get_stream(rs.stream.depth)
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

# Print focal lengths (fx, fy) in pixels
print(f"Focal length (fx): {intrinsics.fx} pixels")
print(f"Focal length (fy): {intrinsics.fy} pixels")

pipeline.stop()
