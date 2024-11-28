import os
import cv2
import numpy as np
import openvino.runtime as ov
from pathlib import Path
import argparse
import time
import subprocess


class VideoUpscaler:
    def __init__(self, model_path, device="CPU"):
        self.model_path = model_path

        # Initialize OpenVINO Core
        self.core = ov.Core()
        self.device_list = self.core.available_devices
        print("Available devices: ", self.device_list)

        # Use the specified device (either CPU or GPU)
        if device in self.device_list:
            self.selected_device = device
            print(f"Using {self.selected_device} for inference.")
        else:
            raise ValueError(f"Specified device {device} is not available. Choose from: {self.device_list}")

        # Load the model
        self.model = self.core.read_model(self.model_path)

        # Reshape the model for dynamic input
        self.model.reshape([1, 3, -1, -1])  # Dynamic height and width

        # Compile the model
        self.compiled_model = self.core.compile_model(self.model, device_name=self.selected_device)

        # Get the input/output names
        self.input_tensor_name = self.compiled_model.input().get_any_name()
        self.output_tensor = self.compiled_model.output()

        # Create an inference request
        self.infer_request = self.compiled_model.create_infer_request()

    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = frame_rgb / 255.0  # Normalize to [0, 1]
        frame_resized = np.transpose(frame_resized, (2, 0, 1))  # Change to channel-first format (C, H, W)
        frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
        return frame_resized

    def run_inference(self, frame):
        preprocessed_frame = self.preprocess_frame(frame)
        result = self.infer_request.infer({self.input_tensor_name: preprocessed_frame})[self.output_tensor]
        return result[0]  # Return the result from the first batch

    def postprocess_output(self, output):
        output = output.squeeze()  # Remove batch dimension
        output = np.clip(output, 0, 1) * 255.0
        output = output.astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))  # Convert back to HWC format
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output_bgr

    def upscale_video(self, video_path, output_video_path, temp_frames_folder="temp_frames"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video at {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video Info: {fps} FPS, {frame_width}x{frame_height}, {total_frames} frames")

        # Create folder to store temporary upscaled frames
        os.makedirs(temp_frames_folder, exist_ok=True)

        # Process each frame and save as an image
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Upscale the frame
            start_time = time.time()
            upscaled_frame = self.run_inference(frame)
            upscaled_frame = self.postprocess_output(upscaled_frame)
            end_time = time.time()

            # Save the frame as an image
            frame_filename = os.path.join(temp_frames_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, upscaled_frame)
            print(f"Processed frame {frame_count}/{total_frames} in {end_time - start_time:.2f} seconds")
            frame_count += 1

        cap.release()

        # Use FFmpeg to combine frames into a video
        ffmpeg_command = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate", str(fps),
            "-i", os.path.join(temp_frames_folder, "frame_%04d.png"),
            "-c:v", "libx264",  # Use H.264 codec
            "-pix_fmt", "yuv420p",  # Ensure compatibility
            output_video_path,
        ]
        print("Combining frames into video with FFmpeg...")
        subprocess.run(ffmpeg_command, check=True)
        print(f"Upscaled video saved to {output_video_path}")

        # Clean up temporary frames
        for file in os.listdir(temp_frames_folder):
            os.remove(os.path.join(temp_frames_folder, file))
        os.rmdir(temp_frames_folder)


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Real-ESRGAN Video Upscaling with OpenVINO")
    parser.add_argument('--model', type=str, required=True, help="Path to the OpenVINO model (.xml file)")
    parser.add_argument('--video', type=str, required=True, help="Path to the video to upscale")
    parser.add_argument('--output', type=str, default="upscaled_video.mp4", help="Path to save the upscaled video")
    parser.add_argument('--device', type=str, default="CPU", help="Device to use for inference (CPU or GPU)")

    args = parser.parse_args()

    # Initialize the upscaler with parsed arguments
    upscaler = VideoUpscaler(model_path=args.model, device=args.device)
    upscaler.upscale_video(args.video, args.output)
