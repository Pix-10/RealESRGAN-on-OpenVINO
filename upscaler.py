import os
import cv2
import numpy as np
import openvino.runtime as ov
from pathlib import Path
import time
import argparse

class Upscale_rESRGAN_Dynamic(object):
    def __init__(self, model_path, output_folder="output/", device="CPU"):
        self.model_path = model_path
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        
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

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot open image at {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = image_rgb / 255.0  # Normalize to [0, 1]
        image_resized = np.transpose(image_resized, (2, 0, 1))  # Change to channel-first format (C, H, W)
        image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
        
        return image_resized

    def run_inference(self, image_path):
        preprocessed_image = self.preprocess_image(image_path)
        
        # Perform inference
        start_time = time.time()
        result = self.infer_request.infer({self.input_tensor_name: preprocessed_image})[self.output_tensor]
        end_time = time.time()

        print(f"Inference completed in {end_time - start_time:.2f} seconds")
        return result[0]  # Return the result from the first batch

    def postprocess_output(self, output):
        output = output.squeeze()  # Remove batch dimension
        output = np.clip(output, 0, 1) * 255.0
        output = output.astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))  # Convert back to HWC format
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output_bgr

    def save_image(self, image, input_image_path):
        filename = os.path.basename(input_image_path)
        image_name = os.path.splitext(filename)[0] + "_upscaled.png"
        output_path = os.path.join(self.output_folder, image_name)
        cv2.imwrite(output_path, image)
        print(f"Upscaled image saved to {output_path}")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Real-ESRGAN Upscaling with OpenVINO")
    parser.add_argument('--model', type=str, required=True, help="Path to the OpenVINO model (.xml file)")
    parser.add_argument('--image', type=str, required=True, help="Path to the image to upscale")
    parser.add_argument('--output', type=str, default="output/", help="Folder to save the upscaled image")
    parser.add_argument('--device', type=str, default="CPU", help="Device to use for inference (CPU or GPU)")

    args = parser.parse_args()

    # Initialize the upscaler with parsed arguments
    upscaler = Upscale_rESRGAN_Dynamic(model_path=args.model, output_folder=args.output, device=args.device)
    result = upscaler.run_inference(args.image)
    upscaled_image = upscaler.postprocess_output(result)
    upscaler.save_image(upscaled_image, args.image)
