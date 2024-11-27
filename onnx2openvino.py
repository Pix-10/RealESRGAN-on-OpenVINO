import openvino as ov
onnx_path = 'real_esrgan_x4.onnx'
ov_model = ov.convert_model(onnx_path)
ov.save_model(ov_model, onnx_path.split(".")[0]+ '.xml')