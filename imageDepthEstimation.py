import cv2
import numpy as np
from pathlib import Path

from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img

# Select model type
#model_type = ModelType.middlebury
#model_type = ModelType.flyingthings
model_type = ModelType.eth3d

if model_type == ModelType.middlebury:
	model_path = "models/middlebury_d400.pb"
elif model_type == ModelType.flyingthings:
	model_path = "models/flyingthings_finalpass_xl.pb"
elif model_type == ModelType.eth3d:
	model_path = "models/eth3d.pb"


# Initialize model
hitnet_depth = HitNet(model_path, model_type)

input_folder = Path("./data")

output_folder = Path("./result")

output_folder.mkdir(parents=True, exist_ok=True)

image_list_left = sorted((input_folder/"left").glob("*.png"))
image_list_right = sorted((input_folder/"right").glob("*.png"))

for x in range(0, len(image_list_left)):

	# Load images
	left_img = open(image_list_left[x], "rb")
	left_img = np.asarray(bytearray(left_img.read()), dtype=np.uint8)
	left_img = cv2.imdecode(left_img, -1) # 'Load it as it is'
	right_img = open(image_list_right[x], "rb")
	right_img = np.asarray(bytearray(right_img.read()), dtype=np.uint8)
	right_img = cv2.imdecode(right_img, -1) # 'Load it as it is'

	# Estimate the depth
	disparity_map = hitnet_depth(left_img, right_img)

	color_disparity = draw_disparity(disparity_map)

	cv2.imwrite(str(output_folder / image_list_left[x].name), color_disparity)
