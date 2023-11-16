import cv2
import numpy as np

from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig


def extract_left_frame(frame, input_width, input_height, image_width, image_height):
    input_half_width = int(input_width/2)
    frame_left =  frame[0:input_height, 0:input_half_width] # extract left side 
    frame_left =cv2.resize(frame_left, (1*image_width, 1*image_height), interpolation = cv2.INTER_CUBIC)
    return frame_left

def extract_right_frame(frame, input_width, input_height, image_width, image_height):
    input_half_width = int(input_width/2)
    frame_right =  frame[0:input_height, input_half_width:input_width] # extract left side 
    frame_right =cv2.resize(frame_right, (1*image_width, 1*image_height), interpolation = cv2.INTER_CUBIC)
    return frame_right

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

def compute_for_video():
	url = "UX3D.Avatar3D.SBS.mp4"
	cap = cv2.VideoCapture(url)


	# Select model type
	model_type = ModelType.middlebury
	# model_type = ModelType.flyingthings
	# model_type = ModelType.eth3d

	if model_type == ModelType.middlebury:
		model_path = "models/middlebury_d400.pb"
	elif model_type == ModelType.flyingthings:
		model_path = "models/flyingthings_finalpass_xl.pb"
	elif model_type == ModelType.eth3d:
		model_path = "models/eth3d.pb"

	video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )  # float `width`
	video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
	video_fps = int(cap.get(cv2.CAP_PROP_FPS))
	video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# Store baseline (m) and focal length (pixel)
	camera_config = CameraConfig(0.546, 1000)
	max_distance = 5

	# Initialize model
	hitnet_depth = HitNet(model_path, model_type, camera_config)


	image_width = 1920
	image_height = 1080
	##### video loop ####
	begin_frame_index= 0
	end_frame_index = video_frame_count # or custom
	frame_index = 0 

	cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
	while cap.isOpened():

		try:
			# Read frame from the video
			ret, frame = cap.read()
			if not ret:	
				break
		except:
			continue

		if frame_index < begin_frame_index:
			frame_index+=1
			continue

		if not ret or frame is None:
			print("Can't receive frame (stream end?). Exiting ...")
			break
		if frame is None:
			print("Can't receive frame (stream end?). Exiting ...")
			break

		# Extract the left and right images
		left_img  = extract_left_frame( frame, video_width, video_height, image_width, image_height) 
		right_img = extract_right_frame( frame, video_width, video_height, image_width, image_height) 
		color_real_depth = frame[:,frame.shape[1]*2//3:]

		# Estimate the depth
		disparity_map = hitnet_depth(left_img, right_img)
		depth_map = hitnet_depth.get_depth()

		color_disparity = draw_disparity(disparity_map)
		color_depth = draw_depth(depth_map, max_distance)
		cobined_image = np.hstack((left_img, color_depth, color_disparity))

		cv2.imshow("Estimated depth", cobined_image)

		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	compute_for_video()