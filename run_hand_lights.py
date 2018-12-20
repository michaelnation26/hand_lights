import cv2
import time
import numpy as np

def draw_point(frame, coord_x, coord_y, finger_ix):
	cv2.circle(frame, (coord_x, coord_y), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
	cv2.putText(frame, "{}".format(finger_ix), (coord_x, coord_y), cv2.FONT_HERSHEY_SIMPLEX, 1, 
				(0, 0, 255), 2, lineType=cv2.LINE_AA)

def set_region_to_zero(probability_map, coord_x, coord_y, region_radius=2):
	x0 = coord_x - region_radius
	x1 = coord_x + region_radius
	y0 = coord_y - region_radius
	y1 = coord_y + region_radius
	
	probability_map[y0:y1, x0:x1] = 0.0

PROTO_FILE = "caffe_model/pose_deploy.prototxt"
WEIGHTS_FILE = "caffe_model/pose_iter_102000.caffemodel"
KEYPOINT_DETECTION_PROB_THRESHOLD = 0.5
NET_INPUT_HEIGHT = 368

THUMB_IX = 4
INDEX_FINGER_IX = 8
MIDDLE_FINGER_IX = 12
RING_FINGER_IX = 16
PINKY_FINGER_IX = 20
FINGER_IXS = [THUMB_IX, INDEX_FINGER_IX, MIDDLE_FINGER_IX, RING_FINGER_IX, PINKY_FINGER_IX]

net = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)

frame = cv2.imread("media/images/front-back.jpg")
frame_copy = np.copy(frame)
frame_width = frame.shape[1]
frame_height = frame.shape[0]
aspect_ratio = frame_width / frame_height

net_input_width = int(aspect_ratio * NET_INPUT_HEIGHT)
input_blob = cv2.dnn.blobFromImage(frame, 1.0/255, (net_input_width, NET_INPUT_HEIGHT), \
								   (0, 0, 0), swapRB=False, crop=False)
net.setInput(input_blob)

t = time.time()
prediction = net.forward()
print("FORWARD time taken by network : {:.3f}".format(time.time() - t))

# The resize ratio values will be used to rescale the coordinate points
# from the probability maps to the original image frame.
probability_map = prediction[0, 0, :, :]
resize_height_ratio = frame_height / probability_map.shape[0]
resize_width_ratio = frame_width / probability_map.shape[1]

for FINGER_IX in FINGER_IXS:
	probability_map = prediction[0, FINGER_IX, :, :]
	
	prob_1 = np.max(probability_map)
	if prob_1 < KEYPOINT_DETECTION_PROB_THRESHOLD:
		continue
		
	coord_1_prob_map = np.unravel_index(probability_map.argmax(), probability_map.shape)
	coord_1_y = int(coord_1_prob_map[0] * resize_width_ratio)
	coord_1_x = int(coord_1_prob_map[1] * resize_height_ratio)
	draw_point(frame_copy, coord_1_x, coord_1_y, FINGER_IX)
	
	# Remove prob_1 from the probability map so that it is not retrieved again.
	set_region_to_zero(probability_map, coord_x=coord_1_prob_map[1], coord_y=coord_1_prob_map[0])
	
	prob_2 = np.max(probability_map)
	if prob_2 < KEYPOINT_DETECTION_PROB_THRESHOLD:
		continue
	
	coord_2_prob_map = np.unravel_index(probability_map.argmax(), probability_map.shape)
	coord_2_y = int(coord_2_prob_map[0] * resize_width_ratio)
	coord_2_x = int(coord_2_prob_map[1] * resize_height_ratio)
	draw_point(frame_copy, coord_2_x, coord_2_y, FINGER_IX)
	
	print("PROB", prob_1, prob_2)
	print("LOC", coord_1_prob_map, coord_2_prob_map, "\n")
		


cv2.imwrite('Output-Keypoints.jpg', frame_copy)

print("Total time taken : {:.3f}".format(time.time() - t))
