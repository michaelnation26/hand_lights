import cv2
import time
import numpy as np

from collections import deque


MAX_HANDS_DETECTED = 2
LIGHT_DURATION_N_SECS = 0.2
KEYPOINT_DETECTION_PROB_THRESHOLD = 0.5
NET_INPUT_HEIGHT = 368

THUMB_IX = 4
INDEX_FINGER_IX = 8
MIDDLE_FINGER_IX = 12
RING_FINGER_IX = 16
PINKY_FINGER_IX = 20
FINGER_IXS = [THUMB_IX, INDEX_FINGER_IX, MIDDLE_FINGER_IX, RING_FINGER_IX, PINKY_FINGER_IX]

class HandLights():

    def __init__(self, proto_file_path, weights_file_path):
        self.net = cv2.dnn.readNetFromCaffe(proto_file_path, weights_file_path)
        self.finger_coords_queue = None
        self.frame_height = None
        self.frame_width = None
        self.net_input_width = None

    def run_video(self, input_video_path, output_video_path="output_video.mp4"):
        video = cv2.VideoCapture(input_video_path)
        if not video.isOpened():
            raise FileNotFoundError("The input video path you provided is invalid.")

        self.set_member_variable_values(video)

        i = 0
        while video.isOpened():
            grabbed, frame = video.read()
            if not grabbed:
                break

            i += 1

            finger_coords = self.get_all_finger_coords_from_frame(frame)
            self.finger_coords_queue.append(finger_coords)
            
            if i % 10 == 0:
                break

        video.release()

    def set_member_variable_values(self, video):
        light_duration_n_frames = int(LIGHT_DURATION_N_SECS * round(video.get(cv2.CAP_PROP_FPS)))
        print("n frames", light_duration_n_frames)
        self.finger_coords_queue = deque(maxlen=light_duration_n_frames)

        self.frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

        aspect_ratio = self.frame_width / self.frame_height
        self.net_input_width = int(aspect_ratio * NET_INPUT_HEIGHT)

    def run_image(self, input_image_path):
        frame = cv2.imread(input_image_path)
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]

        aspect_ratio = self.frame_width / self.frame_height
        self.net_input_width = int(aspect_ratio * NET_INPUT_HEIGHT)

        all_finger_coords = self.get_all_finger_coords_from_frame(frame)
        self.draw_finger_coords(frame, all_finger_coords)

    def get_all_finger_coords_from_frame(self, frame):
        all_finger_coords = []

        input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.net_input_width, NET_INPUT_HEIGHT),
                                           (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(input_blob)
        prediction = self.net.forward()

        for finger_ix in FINGER_IXS:
            probability_map = prediction[0, finger_ix, :, :]

            finger_coords = self.get_single_finger_coords(probability_map)
            all_finger_coords.append(finger_coords)

        return all_finger_coords

    def get_single_finger_coords(self, probability_map):
        """
        Returns up to MAX_HANDS_DETECTED pairs of coordinates for a single type of finger.
        e.g. If MAX_HANDS_DETECTED is 2 and if there are 2 instances of a ring finger in an image, the coordinate points
        for each instance will be returned.
        """
        # The resize ratio values will be used to rescale the coordinate points
        # from the probability maps to the original image frame.
        prob_map_height_ratio = self.frame_height / probability_map.shape[0]
        prob_map_width_ratio = self.frame_width / probability_map.shape[1]

        finger_coords = []

        for _ in range(MAX_HANDS_DETECTED):
            probability, coordinate = self.get_max_prob_coordinate(probability_map)
            if probability < KEYPOINT_DETECTION_PROB_THRESHOLD:
                break

            coord_resized = self.resize_coordinate(coordinate, prob_map_height_ratio, prob_map_width_ratio)
            finger_coords.append(coord_resized)

            # Remove coordinate from the probability map so that it is not retrieved again.
            self.set_region_to_zero(probability_map, coordinate)

        return finger_coords

    def get_single_finger_coords2(self, probability_map):
        """
        Returns up to two pairs of coordinates for a single type of finger.
        e.g. If there are two instances of a ring finger in an image, the coordinate points
        for each instance will be returned.
        """
        # The resize ratio values will be used to rescale the coordinate points
        # from the probability maps to the original image frame.
        prob_map_height_ratio = self.frame_height / probability_map.shape[0]
        prob_map_width_ratio = self.frame_width / probability_map.shape[1]

        finger_coords = []

        prob_0, coord_0 = self.get_max_prob_coordinate(probability_map)
        if prob_0 < KEYPOINT_DETECTION_PROB_THRESHOLD:
            return finger_coords

        coord_0_resized = self.resize_coordinate(coord_0, prob_map_height_ratio, prob_map_width_ratio)
        finger_coords.append(coord_0_resized)
        # Remove prob_0 coordinates from the probability map so that it is not retrieved again.
        self.set_region_to_zero(probability_map, coord_0)

        prob_1, coord_1 = self.get_max_prob_coordinate(probability_map)
        if prob_1 < KEYPOINT_DETECTION_PROB_THRESHOLD:
            return finger_coords

        coord_1_resized = self.resize_coordinate(coord_1, prob_map_width_ratio, prob_map_height_ratio)
        finger_coords.append(coord_1_resized)

        return finger_coords

    def get_max_prob_coordinate(self, probability_map):
        probability = np.max(probability_map)
        coord = np.unravel_index(probability_map.argmax(), probability_map.shape)

        return probability, coord

    def resize_coordinate(self, coordinate, height_ratio, width_ratio):
        coord_y, coord_x = coordinate
        coord_resized_y = int(coord_y * height_ratio)
        coord_resized_x = int(coord_x * width_ratio)

        return coord_resized_y, coord_resized_x

    def set_region_to_zero(self, probability_map, coordinate, region_radius=2):
        coord_y, coord_x = coordinate
        y0 = coord_y - region_radius
        y1 = coord_y + region_radius
        x0 = coord_x - region_radius
        x1 = coord_x + region_radius

        probability_map[y0:y1, x0:x1] = 0.0

    def draw_finger_coords(self, frame, all_finger_coords):
        frame_copy = np.copy(frame)
        for single_finger_coords, finger_ix in zip(all_finger_coords, FINGER_IXS):
            for finger_coord in single_finger_coords:
                self.draw_point(frame_copy, finger_coord, finger_ix)

        cv2.imwrite('output.jpg', frame_copy)

    def draw_point(self, frame, coordinate, finger_ix):
        coord_y, coord_x = coordinate
        cv2.circle(frame, (coord_x, coord_y), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        # cv2.putText(frame, "{}".format(finger_ix), (coord_x, coord_y), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 255), 2, lineType=cv2.LINE_AA)

if __name__== "__main__":
    PROTO_FILE_PATH = "caffe_model/pose_deploy.prototxt"
    WEIGHTS_FILE_PATH = "caffe_model/pose_iter_102000.caffemodel"
    INPUT_IMAGE_PATH = "media/images/front_back.jpg"
    INPUT_VIDEO_PATH = "media/videos/sign_language.mp4"

    hand_lights = HandLights(PROTO_FILE_PATH, WEIGHTS_FILE_PATH)

    t = time.time()

    #hand_lights.run_image(INPUT_IMAGE_PATH)
    hand_lights.run_video(INPUT_VIDEO_PATH, output_video_path="output_video.mp4")

    print("time taken by network : {:.2f}".format(time.time() - t))
