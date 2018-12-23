import cv2
import time
import numpy as np

from collections import deque


MAX_HANDS_DETECTED = 2
LIGHT_DURATION_N_SECS = 0.5
KEYPOINT_DETECTION_PROB_THRESHOLD = 0.5
NET_INPUT_HEIGHT = 368
LIGHT_RADIUS_FRAME_HEIGHT_RATIO = 0.02

THUMB_IX = 4
INDEX_FINGER_IX = 8
MIDDLE_FINGER_IX = 12
RING_FINGER_IX = 16
PINKY_FINGER_IX = 20
FINGER_IXS = [THUMB_IX, INDEX_FINGER_IX, MIDDLE_FINGER_IX, RING_FINGER_IX, PINKY_FINGER_IX]

class HandLights():

    def __init__(self, proto_file_path, weights_file_path):
        self.net = cv2.dnn.readNetFromCaffe(proto_file_path, weights_file_path)

    def run_video(self, input_video_path, output_video_path="output_video.mp4"):
        input_video = cv2.VideoCapture(input_video_path)
        if not input_video.isOpened():
            raise FileNotFoundError("The input video path you provided is invalid.")

        video_fps, frame_h, frame_w = self._get_video_properties(input_video)
        light_duration_n_frames = int(LIGHT_DURATION_N_SECS * video_fps)
        frame_finger_coords_queue = deque(maxlen=light_duration_n_frames)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(output_video_path, fourcc, video_fps,  (frame_w, frame_h))

        i = 0
        while input_video.isOpened():
            grabbed, frame = input_video.read()
            if not grabbed:
                break

            i += 1

            finger_coords = self._get_all_finger_coords_from_frame(frame)
            frame_finger_coords_queue.append(finger_coords)

            frame_drawn = self._draw_lights_on_frame_using_coords_queue(frame, frame_finger_coords_queue)
            output_video.write(frame_drawn)

            if i % 5 == 0:
                print("i", i)

            if i % 5 == 0:
                break

        input_video.release()
        output_video.release()

    def run_image(self, input_image_path):
        frame = cv2.imread(input_image_path)
        all_finger_coords = self._get_all_finger_coords_from_frame(frame)

    def _draw_lights_on_frame_using_coords_queue(self, frame, frame_finger_coords_queue):
        for frame_finger_coords in frame_finger_coords_queue:
            frame = self._draw_lights_on_frame_using_single_frame_coords(frame, frame_finger_coords)

        return  frame

    def _draw_lights_on_frame_using_single_frame_coords(self, frame, frame_finger_coords):
        frame_copy = np.copy(frame)
        light_radius = LIGHT_RADIUS_FRAME_HEIGHT_RATIO * frame.shape[0]

        for single_finger_coords in frame_finger_coords:
            for finger_coord in single_finger_coords:
                mask = self._get_circular_mask(frame_copy, radius=light_radius, coordinate=finger_coord)
                frame_copy[mask] = [255, 255, 255]

        return frame_copy

    def _get_all_finger_coords_from_frame(self, frame):
        all_finger_coords = []

        frame_height, frame_width = frame.shape[0], frame.shape[1]
        aspect_ratio = frame_width / frame_height
        net_input_width = int(aspect_ratio * NET_INPUT_HEIGHT)

        input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (net_input_width, NET_INPUT_HEIGHT),
                                           (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(input_blob)
        prediction = self.net.forward()

        for finger_ix in FINGER_IXS:
            probability_map = prediction[0, finger_ix, :, :]

            finger_coords = self._get_single_finger_coords(probability_map, frame_height, frame_width)
            all_finger_coords.append(finger_coords)

        return all_finger_coords

    def _get_circular_mask(self, frame, radius, coordinate):
        img_h, img_w = frame.shape[0], frame.shape[1]
        center_y, center_x = coordinate

        y, x = np.ogrid[-center_y: img_h - center_y, -center_x: img_w - center_x]
        circular_mask = x * x + y * y <= radius * radius

        return circular_mask

    def _get_max_prob_coordinate(self, probability_map):
        probability = np.max(probability_map)
        coord = np.unravel_index(probability_map.argmax(), probability_map.shape)

        return probability, coord

    def _get_single_finger_coords(self, probability_map, frame_height, frame_width):
        """
        Returns up to MAX_HANDS_DETECTED pairs of coordinates for a single type of finger.
        e.g. If MAX_HANDS_DETECTED is 2 and if there are 2 instances of a ring finger in an image, the coordinate points
        for each instance will be returned.
        """
        # The resize ratio values will be used to rescale the coordinate points
        # from the probability maps to the original image frame.
        prob_map_height_ratio = frame_height / probability_map.shape[0]
        prob_map_width_ratio = frame_width / probability_map.shape[1]

        finger_coords = []

        for _ in range(MAX_HANDS_DETECTED):
            probability, coordinate = self._get_max_prob_coordinate(probability_map)
            if probability < KEYPOINT_DETECTION_PROB_THRESHOLD:
                break

            coord_resized = self._resize_coordinate(coordinate, prob_map_height_ratio, prob_map_width_ratio)
            finger_coords.append(coord_resized)

            # Remove coordinate from the probability map so that it is not retrieved again.
            mask = self._get_circular_mask(probability_map, radius=3, coordinate=coordinate)
            probability_map[mask] = 0.0

        return finger_coords

    def _get_video_properties(self, video):
        video_fps = round(video.get(cv2.CAP_PROP_FPS))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

        return video_fps, frame_height, frame_width

    def _resize_coordinate(self, coordinate, height_ratio, width_ratio):
        coord_y, coord_x = coordinate
        coord_resized_y = int(coord_y * height_ratio)
        coord_resized_x = int(coord_x * width_ratio)

        return coord_resized_y, coord_resized_x


if __name__== "__main__":
    PROTO_FILE_PATH = "caffe_model/pose_deploy.prototxt"
    WEIGHTS_FILE_PATH = "caffe_model/pose_iter_102000.caffemodel"
    INPUT_VIDEO_PATH = "media/videos/sign_language.mp4"
    OUTPUT_VIDEO_PATH = "media/videos/output_video.mp4"

    hand_lights = HandLights(PROTO_FILE_PATH, WEIGHTS_FILE_PATH)
    t = time.time()
    hand_lights.run_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)

    total_minutes = (time.time() - t) / 60.0
    print("Total Time (mins): {:.2f}".format(total_minutes))