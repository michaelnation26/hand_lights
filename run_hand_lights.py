import cv2
import time
import numpy as np

from collections import deque


MAX_HANDS_DETECTED = 2
LIGHT_DURATION_N_SECS = 0.5
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

    def run_video(self, input_video_path, output_video_path="output_video.mp4"):
        input_video = cv2.VideoCapture(input_video_path)
        if not input_video.isOpened():
            raise FileNotFoundError("The input video path you provided is invalid.")

        video_fps, frame_height, frame_width = self._get_video_properties(input_video)
        light_duration_n_frames = int(LIGHT_DURATION_N_SECS * video_fps)
        frame_finger_coords_queue = deque(maxlen=light_duration_n_frames)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(output_video_path, fourcc, video_fps,  (frame_width, frame_height))

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

            # if i % 10 == 0:
            #     break

        input_video.release()
        output_video.release()

    def run_image(self, input_image_path):
        frame = cv2.imread(input_image_path)
        all_finger_coords = self._get_all_finger_coords_from_frame(frame)
        self._draw_finger_coords(frame, all_finger_coords)

    def _draw_finger_coords(self, frame, all_finger_coords):
        frame_copy = np.copy(frame)
        for single_finger_coords, finger_ix in zip(all_finger_coords, FINGER_IXS):
            for finger_coord in single_finger_coords:
                self._draw_point(frame_copy, finger_coord, finger_ix)

        cv2.imwrite('output.jpg', frame_copy)

    def _draw_lights_on_frame_using_coords_queue(self, frame, frame_finger_coords_queue):
        for frame_finger_coords in frame_finger_coords_queue:
            frame = self._draw_lights_on_frame_using_single_frame_coords(frame, frame_finger_coords)

        return  frame

    def _draw_lights_on_frame_using_single_frame_coords(self, frame, frame_finger_coords):
        for single_finger_coords in frame_finger_coords:
            for finger_coord in single_finger_coords:
                frame = self._set_region_to_value(frame, finger_coord, value=[255, 255, 255], region_radius=8)

        return frame

    def _draw_point(self, frame, coordinate, finger_ix=0):
        coord_y, coord_x = coordinate
        cv2.circle(frame, (coord_x, coord_y), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        # cv2.putText(frame, "{}".format(finger_ix), (coord_x, coord_y), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 0, 255), 2, lineType=cv2.LINE_AA)

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
            probability_map = self._set_region_to_value(probability_map, coordinate, value=0.0)

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

    def _set_region_to_value(self, np_array, coordinate, value, region_radius=2):
        np_array_copy = np.copy(np_array)

        height = np_array.shape[0] - 1
        width = np_array.shape[1] - 1

        coord_y, coord_x = coordinate
        y0 = max(coord_y - region_radius, 0)
        y1 = min(coord_y + region_radius, height)
        x0 = max(coord_x - region_radius, 0)
        x1 = min(coord_x + region_radius, width)

        np_array_copy[y0:y1, x0:x1] = value

        return np_array_copy


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