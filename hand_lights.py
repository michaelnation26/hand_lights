# Copyright (c) 2018 Michael Nation
# Licensed under the MIT License.
import cv2
import numpy as np

from collections import deque


MAX_HANDS_DETECTED = 2
LIGHT_DURATION_N_SECS = 0.5
KEYPOINT_DETECTION_PROB_THRESHOLD = 0.5
NET_INPUT_HEIGHT = 368

LIGHT_RADIUS_FRAME_HEIGHT_RATIO = 0.02
LIGHT_MAX_ALPHA = 1.0
LIGHT_GAUSSIAN_BLUR_K_SIZE = (17, 17)

BGR_GREEN = [57, 255, 20]

THUMB_IX = 4
INDEX_FINGER_IX = 8
MIDDLE_FINGER_IX = 12
RING_FINGER_IX = 16
PINKY_FINGER_IX = 20

class HandLights():

    def __init__(self, proto_file_path, weights_file_path):
        self.net = cv2.dnn.readNetFromCaffe(proto_file_path, weights_file_path)

    def run_video(self, input_video_path, output_video_path="output_video.mp4", fingers="all"):
        input_video = cv2.VideoCapture(input_video_path)
        if not input_video.isOpened():
            raise FileNotFoundError("The input video path you provided is invalid.")

        video_fps, frame_h, frame_w = self._get_video_properties(input_video)
        light_duration_n_frames = int(LIGHT_DURATION_N_SECS * video_fps)
        frame_finger_coords_queue = deque(maxlen=light_duration_n_frames)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(output_video_path, fourcc, video_fps,  (frame_w, frame_h))

        finger_ixs = self._get_finger_ixs(fingers)

        i = 0
        while input_video.isOpened():
            grabbed, frame = input_video.read()
            if not grabbed:
                break

            i += 1

            finger_coords = self._get_all_finger_coords_from_frame(frame, finger_ixs)
            frame_finger_coords_queue.append(finger_coords)

            frame_drawn = self._draw_lights_on_frame_using_coords_queue(frame, frame_finger_coords_queue)
            output_video.write(frame_drawn)

            if i % 5 == 0:
                print("i", i)

            if i % 5 == 0:
                break

        input_video.release()
        output_video.release()

    def run_image(self, input_image_path, fingers="all"):
        frame = cv2.imread(input_image_path)
        finger_ixs = self._get_finger_ixs(fingers)
        all_finger_coords = self._get_all_finger_coords_from_frame(frame, finger_ixs)
        frame_drawn = self._draw_lights_on_frame_using_single_frame_coords(frame, all_finger_coords)
        cv2.imwrite("media/images/output_img.jpg", frame_drawn)

    def _blur_lights(self, frame, lights_mask):
        """
        A Gaussian Blur is applied to all of the lights to smoothen out the lines that are created
        from overlapping lights.
        """
        frame_copy = np.copy(frame)
        frame_blurred = cv2.GaussianBlur(frame, LIGHT_GAUSSIAN_BLUR_K_SIZE, 0)
        frame_copy[lights_mask] = frame_blurred[lights_mask]

        return frame_copy


    def _draw_lights_on_frame_using_coords_queue(self, frame, frame_finger_coords_queue):
        queue_size = len(frame_finger_coords_queue)
        alphas = np.linspace(0.0, LIGHT_MAX_ALPHA, num=queue_size+1)[1:] # skip first alpha value (0.0)
        masks_merged_all_frames = np.full(frame.shape[:2], fill_value=False, dtype=bool)

        for frame_finger_coords, alpha in zip(frame_finger_coords_queue, alphas):
            frame, masks_merged = self._draw_lights_on_frame_using_single_frame_coords(frame, frame_finger_coords, alpha)
            masks_merged_all_frames = np.logical_or(masks_merged_all_frames, masks_merged)

        frame = self._blur_lights(frame, masks_merged_all_frames)

        return  frame

    def _draw_lights_on_frame_using_single_frame_coords(self, frame, frame_finger_coords, alpha=LIGHT_MAX_ALPHA):
        """
        Using the finger coordinates of a single frame, lights are drawn in a circular shape with the alpha parameter.
        The circular mask for each finger coordinate is saved into a single mask called masks_merged.
        """
        frame_with_lights = np.copy(frame)
        light_radius = LIGHT_RADIUS_FRAME_HEIGHT_RATIO * frame.shape[0]
        masks_merged = np.full(frame.shape[:2], fill_value=False, dtype=bool)

        for single_finger_coords in frame_finger_coords:
            for finger_coord in single_finger_coords:
                mask = self._get_circular_mask(frame_with_lights, radius=light_radius, coordinate=finger_coord)
                frame_with_lights[mask] = BGR_GREEN

                masks_merged = np.logical_or(masks_merged, mask)

        frame_with_transparent_lights = cv2.addWeighted(frame_with_lights, alpha, frame, 1.0-alpha, 0)

        return frame_with_transparent_lights, masks_merged

    def _get_finger_ixs(self, fingers):
        if fingers == "index":
            finger_ixs = [INDEX_FINGER_IX]
        else: # all
            finger_ixs = [THUMB_IX, INDEX_FINGER_IX, MIDDLE_FINGER_IX, RING_FINGER_IX, PINKY_FINGER_IX]

        return finger_ixs

    def _get_all_finger_coords_from_frame(self, frame, finger_ixs):
        all_finger_coords = []

        frame_height, frame_width = frame.shape[0], frame.shape[1]
        aspect_ratio = frame_width / frame_height
        net_input_width = int(aspect_ratio * NET_INPUT_HEIGHT)

        input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (net_input_width, NET_INPUT_HEIGHT),
                                           (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(input_blob)
        prediction = self.net.forward()

        for finger_ix in finger_ixs:
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