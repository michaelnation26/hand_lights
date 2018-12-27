# Copyright (c) 2018 Michael Nation
# Licensed under the MIT License.
"""
Driver program for the HandLights class.

python run_hand_lights.py \
--input_video_path media/videos/sign_language.mp4 \
--output_video_path media/videos/output_video.mp4 \
--fingers all \
--light_color yellow \
--mirror True \
--verbose True

"""
from hand_lights import HandLights

import argparse


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Draws virtual lights using bare hands.')
    parser.add_argument("--proto_file_path", type=str, default="caffe_model/pose_deploy.prototxt",
                        help='Path to the Caffe Pose prototxt file.')
    parser.add_argument("--weights_file_path", type=str, default="caffe_model/pose_iter_102000.caffemodel",
                        help='Path to the Caffe Pose model.')
    parser.add_argument("--input_video_path", type=str, required=True,
                        help='Path to the video that will be processed. e.g. media/videos/my_video.mp4')
    parser.add_argument("--output_video_path", type=str, required=True,
                        help='Where to save the processed video. e.g. media/videos/my_video_output.mp4')
    parser.add_argument("--fingers", type=str, default="all", choices=["all", "index"],
                        help='Specifies which finger(s) will be detected in the video.')
    parser.add_argument("--light_color", type=str, default="green",
                        choices=["all", "blue", "green", "red", "white", "yellow"],
                        help="The color of the lights that are drawn. 'all' should only be used when fingers=all.")
    parser.add_argument("--light_duration_n_secs", type=float, default=0.2,
                        help="How long a light will be visible for.")
    parser.add_argument("--mirror", type=bool, default=False,
                        help="Each frame in the output video will be a mirror image.")
    parser.add_argument("--verbose", type=bool, default=False,
                        help='Prints processing status information to console.')

    args = parser.parse_args()

    hand_lights = HandLights(args.proto_file_path, args.weights_file_path)


    #INPUT_IMAGE_PATH = "media/images/front_back.jpg"
    #hand_lights.run_image(INPUT_IMAGE_PATH)
    hand_lights.run_video(args.input_video_path, args.output_video_path, args.fingers, args.light_color,
                          args.light_duration_n_secs, args.mirror, args.verbose)