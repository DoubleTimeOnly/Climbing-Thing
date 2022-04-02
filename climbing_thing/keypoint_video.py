import os

import cv2
from climbing_thing.climbernet.climbernet import ClimberNet
from climbing_thing.climbernet.utils import visualizer as viz
from climbing_thing.utils.image import imshow


def show_keypoints(video_path):
    video = cv2.VideoCapture(video_path)
    model = ClimberNet(model_path=None, device="cuda")

    while(video.isOpened()):
        ret, image = video.read()
        image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2)

        # do climbernet stuff
        if ret:
            instances = model(image)
            output_image = viz.draw_instance_predictions(
                image, instances, model.predictor.metadata, scale=1
            )
            cv2.imshow("output", output_image)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
        else:
            break

if __name__ == '__main__':
    video_path = r"data\videos\20220207_211710.mp4"
    if not os.path.exists(video_path):
        raise ValueError(f"Could not find {os.path.abspath(video_path)}")
    show_keypoints(video_path)