import detector
import cv2
from config import FLAGS
import tensorflow as tf


def main(argv):

    cam = cv2.VideoCapture(FLAGS.cam_id)
    hand_detector = detector.Detector()
    # use model
    while True:
        # Prepare input image
        _, full_img = cam.read()
        hand_detector.detect(full_img)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    tf.app.run()

