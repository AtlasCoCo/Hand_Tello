#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
 @version: V0.1
 @license: MIT Licence
 @author: danquxunhuan
 @mail: danquxunhuan130@gmail.com
 @project: tello
 @file: test.py
 @create time: 18-12-30 11:50
 @description:
 """

import cv2
from config import FLAGS
import tensorflow as tf
from telloCV import TelloCV
from detector import Detector


def main(argv):
    tellotrack = TelloCV()

    for packet in tellotrack.container.demux((tellotrack.vid_stream,)):
        for frame in packet.decode():
            image = tellotrack.process_frame(frame)
            cv2.imshow('tello', image)
            # print(image.shape)
            if cv2.waitKey(1) == 27:
                if tellotrack.record:
                    tellotrack.toggle_recording(1)
                tellotrack.drone.quit()


if __name__ == '__main__':
    tf.app.run()

