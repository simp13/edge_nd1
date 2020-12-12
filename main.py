"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network, preprocess_image, draw_bboxes
import logging


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Path to image or video file")
    parser.add_argument("-w","--webcam",type=int,help="Webcam number to use")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client



def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network(args.model,args.device)
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model()

    ### TODO: Handle the input stream ###
    if args.webcam != None:
        vc = cv2.VideoCapture(args.webcam)
    else:
        # handle the video or image with -i resources/image_0100.jpeg or .mp4
        vc = cv2.VideoCapture(args.input)

    if not vc.isOpened():
        logging.error(f"Error opening input file (video or image {args.input})")
        exit(1)
    
    does_got_frame,frame = vc.read()

    # last_count = 0
    
    # predict_time_count = 0

    person_in_frame = False
    real_count = 0
    total_count = 0
    input_shape = infer_network.get_input_shape()
    
    while does_got_frame:
        image = preprocess_image(frame,input_shape[3],input_shape[2])

        infer_request_handle = infer_network.async_exec_net(image)
        detections = infer_network.wait(infer_request_handle)
        detections = infer_network.get_output(detections)
        current_count = detections['num_detections']
        # predict_time_count += 1

        # if current_count > last_count and last_count == 0:
        #     start_time = vc.get(cv2.CAP_PROP_POS_MSEC)
        #     total_count = total_count + current_count - last_count
        
        # if current_count < last_count and current_count == 0 and predict_time_count >= 3:
        #     # Person duration in the video is calculated
        #     duration = int((vc.get(cv2.CAP_PROP_POS_MSEC) - start_time) / 1000.0)
        #     # Publish messages to the MQTT server
        #     client.publish("person/duration",
        #                    json.dumps({"duration": duration}))
        
        # when person exit frame
        if current_count == 0:
            if person_in_frame:
                miss_count += 1
                if miss_count > 20:
                    real_count -= 1
                    miss_count = 0
                    duration = int(time.time() - start_time)
                    client.publish("person/duration",json.dumps({"duration": duration}))
                    person_in_frame = False 
        else:
            miss_count = 0
            if real_count == 0:
                real_count += 1
                total_count += 1
                start_time = time.time()
                person_in_frame = True 
                client.publish("person", json.dumps({"total": total_count}))


        # if predict_time_count >= 5:
        #     last_count = current_count
        #     predict_time_count = 0

        client.publish("person", json.dumps({"count": real_count}))

        ### Draw bounding boxes to provide intuition ###
        img = draw_bboxes(frame, detections)
        cv2.putText(img,
            f'current: {real_count} total: {total_count}',
            (0, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5,
            (255,255,255),
            2,
            cv2.LINE_AA)
        sys.stdout.buffer.write(img)
        sys.stdout.flush()

        ### Write an output image if `single_image_mode` ###
        if vc.get(cv2.CAP_PROP_FRAME_COUNT) == 1.0:
            cv2.imwrite('detected.png', img)

        ### Read from the video capture ###
        does_got_frame, frame = vc.read()
    vc.release()
        

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
