#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import cv2
import logging
from openvino.inference_engine import IENetwork, IECore

import numpy as np

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self,model,device,batch_size=1):
        ### TODO: Initialize any class variables desired ###
        self.model = model
        self.device = device 
        self.batch_size = batch_size

    def load_model(self):
        model_weights = self.model+".bin"
        model_structure = self.model+".xml"

        core = IECore()
        self.net = core.read_network(model=model_structure,weights=model_weights)
        self.net.batch_size = self.batch_size
        self.exec_net = core.load_network(network=self.net,device_name=self.device)
        
        # supported layers
        supported_layers = core.query_network(network=self.net,device_name=self.device)

        # check unsupported layers
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            logging.error("Unsupported layers found: {}".format(unsupported_layers))
            logging.error("Check whether extensions are available to add to IECore.")
            exit(1)
        
        self.input_blob = next(iter(self.exec_net.inputs))
        self.output_blob = next(iter(self.exec_net.outputs))
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.net.inputs[self.input_blob].shape

    def async_exec_net(self,batch,request_id=0):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        infer_request_handle = self.exec_net.start_async(request_id=request_id,inputs={self.input_blob: batch})
        return infer_request_handle

    def wait(self,infer_request_handle):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        while True:
            status = infer_request_handle.wait(-1)
            if status == 0:
                break 
            else:
                time.sleep(1)
        detection_outputs = infer_request_handle.outputs
        return detection_outputs
    
    # def sync_inference(self,batch):
    #     detection_outputs = self.exec_net.infer({self.input_blob: batch})
    #     return detection_outputs

    def get_output(self,detections):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        
        # filter based on threshold
        output = detections['DetectionOutput'][0][0]
        output = output[output[:, 2]> 0.5, :]
        num_detections = output.shape[0]
        return {'num_detections': num_detections,
                'batch': output[:, 0],
                'class': output[:, 1],
                'score': output[:, 2],
                'bbox': output[:, 3:]}


def preprocess_image(image,height=640,width=640):
    image = cv2.resize(image.copy(),(width,height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)
    return image

def draw_bboxes(image, detections):
    img = image.copy()
    for i in range(detections['batch'].shape[0]):
        classId = int(detections['class'][i])
        score = float(detections['score'][i])
        bbox = [float(v) for v in detections['bbox'][i]]
        if score > 0.3 and classId == 1:
            logging.debug(f"batch: {detections['batch'][i]} class: {classId}, score: {score}, bbox: {bbox}")
            y = bbox[1] * img.shape[0]
            x = bbox[0] * img.shape[1]
            bottom = bbox[3] * img.shape[0]
            right = bbox[2] * img.shape[1]
            cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
    return img

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Use OpenVino to detect objects in an image, uses intermediate representation with tensorflow object detection api')
    parser.add_argument('--filepath', default='resources/image_0100.jpeg', type=str, help='path to the image file')
    parser.add_argument('--model', default='frozen_inference_graph', type=str, help='path to the model excluding the extension')
    parser.add_argument('--device', default='CPU', type=str, help='the device to run inference on, one of CPU, GPU, MYRIAD, FPGA')
    parser.add_argument('--batch-size', default=1, type=int, help='size of the batch')
    parser.add_argument('--log-level',
                        default='error',
                        type=str,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the log level, one of debug, info, warning, error, critical')

    args = parser.parse_args()

    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}
    log_level = LEVELS.get(args.log_level, logging.ERROR)
    logging.basicConfig(level=log_level)

    img = cv2.imread(args.filepath)

    net = Network(args.model, args.device, args.batch_size)
    net.load_model()
    input_shape = net.get_input_shape()

    image = preprocess_image(img, input_shape[3], input_shape[2])
    # asynchronous detection
    handle = net.async_exec_net(image)
    detections = net.wait(handle)
    detections_dict = net.get_output(detections)
    img = draw_bboxes(img, detections_dict)

    cv2.imwrite('detected.png', img)


