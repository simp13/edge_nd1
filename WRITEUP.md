# Project Write-Up

## Requirements

The model will be deployed on the edge, such that only data on:
1) the number of people in the frame
2) time those people spent in frame
3) the total number of people counted are sent to a MQTT server

## Choosing a Model and the Model Optimizer

I used the same model that we used in this course ssd mobile net v2.I found the list of
supported tensorflow models at this link:

https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html

The tensorflow object detection model zoo is here:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

The ssd MobileNet V2 model is here:

http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

The commands I used to convert my model to an intermediate representation was:

```bash
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar xvf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer

python3 $MOD_OPT/mo.py \
  --input_model ./models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb \
  --tensorflow_object_detection_api_pipeline_config ./models/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \
  --reverse_input_channels \
  --transformations_config $MOD_OPT/extensions/front/tf/ssd_v2_support.json \
  --data_type=FP32
```

## Explaining Custom Layers

The process I used for converting custom layers was to use the Intel OpenVINO model optimizer with
it's ssd_v2_support.json file to handle some of the custom layers is
explained in the previous section.

Some of the potential reasons for handling custom layers include:
* actual implementation of a layer may depend on the framework, if Intel writes a meta-layer to
encapsulate some of these minor differences they can then specify what the layer does using a
previously defined layer
* researchers haven't settled on an ideal set of layers, many of the layers being experimented
with are easily implemented with common mathematical and image filtering operations, by supporting
custom layers users can rely on third party libraries to provide the functionality required.

## Comparing Model Performance

I compared the model size before and after conversion to the Intermediate Representation, and I
also compared the inference time.

Results:

Model Type                | Model Size | Inference Time (single) 
--------------------------|------------|-------------------------
frozen_inference_graph.pb | 69 MB      | 170 MS                     
saved_model.pb            | 70 MB      | 200 MS                      
FP32 OpenVINO IR          | 67 MB      | 159 MS                      

## Assess Model Use Cases

Some theme parks only allow a certain number of people in at a time, they count people as they go
in and out.  It would be great to automate that. I have seen them open the exit get everyone out
and then open the entrance and let new people in.  With a people counter people could flow in and
out easily.

Some elevators are only rated to hold a certain amount of weight or people, once again counting
people would be useful here to sound an alarm when the maximum number of people has been exceeded.

Many buildings have fire codes that say only a certain number of people can be in large meeting
rooms at a time, it is really difficult to keep track of that, but with an automated people counter
it could be enforced.

National Parks are interested in knowing the number of people who visit, a people counter could
help with this giving an idea of how many people visit actual landmarks each year and at a more
granular level times of day, times of the year, days of the week, etc.

Museums/Restaurants/Stores/etc. typically need to make sure everyone has left before locking up
for the night, with a people counter you could count people as they come and go and make sure
everyone has left before locking up.

## Assess Effects on End User Needs

Lighting - need to determine if the end-user will run during the day as well as at night or in the
dark.  We may need to create some artificial light, such as used by IR leds for night vision.  We
would need to assess model accuracy under the different lighting conditions and verify the accuracy
is at an acceptable level for lighting conditions.  I have seem some models make no predictions in
the dark while others make false predictions endlessly.

Model accuracy - if the accuracy is low we may have a lot of false positives, which can
artificially increase our people count, we may also have a lot of false negatives, which can
artificially decrease our people count. We would want to know in advance if we tend to have more
false positives or false negatives and whether the false positives move around.  If our cameras
are stationary and the false positives are consistent we may not identify them as new people. If
the accuracy is pretty good it can still be used for some of the use cases I outlined, but the end
user may want to include a margin of error when designing systems that use the people count.

Camera focal length - the training set should be based on the same types of cameras that the model
is deployed to if possible.  If impossible then it should be validated on a sampling of cameras.
In some cases we may not be able to validate that the model works well on all cameras, eg if we
made an app where app users have a wide variety of cell phones.  In that case we may want to plan
ahead how much of the market and which devices we will commit to supporting.  Accuracy may be
lower on devices that are not explicitly supported.  Focal length can be a cause of that. Ideally
we would fix focal length.  People can become blurry if they get too close or too far from the
camera's focal plane.

Image size - some models resize the image without preserving aspect ratio, for these models the
image resize operation can distort the image which can reduce accuracy.  The particular model I
selected preserves aspect ratio, so this is not a problem.  However, low resolution images may be
a source of poor accuracy.  The model I selected takes in 640x640 images which is large by
neural network standards.  Images that have both a width and height smaller than 640 will need to
be enlarged.  Unless the training set has a lot of examples that are also enlarged the model
will likely do worse.

Motion blur - video often suffers from artifacts that still images do not such as motion blur,
this is another possible source of blurring that could cause a drop in accuracy. We may want to
check in advance if our end-user has a nice camera with a global shutter or if they have a rolling
shutter, in which case we would want to collect some images from video with rolling shutter and
determine if it is better to ignore images with these artifacts or to include them.

If we can accept lower or comparable accuracy but need it to run faster we could try batching
or a faster model.

## Running In Project Folder Path

# terminal 1
cd ./webservice/server/node-server
node ./server.js

# terminal 2
cd ./webservice/ui
npm run dev

# terminal 3
ffserver -f ./ffmpeg/server.conf

# terminal 4
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph -d CPU | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 60 -i - http://0.0.0.0:3004/fac.ffm
```

Finally in a browser navigate to http://0.0.0.0:3000

