#!/usr/bin/env python
# coding: utf-8

# Proyecto Final
# ISWZ3401-01 Inteligencia Artificial I
# Diego Hiriart, Luis Corales

#Necessary modules import
import pathlib
import os

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io
import scipy.misc
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen
import wget
import tarfile
import zipfile

import tensorflow as tf#needs protobuf 3.20 or lower and 3.9.2 or higher for Tensorflow's tutorial https://www.tensorflow.org/hub/tutorials/tf2_object_detection

try:
    #Import object detection dependencies
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.utils import ops as utils_ops
    from object_detection.utils import dataset_util
except Exception as excpt:
    print("ERROR, you might be missing TF object detection API installation")
    print("Exception: {0}".format(excpt))
    print("To install TF object detection")
    print("Install Tensorflow's object detection API, using protocol builder to compile the models.")
    print("To do that, you need to clone or download https://github.com/tensorflow/models. Create a folder called object_detection, in it place the contents from research/object_detection.")
    print("Protocol buffers (protoc) for TF 2.10 is version <3.20,>=3.9.2, which can be found here https://github.com/protocolbuffers/protobuf/releases, ideally use 3.19.6. Install protoc by downloading the precompiled binaries and adding the executable in /bin to your Path")
    print("Then, run: 'protoc object_detection/protos/*.proto --python_out=.' in the project folder. The utils and core folders are used later.")
    print("After this, run 'copy object_detection\packages\tf2\setup.py .', 'python -m pip install .', and 'python setup.py install' in the project folder. Any version incompatibilities will be shown while running by the last command.")
    exit()

tf.get_logger().setLevel('ERROR')


# Load adn view images
def loadDataset():
    #Import training images
    #https://www.kaggle.com/datasets/sachinpatel21/pothole-image-dataset
    dataDir = '.\data\pothole_image_data'
    exampleDataset = tf.keras.utils.image_dataset_from_directory(directory=dataDir, validation_split=0.3, subset="training", seed=123)
    classNames = exampleDataset.class_names
    print("Classes in datasets: {0}".format(classNames))
    plt.figure(figsize=(10, 10))
    for images, labels in exampleDataset.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classNames[labels[i]])
        plt.axis("off")
    plt.show()

#Utilities
# - Start the object detection part, see this tutorial as reference [TensorFlow Hub Object Detection ColabTensorFlow Hub Object Detection Colab](https://www.tensorflow.org/hub/tutorials/tf2_object_detection)
# - First, some utilities must be created. These will not be used later when the model is only used to recognize potholes
def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  image = None
  if(path.startswith('http')):
    response = urlopen(path)
    image_data = response.read()
    image_data = BytesIO(image_data)
    image = Image.open(image_data)
  else:
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))

  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)

#List of all TF models
ALL_MODELS = {
'CenterNet HourGlass104 512x512' : 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1',
'CenterNet HourGlass104 Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1',
'CenterNet HourGlass104 1024x1024' : 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1',
'CenterNet HourGlass104 Keypoints 1024x1024' : 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1',
'CenterNet Resnet50 V1 FPN 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1',
'CenterNet Resnet50 V1 FPN Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1',
'CenterNet Resnet101 V1 FPN 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1',
'CenterNet Resnet50 V2 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1',
'CenterNet Resnet50 V2 Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1',
'EfficientDet D0 512x512' : 'https://tfhub.dev/tensorflow/efficientdet/d0/1',
'EfficientDet D1 640x640' : 'https://tfhub.dev/tensorflow/efficientdet/d1/1',
'EfficientDet D2 768x768' : 'https://tfhub.dev/tensorflow/efficientdet/d2/1',
'EfficientDet D3 896x896' : 'https://tfhub.dev/tensorflow/efficientdet/d3/1',
'EfficientDet D4 1024x1024' : 'https://tfhub.dev/tensorflow/efficientdet/d4/1',
'EfficientDet D5 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d5/1',
'EfficientDet D6 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d6/1',
'EfficientDet D7 1536x1536' : 'https://tfhub.dev/tensorflow/efficientdet/d7/1',
'SSD MobileNet v2 320x320' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2',
'SSD MobileNet V1 FPN 640x640' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1',
'SSD MobileNet V2 FPNLite 320x320' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1',
'SSD MobileNet V2 FPNLite 640x640' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1',
'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)' : 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1',
'SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)' : 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1',
'SSD ResNet101 V1 FPN 640x640 (RetinaNet101)' : 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1',
'SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)' : 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1',
'SSD ResNet152 V1 FPN 640x640 (RetinaNet152)' : 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1',
'SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)' : 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1',
'Faster R-CNN ResNet50 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1',
'Faster R-CNN ResNet50 V1 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1',
'Faster R-CNN ResNet50 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1',
'Faster R-CNN ResNet101 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1',
'Faster R-CNN ResNet101 V1 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1',
'Faster R-CNN ResNet101 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1',
'Faster R-CNN ResNet152 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1',
'Faster R-CNN ResNet152 V1 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1',
'Faster R-CNN ResNet152 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1',
'Faster R-CNN Inception ResNet V2 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1',
'Faster R-CNN Inception ResNet V2 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1',
'Mask R-CNN Inception ResNet V2 1024x1024' : 'https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1'
}

#List of test images for default model
IMAGES_FOR_TEST = {
  'Beach' : 'object_detection/test_images/image2.jpg',
  'Dogs' : 'object_detection/test_images/image1.jpg',
  # By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
  'Naxos Taverna' : 'https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg',
  # Source: https://commons.wikimedia.org/wiki/File:The_Coleoptera_of_the_British_islands_(Plate_125)_(8592917784).jpg
  'Beatles' : 'https://upload.wikimedia.org/wikipedia/commons/1/1b/The_Coleoptera_of_the_British_islands_%28Plate_125%29_%288592917784%29.jpg',
  # By Américo Toledano, Source: https://commons.wikimedia.org/wiki/File:Biblioteca_Maim%C3%B3nides,_Campus_Universitario_de_Rabanales_007.jpg
  'Phones' : 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg/1024px-Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg',
  # Source: https://commons.wikimedia.org/wiki/File:The_smaller_British_birds_(8053836633).jpg
  'Birds' : 'https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg',
  'Pothole': 'data/pothole_image_data/pothole/481.jpg',
}    

# Load object labels to be used
def loadObjectLabels(labelsPath='.\data\pothole_label_map.pbtxt'):
    category_index = label_map_util.create_category_index_from_labelmap(labelsPath, use_display_name=True)
    return category_index

#Load model weights
def loadModel(path):
    model = tf.saved_model.load(path)
    return model

# # Select and load a pre-existing model from Tensorflow's repo 
# It is normal to get a bunch of function importing warnings when loading the model, as long as it loads

def downloadTFModel():
    model_display_name = 'SSD MobileNet V2 FPNLite 640x640'
    modelDownloadURL = ALL_MODELS[model_display_name]+"?tf-hub-format=compressed"
    
    print('Selected model:'+ model_display_name)
    print('Model Handle at TensorFlow Hub: {}'.format(modelDownloadURL))
    #Download model and save it, only needed once since it can be reloaded from local files later
    print("Downloading model file...")
    modelFileName = model_display_name.replace(' ', '_')+".tar.gz"
    if os.path.exists(modelFileName):
        os.remove(modelFileName)
    wget.download(modelDownloadURL, modelFileName)
    print("\nModel file downloaded!")
    print("Extracting model...")
    tarFile = tarfile.open(modelFileName)
    extractionPaths = ['.\hub-model', '.\detection-model']
    for extractPath in extractionPaths:
        tarFile.extractall(extractPath) # specify which folder to extract to
    tarFile.close()
    print("Model extracted!")


#Train the model and save it
# Re-train the model so it only recognizes potholes, then save it. Warnings may be displayed
def trainModel():
    print("Make sure you have already:")
    print("\t-Ran XMLToCSVsScript.py")
    print("\t-Ran CreateTFRecordsScript.py with '--csv_input=data/training_potholes_images_labels.csv --output_path=data/train.record --image_dir=data/pothole_image_data/pothole/'")
    print("\t-Ran CreateTFRecordsScript.py with '--csv_input=data/testing_potholes_images_labels.csv --output_path=data/test.record --image_dir=data/pothole_image_data/pothole/'")
    cont = input("Continue (yes/no)")
    if cont == "no":
        return
    else:
        print("Make sure the config file for the model is downloaded and adjusted (https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2), as well as the checkoint file for the model (http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/mobilenet_v2.tar.gz) into the path you specify for config")
        print("To train the model, run: python model_main_tf2.py --pipeline_config_path=ssd_mobilenet_v2_FPNLite_640x640/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.config"+
              "--model_dir=detection-model --alsologtostderr --num_train_steps={num_steps}"+
              "--sample_1_of_n_eval_examples={n_eval_examples}")
        print("This script from TF will train and save the model")

# Create functions to do the inference (recognize objects) and view results
# - To recognize objects and view results, code form the Tensorflow tutorial was followed.
# - In order to use live video input and view results, this tutorial was followed [Adapting to video feed - TensorFlow Object Detection API Tutorial p.2](https://www.youtube.com/watch?v=MyAOtvwTkT0&list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku&index=2) or [Streaming Object Detection Video - Tensorflow Object Detection API Tutorial](https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/)

def recognizeObjectsImage(model, image_np):
    # running inference
    results = model(image_np)

    # different object detection models have additional results
    # all of them are explained in the documentation
    result = {key:value.numpy() for key,value in results.items()}
    print(result.keys())
    return result

def viewResultsImage(result, image_np, category_index):
    label_id_offset = 0
    image_np_with_detections = image_np.copy()

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
      keypoints = result['detection_keypoints'][0]
      keypoint_scores = result['detection_keypoint_scores'][0]

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections[0],
          result['detection_boxes'][0],
          (result['detection_classes'][0] + label_id_offset).astype(int),
          result['detection_scores'][0],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False,
          keypoints=keypoints,
          keypoint_scores=keypoint_scores)

    plt.figure(figsize=(24,32))
    plt.imshow(image_np_with_detections[0])
    plt.show()
    return

def recognizeAndViewObjectsVideo(model, category_index):
    #Start video capture
    liveCapture = cv2.VideoCapture(0)
    # running inference
    while True:
        ret, image_np = liveCapture.read()#Read images from cv2's live video catpure
        
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        # running inference
        results = model(image_np_expanded)
        # different object detection models have additional results
        # all of them are explained in the documentation
        result = {key:value.numpy() for key,value in results.items()}
        
        #Visualization
        label_id_offset = 0
        # Use keypoints if available in detections
        keypoints, keypoint_scores = None, None
        viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_expanded[0],
          result['detection_boxes'][0],
          (result['detection_classes'][0] + label_id_offset).astype(int),
          result['detection_scores'][0],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False,
          keypoints=keypoints,
          keypoint_scores=keypoint_scores)
        
        cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


# # Testing with a sample image
# Test the model with an existing image from the repo, needs the test_images folder from object_detection

def testWithImage(selected_image='Pothole', modelPath='.\detection-model', labelsPath='.\data\pothole_label_map.pbtxt'):
    #selected_image @param ['Beach', 'Dogs', 'Naxos Taverna', 'Beatles', 'Phones', 'Birds', 'Pothole']
    flip_image_horizontally = False
    convert_image_to_grayscale = False
    
    image_path = IMAGES_FOR_TEST[selected_image]
    image_np = load_image_into_numpy_array(image_path)
    
    # Flip horizontally
    if(flip_image_horizontally):
      image_np[0] = np.fliplr(image_np[0]).copy()
    
    # Convert image to grayscale
    if(convert_image_to_grayscale):
      image_np[0] = np.tile(
        np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)).astype(np.uint8)
    
    plt.figure(figsize=(24,32))
    plt.imshow(image_np[0])
    plt.show()
    
    objectRecogModel = loadModel(modelPath)
    results = recognizeObjectsImage(objectRecogModel, image_np)
    category_index =  loadObjectLabels(labelsPath)
    viewResultsImage(results, image_np, category_index)


#Testing with live video feed
def liveVideoRecognition(modelPath='.\detection-model', labelsPath='.\data\pothole_label_map.pbtxt'):
    objectRecogModel = loadModel(modelPath)
    category_index = loadObjectLabels(labelsPath)
    recognizeAndViewObjectsVideo(objectRecogModel, category_index)

#Object detection from video file
def videoFileRecognition(modelPath='.\detection-model'):
    pass

class MainMenuSwitch():
    def option(self, opc):
        default = "Invalid option"
        return getattr(self, "case{0}".format(opc), lambda: print(default))()
    
    def case1(self):
        return loadDataset()
    
    def case2(self):
        return downloadTFModel()

    def case3(self):
        return trainModel(modelPath = '.\hub-model')
    
    def case4(self):
        return trainModel()
    
    def case5(self):
        return testWithImage(selected_image='Naxos Taverna', modelPath='.\hub-model', labelsPath='.\data\mscoco_label_map.pbtxt')
    
    def case6(self):
        return testWithImage()
    
    def case7(self):
        return liveVideoRecognition(modelPath='.\hub-model', labelsPath='.\data\mscoco_label_map.pbtxt')
    
    def case8(self):
        return liveVideoRecognition()
    
    def case9(self):
        return videoFileRecognition(modelPath='.\hub-model', labelsPath='.\data\mscoco_label_map.pbtxt')
    
    def case10(self):
        return videoFileRecognition()    

def main():
    matplotlib.use('TkAgg')#So that matplotlib can render graphics
    print("\nDiego Hiriart, Luis Corales\nISWZ3401-01 Inteligencia Artificial I\nProyecto Final")
    menuSwitch = MainMenuSwitch()
    while True:
        print("\nMenu\n1.Load potholes dataset and show images\t2.Download default model from TensorFlow\n3.Train TF model"+
              "\t4.Train local (potholes) model\n5.Test TF model with image\t6.Test potholes model with image"+
              "\n7.Test with live video using default TF\t8.Live video with poholes model"+
              "\n9.Test TF model with video file\t10.Test potholes model with video file\n11.Exit")
        opc = int(input("Select an option: "))
        if opc == 11:
            break
        menuSwitch.option(opc)
    print("Program ended")
    input("Press any key to exit")

if __name__ == '__main__':
    main()