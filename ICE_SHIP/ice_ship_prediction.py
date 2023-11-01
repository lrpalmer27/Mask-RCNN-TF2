import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import random

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

TestDir="\\data\\test_imgs\\"
ROOT_DIR=os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ['BG', 'Ice','Ship']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "Ice_ship_interference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath=ROOT_DIR+"\\data\\NRC_data_multi_stage_big\\mrcnn_big.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
Test_Dir=os.listdir(ROOT_DIR+ TestDir) #lists the kangaroo test image dir
randomImg=Test_Dir[random.randint(0,len(Test_Dir)-1)]
image = cv2.imread(ROOT_DIR+TestDir+randomImg) #picks a random image in the kangaroo test image dir.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
print('Chosen random file to display with mask predictions: ',randomImg)
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])
