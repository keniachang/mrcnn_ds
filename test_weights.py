import os
import sys
import numpy as np
import pandas as pd
from mrcnn import utils
import mrcnn.model as modellib
import coco_train as coco

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = coco.CocoConfig()

DATASET_DIR = './cocoDS'


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def compute_batch_ap(image_ids):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id,
                                                                                  use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r['rois'], r['class_ids'], r['scores'], r['masks'])
        print(image_id, AP)
        APs.append(AP)
    return APs


def loop_weight(length, path):
    weights_path = path + ("%04d" % length) + '.h5'
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    image_ids = np.random.choice(dataset.image_ids, 30, replace=False, p=None)
    APs = compute_batch_ap(dataset.image_ids)
    print(APs)
    print("mAP @ IoU=50: ", np.mean(APs))
    return np.mean(APs)


def save_data(data, path):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(path, index=False, header=False)


config = InferenceConfig()
config.display()
TEST_MODE = "inference"
# with tf.device("/gpu:0"):
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load validation dataset
dataset = coco.CocoDataset()
dataset.load_coco(DATASET_DIR, "val2017")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

eval_mode = input('Enter the mode for computing AP (first2/full/last5): ')
weights_folder = input('Enter the folder name containing weights: ')
output_path = input('Enter the save path for output (e.g., ./coco_eval.csv): ')

if eval_mode == 'first2':
    weight_amount = 1 + 2
    output = np.zeros(2)
    w_path = './logs/' + weights_folder + '/mask_rcnn_coco_'
    i = 0
    for index in range(1, weight_amount):
        output[i] = loop_weight(index, w_path)
        i += 1
    save_data(output, output_path)
elif eval_mode == 'full':
    weight_amount = 1 + 150
    output = np.zeros(150)
    w_path = './logs/' + weights_folder + '/mask_rcnn_coco_'
    for i in range(weight_amount):
        output[i] = loop_weight(i, w_path)
    save_data(output, output_path)
elif eval_mode == 'last5':
    weight_amount = 1 + 150
    output = np.zeros(5)
    w_path = './logs/' + weights_folder + '/mask_rcnn_coco_'
    i = 0
    for index in range(146, weight_amount):     # 146-151 = m146 to m150
        output[i] = loop_weight(index, w_path)
        i += 1
    save_data(output, output_path)
else:
    print('Invalid mode entered!')

print('Finish process.')
