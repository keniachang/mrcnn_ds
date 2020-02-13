import os
import sys
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import coco_train as coco
import random
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

DATASET_DIR = './cocoDS'


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def show_prediction(length, path):
    weights_path = path + ("%04d" % length) + '.h5'
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                                                       image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names,
                                figsize=(8, 8))
    results = model.detect([original_image], verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names,
                                r['scores'], ax=get_ax())


inference_config = InferenceConfig()
inference_config.display()
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
# Load dataset
dataset_train = coco.CocoDataset()
dataset_train.load_coco(DATASET_DIR, "train2017")
dataset_train.prepare()

dataset_val = coco.CocoDataset()
dataset_val.load_coco(DATASET_DIR, "val2017")
dataset_val.prepare()
print("Images: {}\nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))

eval_mode = input('Enter the mode for computing AP (first2/full/last5): ')
weights_folder = input('Enter the folder name containing weights: ')

w_path = './logs/' + weights_folder + '/mask_rcnn_coco_'
show_prediction(2, w_path)

print('Finish process.')
