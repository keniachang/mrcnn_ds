"""
Mask R-CNN
Train on the workers dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
"""

import os
import sys
import json
import numpy as np
import skimage.io
import skimage.draw
import argparse
import math

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
initial_weight_path = '../drive/My Drive/NetwB_InitW/mrcnn_coco_0001.h5'

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Mask R-CNN on worker data.')
parser.add_argument('--dataset', required=True,
                    metavar="<WorkerData|silhouette320_latest>",
                    help='Directory of the MS-COCO dataset')
parser.add_argument('--logs', required=True,
                    metavar="<mrcnn_worker_real|mrcnn_worker_silhouette>",
                    help='Logs and checkpoints directory (../drive/My Drive/{}/logs)')
args = parser.parse_args()
dataset = args.dataset
assert dataset == 'WorkerData' or dataset == 'silhouette320_latest'
log_folder = args.logs
print("Dataset: ", args.dataset)
print("Logs: ", args.logs)


class WorkerConfig(Config):
    # Give the configuration a recognizable name
    NAME = "worker"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + action

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 20

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

    DEFAULT_LOGS_DIR = '../drive/My Drive/{}/logs'.format(log_folder)


class WorkerDataset(utils.Dataset):
    def load_worker(self, dataset_dir, subset, return_subset_size=False):
        """Load a subset of the worker dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("action", 1, "other")
        self.add_class("action", 2, "bend")
        self.add_class("action", 3, "squat")
        self.add_class("action", 4, "stand")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        print(subset, 'size:', len(annotations))

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            polygons = [r['shape_attributes'] for r in a['regions']]
            name = [r['region_attributes']['action'] for r in a['regions']]
            name_dict = {"other": 1, "bend": 2, "squat": 3, "stand": 4}
            name_id = [name_dict[a] for a in name]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only manageable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            # print(height, width)

            self.add_image(
                "action",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                class_id=name_id,
                width=width, height=height,
                polygons=polygons)

        if return_subset_size:
            return len(annotations)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a worker dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "action":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]

        name_id = image_info["class_id"]
        print(name_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = np.array(name_id, dtype=np.int32)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # print("mask.shape, min(mask),max(mask): {}, {},{}".format(mask.shape, np.min(mask), np.max(mask)))
            # print("rr.shape, min(rr),max(rr): {}, {},{}".format(rr.shape, np.min(rr), np.max(rr)))
            # print("cc.shape, min(cc),max(cc): {}, {},{}".format(cc.shape, np.min(cc), np.max(cc)))

            """
            Note that this modifies the existing array arr, instead of creating a result array
            Ref: https://stackoverflow.com/questions/19666626/replace-all-elements-of-python-numpy-array-that-are-
            greater-than-some-value
            """
            rr[rr > mask.shape[0] - 1] = mask.shape[0] - 1
            cc[cc > mask.shape[1] - 1] = mask.shape[1] - 1

            # print("After fixing the dirt mask, new values:")
            # print("rr.shape, min(rr),max(rr): {}, {},{}".format(rr.shape, np.min(rr), np.max(rr)))
            # print("cc.shape, min(cc),max(cc): {}, {},{}".format(cc.shape, np.min(cc), np.max(cc)))

            mask[rr, cc, i] = 1

            """
            Return mask, and array of class IDs of each instance. Since we have one class ID only, we return an array 
            of 1s
            """
        # return (mask.astype(np.bool), class_ids)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "action":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, dataset):
    """Train the model."""
    # Training dataset.
    dataset_train = WorkerDataset()
    train_subset_size = dataset_train.load_worker(dataset, "train", return_subset_size=True)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = WorkerDataset()
    dataset_val.load_worker(dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    """
    layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
    """
    print("Training network heads")
    total_epochs = int(math.ceil(train_subset_size / config.STEPS_PER_EPOCH))
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=total_epochs,
                layers='heads')

    # print("Fine tune Resnet stage 4 and up")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE/10,
    #             epochs=40,
    #             layers='heads')

    # print("train all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE/10,
    #             epochs=60,
    #             layers='all')


if __name__ == '__main__':
    # Configurations
    config = WorkerConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=config.DEFAULT_LOGS_DIR)

    # Load weights
    model_path = initial_weight_path
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train
    train(model, dataset)
