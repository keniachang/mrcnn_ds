"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import numpy as np
import math
from PythonAPI.pycocotools.coco import COCO
from PythonAPI.pycocotools import mask as maskUtils
from find_catID import coco_categories

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# COCO Dataset
dataset = 'coco_datasets'
DEFAULT_DATASET_YEAR = "2014"

network_label_num = int(input('Enter number of label this network will be trained on: '))

# Constants
label_size = 500
# epochs_per_label = 10
# images_per_weight = label_size / epochs_per_label
images_per_weight = 10
initial_weight_path = '../drive/My Drive/NetwB_InitW/mrcnn_coco_0001.h5'


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # Background + catagory

    # Number of training steps per epoch
    STEPS_PER_EPOCH = images_per_weight

    VALIDATION_STEPS = 20

    # Adjust learning rate if needed
    LEARNING_RATE = 0.0001

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    DEFAULT_LOGS_DIR = '../drive/My Drive/mrcnn_coco80_lb{}_weights/logs'.format(network_label_num)


class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, network_cats, year=DEFAULT_DATASET_YEAR,
                  class_ids=None, return_subset_size=False):
        """Load a subset of the COCO dataset."""

        coco = COCO("../drive/My Drive/{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        selected_class_ids = coco.getCatIds(catNms=network_cats)
        image_ids = []
        if subset == 'train':
            for cat_id in selected_class_ids:
                cat_img_ids = list(coco.getImgIds(catIds=[cat_id]))
                cat_img_ids = list(set(cat_img_ids))
                if len(cat_img_ids) < label_size:
                    image_ids.extend(cat_img_ids)
                else:
                    cat_img_ids = cat_img_ids[:label_size]
                    image_ids.extend(cat_img_ids)
        else:
            for cat_id in selected_class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[cat_id])))
        # Remove duplicates
        image_ids = list(set(image_ids))

        print(subset, 'size:', len(image_ids))

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                # path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                path=coco.imgs[i]['coco_url'],
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=selected_class_ids, iscrowd=None)))
        if return_subset_size:
            return len(image_ids)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("model",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'initial'")
    args = parser.parse_args()

    model_path = args.model
    if model_path.lower() == "initial":
        model_path = initial_weight_path

    # Which network?
    must_have_label = input('A must have label? (<name>/none): ')
    assert 0 < network_label_num < 81
    if must_have_label.lower() == 'none':
        must_have_label = None
    else:
        must_have_label = must_have_label.lower()
        assert must_have_label in coco_categories

    network_labels = []
    if must_have_label:
        if network_label_num == 1:
            network_labels.append(must_have_label)
        else:
            coco_labels = coco_categories[:network_label_num]
            if must_have_label in coco_labels:
                network_labels = coco_labels
            else:
                network_labels.extend(coco_labels[:(network_label_num - 1)])
                network_labels.append(must_have_label)
    else:
        coco_labels = coco_categories[:network_label_num]
        network_labels.extend(coco_labels)

    # Configurations
    config = CocoConfig()
    config.display()

    logs_dir = config.DEFAULT_LOGS_DIR

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=logs_dir)

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Training dataset
    dataset_train = CocoDataset()
    train_subset_size = dataset_train.load_coco(dataset, "train", network_labels, return_subset_size=True)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CocoDataset()
    dataset_val.load_coco(dataset, "val", network_labels)
    dataset_val.prepare()

    # Training: fine tune all layers
    total_epochs = int(math.ceil(train_subset_size / images_per_weight))
    print('Training on', network_label_num, 'coco label data')
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=total_epochs,
                layers='all')

    print('Finish process.')
