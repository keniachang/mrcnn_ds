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
from PythonAPI.pycocotools.coco import COCO
from PythonAPI.pycocotools import mask as maskUtils
from find_catID import coco_categories
from coco_label_imageIds import get_label_image_ids_file_path as get_label_path
from coco_label_imageIds import save_label_image_ids, read_label_image_ids

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# COCO Dataset
dataset = 'coco_datasets'
DEFAULT_DATASET_YEAR = "2014"

# Which network?
network_label_num = int(input('Enter number of label this network will be trained on: '))
assert 0 < network_label_num < 81

# Constants
subset_size = 500
epochs_per_label = 10
images_per_weight = subset_size / epochs_per_label
initial_weight_path = 'drive/My Drive/NetwB_InitW/mrcnn_coco_0001.h5'
must_have_label = 'dog'
network_labels = []
if network_label_num == 1:
    network_labels.append(must_have_label)
else:
    coco_labels = coco_categories[:network_label_num]
    if must_have_label in coco_labels:
        network_labels = coco_labels
    else:
        network_labels.extend(coco_labels[:(network_label_num - 1)])
        network_labels.append(must_have_label)


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

    # Adjust learning rate if needed
    LEARNING_RATE = 0.0001

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    DEFAULT_LOGS_DIR = '../drive/My Drive/mrcnn_coco80_l{}_weights/logs'.format(network_label_num)


class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, selected_cat_name, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  return_coco=False):
        """Load a subset of the COCO dataset."""

        coco = COCO("../drive/My Drive/{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        selected_class_id = coco.getCatIds(catNms=[selected_cat_name])

        if subset == 'train':
            label_img_ids_path = get_label_path(selected_cat_name, subset_size)
            if not os.path.exists(label_img_ids_path):
                image_ids = save_label_image_ids(coco, selected_class_id, subset_size, label_img_ids_path)
            else:
                image_ids = read_label_image_ids(label_img_ids_path)
        else:
            image_ids = []
            image_ids.extend(list(coco.getImgIds(catIds=selected_class_id)))
            # Remove duplicates
            image_ids = list(set(image_ids))

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                # path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                path=coco.imgs[i]['coco_url'],
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=selected_class_id, iscrowd=None)))
        if return_coco:
            return coco

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
    # Configurations
    config = CocoConfig()
    config.display()

    logs_dir = config.DEFAULT_LOGS_DIR

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=logs_dir)

    # Train epochs_per_label (10) for each label
    for i, category in enumerate(network_labels):
        if i == 0:
            model_path = initial_weight_path
        else:
            model_index = i * epochs_per_label
            model_index = str(model_index).zfill(4)
            model_path = logs_dir + '/' + 'mask_rcnn_' + config.NAME.lower() + '_{}.h5'.format(model_index)

        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

        # Training dataset
        dataset_train = CocoDataset()
        dataset_train.load_coco(dataset, "train", category)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(dataset, "val", category)
        dataset_val.prepare()

        # Training: fine tune all layers
        print('Training on', category, 'data')
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epochs_per_label,
                    layers='all')

print('Finish process.')
