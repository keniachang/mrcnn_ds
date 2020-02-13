"""
Mask R-CNN
Train on the workers dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import os
import sys
import json
import numpy as np
import skimage.io
import skimage.draw
import requests
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################
class CocoConfig(Config):
    """Configuration for training on the action dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + catagory

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 120

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    # VALIDATION_STEPS = 60

    # Length of square anchor side in pixels
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

    # TRAIN_ROIS_PER_IMAGE = 512


############################################################
#  Dataset
############################################################
class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset):
        """ Load a subset of the worker dataset.
            dataset_dir: Root directory of the dataset.
            subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("coco", 1, "dog")
        self.add_class("coco", 2, "cake")
        self.add_class("coco", 3, "bed")

        # Train or validation dataset?
        assert subset in ["train2017", "val2017"]
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
        if subset == "train2017":
            json_file = "./new_annotations/train/via_export_json.json"
        else:
            json_file = "./new_annotations/val/via_export_json.json"
        annotations = json.load(open(json_file))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        amount = 0
        img_dl = 0
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            polygons = [r['shape_attributes'] for r in a['regions']]
            name = [r['region_attributes']['category'] for r in a['regions']]
            name_dict = {"dog": 1, "cake": 2, "bed": 3}
            name_id = [name_dict[a] for a in name]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only manageable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])

            if os.path.exists(image_path) is False:
                coco_url = 'http://images.cocodataset.org/'
                url_link = coco_url + subset + '/' + a['filename']
                img_data = requests.get(url_link).content

                with open(image_path, 'wb') as handler:
                    handler.write(img_data)

                img_dl += 1

                print('Downloaded image ' + '(' + str(img_dl) + ')' + ': ' + (a['filename']))

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            # if a['filename'] == '000000207597.jpg':
            #     print('000000207597.jpg ID:', amount)
            amount += 1
            if subset == "train2017":
                if amount % 1000 == 0:
                    print(("Img #" + str(amount) + ":"), height, width)
            else:
                if amount % 100 == 0:
                    print(("Img #" + str(amount) + ":"), height, width)

            self.add_image(
                "coco",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                class_id=name_id,
                width=width, height=height,
                polygons=polygons)

        print("********************\n" + subset + " contains " + str(amount) + "\n********************")

    def load_mask(self, image_id):
        """ Generate instance masks for an image.
            Returns:
                masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a worker dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape [height, width, instance_count]
        name_id = image_info["class_id"]
        # print("\nname_id (class_id):", name_id, "\n")

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = np.array(name_id, dtype=np.int32)

        # print(len(info["polygons"]))
        # print(info["polygons"])

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # print("mask.shape, min(mask),max(mask): {}, {},{}".format(mask.shape, np.min(mask), np.max(mask)))
            # print("rr.shape, min(rr),max(rr): {}, {},{}".format(rr.shape, np.min(rr), np.max(rr)))
            # print("cc.shape, min(cc),max(cc): {}, {},{}".format(cc.shape, np.min(cc), np.max(cc)))

            # Note that this modifies the existing array arr, instead of creating a result array
            # Ref: https://stackoverflow.com/questions/19666626/replace-all-elements-of-python-numpy-array-that-are-greater-than-some-value
            rr[rr > mask.shape[0] - 1] = mask.shape[0] - 1
            cc[cc > mask.shape[1] - 1] = mask.shape[1] - 1
            # print("After fixing the dirt mask, new values:")
            # print("rr.shape, min(rr),max(rr): {}, {},{}".format(rr.shape, np.min(rr), np.max(rr)))
            # print("cc.shape, min(cc),max(cc): {}, {},{}".format(cc.shape, np.min(cc), np.max(cc)))

            mask[rr, cc, i] = 1

        # for i in range(len(info["polygons"])):
        #     img = mask[:, :, i]
        #     img = img * 127
        #     cv2.imshow(('mask' + str(i)), img)
        #     filename = 'mask' + str(i) + '.jpg'
        #     cv2.imwrite(filename, img)

        # Return mask, and array of class IDs of each instance.
        # Since we have one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """ Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, dataset):
    """ Train the model."""
    # Training dataset.
    dataset_train = CocoDataset()
    dataset_train.load_coco(dataset, "train2017")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CocoDataset()
    dataset_val.load_coco(dataset, "val2017")
    dataset_val.prepare()

    # print("Testing load mask for training dataset...")
    # img_id = 3321  # '000000207597.jpg'
    # dataset_train.load_mask(img_id)
    # print("Success!")

    # *** This training schedule is an example. Update to your needs ***
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
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=150,
    #             layers='heads')

    # print("Fine tune Resnet stage 4 and up")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE/10,
    #             epochs=40,
    #             layers='heads')

    print("train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10,
                epochs=150,
                layers='all')


############################################################
#  Training
############################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on custom MS COCO.')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'imagenet'")
    args = parser.parse_args()

    weight = args.model

    # Configurations
    config = CocoConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

    # Load weights
    if weight == "imagenet":
        model_path = model.get_imagenet_weights()
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc',
                                                              'mrcnn_bbox', 'mrcnn_mask'])
    else:
        print("Loading weights ", weight)
        model.load_weights(weight, by_name=True)

    # ******************************************* train coco
    dataset = './cocoDS'
    train(model, dataset)
