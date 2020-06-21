import os
import sys
import numpy as np
import pandas as pd
from PythonAPI.pycocotools.coco import COCO
from PythonAPI.pycocotools import mask as maskUtils
from mrcnn import utils
import mrcnn.model as modellib
import train_coco80_three as train_coco
from calculate import read_csv
from extract_category_imgIds import get_specified_category_image_ids


ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

config = train_coco.CocoConfig()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# variables
"""
    According to class CocoDataset() implementation in train_coco:
        Annotation dir is ../drive/My Drive/{dataset_folder}/annotations/instances_{dataset_type}{year}.json
        Images will be loaded via coco_url so no need to worry about the Dataset directory
"""
dataset_folder = 'coco_datasets'
dataset_type = 'val'
dataset_year = '2014'


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class CocoValDog(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year='2014', class_ids=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        """

        coco = COCO("../drive/My Drive/{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # Add classes
        for cat_id in class_ids:
            self.add_class("coco", cat_id, coco.loadCats(cat_id)[0]["name"])

        # only use image ids of pure single category data
        selected_cat_id0 = 61
        dataset_size = 50
        image_ids = get_specified_category_image_ids(coco, selected_cat_id0, dataset_size)

        selected_class_ids = [selected_cat_id0]

        # Add images
        for img_id in image_ids:
            self.add_image(
                "coco", image_id=img_id,
                # path=os.path.join(image_dir, coco.imgs[img_id]['file_name']),
                path=coco.imgs[img_id]['coco_url'],
                width=coco.imgs[img_id]["width"],
                height=coco.imgs[img_id]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=selected_class_ids, iscrowd=None)))
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
            return super(CocoValDog, self).load_mask(image_id)

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
            return super(CocoValDog, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoValDog, self).image_reference(image_id)

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


def loop_weight(path_ex_ep, ep):
    weights_path = path_ex_ep + ("%04d" % ep) + '.h5'
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # an option to randomly select 50 image ids for testing
    image_ids = np.random.choice(dataset.image_ids, 50, replace=False, p=None)
    APs = compute_batch_ap(image_ids)

    # APs = compute_batch_ap(dataset.image_ids)

    print(APs)
    print("mAP @ IoU=50: ", np.mean(APs))
    return np.mean(APs)


def save_data(data, path):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(path, index=False, header=False)


if __name__ == '__main__':
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

    # Load validation dataset
    dataset = CocoValDog()
    dataset.load_coco(dataset_folder, dataset_type, year=dataset_year)
    dataset.prepare()

    print('\nThe save path for output needs to be the same level as logs folder containing weights.\n')
    # ask users for evaluation mode and the path of accuracy output file
    eval_mode = input('Enter the mode for computing AP (first2/full/last5): ')
    output_path = input('Enter the save path for output '
                        '(e.g., ../drive/My Drive/mrcnn_coco_B_weights/dog150_last5_acc.csv): ')

    dir_path = os.path.dirname(output_path)
    w_path = dir_path + '/logs/mask_rcnn_coco_'

    if eval_mode == 'first2':
        amount = 2
        weight_amount = 1 + amount
        output = np.zeros(amount)
        i = 0
        for index in range(1, weight_amount):
            output[i] = loop_weight(w_path, index)
            i += 1
        save_data(output, output_path)

    elif eval_mode == 'full':
        # setup the path of the temp output file which will be saved every 5 times
        output_dir = os.path.dirname(output_path)
        filename = (os.path.basename(output_path)).split('.')
        temp_filename = str(filename[0]) + '_temp.' + str(filename[1])
        temp_path = os.path.join(output_dir, temp_filename)

        # know when to end
        m_amount = int(input('Enter the last model number (e.g., 150 for m150):'))
        end = 1 + m_amount

        # output array
        output = np.zeros(m_amount)

        # ask if user want to resume computation of mAPs
        is_resumed = input('Do you want to resume computations of mAPs from before? (y/n): ')
        if is_resumed.lower() == 'y':
            mAPs_file = input('Enter the previous output file path: ')
            start = int(input('Start from which for mAPs computation (e.g., 20 mAPs(0-19) saved, then 21)? Enter: '))
            previous_mAPs = np.asarray(read_csv(mAPs_file), dtype=np.float64)
            current_shape = start - 1
            previous_mAPs = previous_mAPs[:current_shape]
            previous_mAPs = previous_mAPs.reshape(current_shape)
            output[:current_shape] = previous_mAPs
        else:
            start = 1

        i = start - 1
        for index in range(start, end):
            output[i] = loop_weight(w_path, index)
            i += 1

            if (index % 5 == 0) and (index + 1 != end):
                save_data(output, temp_path)

        save_data(output, output_path)
        os.remove(temp_path)

    elif eval_mode == 'last5':
        m_num = int(input('Enter the last model number (e.g., 150 for m150):'))
        weight_amount = 1 + m_num
        amount = 5
        start_num = weight_amount - amount
        output = np.zeros(amount)
        i = 0
        for index in range(start_num, weight_amount):     # 146-151 = m146 to m150
            output[i] = loop_weight(w_path, index)
            i += 1
        save_data(output, output_path)

    else:
        print('Invalid mode entered!')

    print('Finish process.')
