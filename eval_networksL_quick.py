import os
import sys
import numpy as np
import pandas as pd
from mrcnn import utils
import mrcnn.model as modellib
import train_networksL as networksL


ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

config = networksL.CocoConfig()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# dataset
dataset_folder = 'coco_datasets'
dataset_type = 'val'
dataset_year = '2014'
eval_labels = ['person']

# final weights indicies
final_weights = [50, 98, 138, 176, 223, 256, 300, 325, 368, 386]

# option for detecting x images randomly
detect_num = int(input('Enter amount of images used for evaluating (0 means all): '))


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # DETECTION_MIN_CONFIDENCE = 0


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

    #  an option to randomly select 50 image ids for testing
    if detect_num != 0:
        image_ids = np.random.choice(dataset.image_ids, detect_num, replace=False, p=None)
        APs = compute_batch_ap(image_ids)
    else:
        APs = compute_batch_ap(dataset.image_ids)

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
    dataset = networksL.CocoDataset()
    dataset.load_coco(dataset_folder, dataset_type, eval_labels, year=dataset_year)
    dataset.prepare()

    dir_path_template = '../drive/My Drive/mrcnn_coco80_lb{}_weights'

    outputs = []
    for i, weight in enumerate(final_weights):
        dir_path = dir_path_template.format(i+1)
        w_path = dir_path + '/logs/mask_rcnn_coco_'
        m_num = weight

        output = loop_weight(w_path, m_num)
        outputs.append(output)

    print('mAPs of each network:')
    print(outputs)

    print('\nFinish process.')
