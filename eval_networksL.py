import os
import sys
import numpy as np
import pandas as pd
from mrcnn import utils
import mrcnn.model as modellib
import train_networksL as networksL
from calculate import read_csv


ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

config = networksL.CocoConfig()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# dataset
dataset_folder = 'coco_datasets'
dataset_type = 'val'
dataset_year = '2014'
eval_labels = ['person']

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

    print('\nThe save path for output needs to be the same level as logs folder containing weights.\n')
    # ask users for evaluation mode and the path of accuracy output file
    eval_mode = input('Enter the mode for computing AP (first2/full/last5/last): ')
    output_path = input('Enter the save path for output '
                        '(e.g., ../drive/My Drive/mrcnn_coco80_lb1_weights/dog_last_acc.csv): ')

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
        m_amount = int(input('Enter the last model number (e.g., 150 for m150): '))
        end = 1 + m_amount

        # save output every n time
        save_interval = int(input('Enter interval for saving during evaluation: '))

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

            if (index % save_interval == 0) and (index + 1 != end):
                save_data(output, temp_path)

        save_data(output, output_path)
        os.remove(temp_path)

    elif eval_mode == 'last5':
        m_num = int(input('Enter the last model number (e.g., 150 for m150): '))
        weight_amount = 1 + m_num
        amount = 5
        start_num = weight_amount - amount
        output = np.zeros(amount)
        i = 0
        for index in range(start_num, weight_amount):     # 146-151 = m146 to m150
            output[i] = loop_weight(w_path, index)
            i += 1
        save_data(output, output_path)

    elif eval_mode == 'last':
        m_num = int(input('Enter the last model number (e.g., 150 for m150): '))
        output = np.zeros(1)
        output[0] = loop_weight(w_path, m_num)
        save_data(output, output_path)

    else:
        print('Invalid mode entered!')

    print('Finish process.')
