import os
import numpy as np
import csv
import pandas as pd
from mrcnn import utils
import mrcnn.model as modellib
import train_network_xyL as networksL


ROOT_DIR = os.path.abspath("./")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = networksL.CocoConfig()


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
    image_ids = dataset.image_ids[:50]
    APs = compute_batch_ap(image_ids)

    print(APs)
    print("mAP @ IoU=50: ", np.mean(APs))
    return np.mean(APs)


def read_csv(path):
    csv_file = open(path, "r")
    reader = csv.reader(csv_file)
    result = []
    for item in reader:
        result.append(item)
    csv_file.close()
    return result


def save_data(data, path):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(path, index=False, header=False)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate x, y Label Network based on evaluate mode.')
    parser.add_argument("mode",
                        metavar="<full|last>",
                        help="Mode to evaluate")
    args = parser.parse_args()
    eval_mode = args.mode

    # dataset
    dataset_folder = 'coco_datasets'
    dataset_type = 'val'
    dataset_year = '2014'
    label = networksL.label

    if label == 'mix':
        # Accuracy on which label?
        eval_label = input('Choose which coco label to evaluate (person/bicycle): ')
        assert eval_label == 'person' or eval_label == 'bicycle'
        m_amount = 49
    else:
        eval_label = label
        m_amount = 50
    eval_labels = [eval_label]

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

    # Load validation dataset
    dataset = networksL.CocoDataset()
    dataset.load_coco(dataset_folder, dataset_type, eval_labels, year=dataset_year)
    dataset.prepare()

    output_path = '../drive/My Drive/mrcnn_{}_weights/{}_{}_acc.csv'.format(label, eval_label, eval_mode)
    dir_path = os.path.dirname(output_path)
    w_path = dir_path + '/logs/mask_rcnn_coco_'

    if eval_mode == 'full':
        # setup the path of the temp output file which will be saved every 5 times
        output_dir = os.path.dirname(output_path)
        filename = (os.path.basename(output_path)).split('.')
        temp_filename = str(filename[0]) + '_temp.' + str(filename[1])
        temp_path = os.path.join(output_dir, temp_filename)

        # output array
        output = np.zeros(m_amount)

        # ask if user want to resume computation of mAPs
        is_resumed = input('Do you want to resume computations of mAPs from before? (y/n): ')
        if is_resumed.lower() == 'y':
            start = int(input('Start from which for mAPs computation (e.g., 20 mAPs(0-19) saved, then 21)? Enter: '))
            previous_mAPs = np.asarray(read_csv(temp_path), dtype=np.float64)
            current_shape = start - 1
            previous_mAPs = previous_mAPs[:current_shape]
            previous_mAPs = previous_mAPs.reshape(current_shape)
            output[:current_shape] = previous_mAPs
        else:
            start = 1
        end = 1 + m_amount

        i = start - 1
        for index in range(start, end):
            output[i] = loop_weight(w_path, index)
            i += 1
            if (index % 5 == 0) and (index + 1 != end):
                save_data(output, temp_path)
        save_data(output, output_path)
        os.remove(temp_path)

    elif eval_mode == 'last':
        output = np.zeros(1)
        output[0] = loop_weight(w_path, m_amount)
        save_data(output, output_path)

    else:
        print('Invalid mode entered!')

    print('Finish process.')
