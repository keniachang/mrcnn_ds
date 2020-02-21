import sys
import os
import cfs_coco_train as coco
import numpy as np
import pandas as pd
import csv
import shutil
import matplotlib.pyplot as plt
import scipy.stats as ss
from keras import backend as K
from mrcnn.config import Config
import mrcnn.model as mlib
import pathlib

#
# mrcnn_class_logits
# mrcnn_class
# mrcnn_bbox_fc
# mrcnn_bbox
# roi_align_classifier
# mrcnn_class_conv1
# mrcnn_class_bn1
# mrcnn_class_conv2
# mrcnn_class_bn2
# pool_squeeze
#

# head structure
train_conv_layers = ['fpn_c5p5', 'fpn_c4p4', 'fpn_c3p3', 'fpn_c2p2', 'fpn_p5', 'fpn_p2', 'fpn_p3', 'fpn_p4']
train_dense_layers = ['mrcnn_mask_conv1', 'mrcnn_mask_conv2', 'mrcnn_mask_conv3', 'mrcnn_mask_conv4',
                      'mrcnn_bbox_fc', 'mrcnn_mask_deconv', 'mrcnn_class_logits', 'mrcnn_mask']
train_normal_layers = ['mrcnn_mask_bn1', 'mrcnn_mask_bn2', 'mrcnn_mask_bn3', 'mrcnn_mask_bn4']
train_rpn_model = 'rpn_model'

# resnet structure
train_resnet_conv = ['conv1',
                     'res2a_branch2a', 'res2a_branch2b', 'res2a_branch2c',
                     'res2a_branch1',
                     'res2b_branch2a', 'res2b_branch2b', 'res2b_branch2c',
                     'res2c_branch2a', 'res2c_branch2b', 'res2c_branch2c',

                     'res3a_branch2a', 'res3a_branch2b', 'res3a_branch2c',
                     'res3a_branch1',
                     'res3b_branch2a', 'res3b_branch2b', 'res3b_branch2c',
                     'res3c_branch2a', 'res3c_branch2b', 'res3c_branch2c',

                     'res4a_branch2a', 'res4a_branch2b', 'res4a_branch2c',
                     'res4a_branch1',
                     'res4b_branch2a', 'res4b_branch2b', 'res4b_branch2c',
                     'res4c_branch2a', 'res4c_branch2b', 'res4c_branch2c',
                     'res4d_branch2a', 'res4d_branch2b', 'res4d_branch2c',
                     'res4e_branch2a', 'res4e_branch2b', 'res4e_branch2c',
                     'res4f_branch2a', 'res4f_branch2b', 'res4f_branch2c',
                     'res4g_branch2a', 'res4g_branch2b', 'res4g_branch2c',
                     'res4h_branch2a', 'res4h_branch2b', 'res4h_branch2c',
                     'res4i_branch2a', 'res4i_branch2b', 'res4i_branch2c',
                     'res4j_branch2a', 'res4j_branch2b', 'res4j_branch2c',
                     'res4k_branch2a', 'res4k_branch2b', 'res4k_branch2c',
                     'res4l_branch2a', 'res4l_branch2b', 'res4l_branch2c',
                     'res4m_branch2a', 'res4m_branch2b', 'res4m_branch2c',
                     'res4n_branch2a', 'res4n_branch2b', 'res4n_branch2c',
                     'res4o_branch2a', 'res4o_branch2b', 'res4o_branch2c',
                     'res4p_branch2a', 'res4p_branch2b', 'res4p_branch2c',
                     'res4q_branch2a', 'res4q_branch2b', 'res4q_branch2c',
                     'res4r_branch2a', 'res4r_branch2b', 'res4r_branch2c',
                     'res4s_branch2a', 'res4s_branch2b', 'res4s_branch2c',
                     'res4t_branch2a', 'res4t_branch2b', 'res4t_branch2c',
                     'res4u_branch2a', 'res4u_branch2b', 'res4u_branch2c',
                     'res4v_branch2a', 'res4v_branch2b', 'res4v_branch2c',
                     'res4w_branch2a', 'res4w_branch2b', 'res4w_branch2c',

                     'res5a_branch2a', 'res5a_branch2b', 'res5a_branch2c',
                     'res5a_branch1',
                     'res5b_branch2a', 'res5b_branch2b', 'res5b_branch2c',
                     'res5c_branch2a', 'res5c_branch2b', 'res5c_branch2c',
                     ]
train_resnet_conv_a = [
    'res2a_branch2a',
    'res2b_branch2a',
    'res2c_branch2a',
    'res3a_branch2a',
    'res3b_branch2a',
    'res3c_branch2a',
    'res4a_branch2a',

    'res4b_branch2a',
    'res4c_branch2a',
    'res4d_branch2a',
    'res4e_branch2a',
    'res4f_branch2a',
    'res4g_branch2a',
    'res4h_branch2a',
    'res4i_branch2a',
    'res4j_branch2a',
    'res4k_branch2a',
    'res4l_branch2a',
    'res4m_branch2a',
    'res4n_branch2a',
    'res4o_branch2a',
    'res4p_branch2a',
    'res4q_branch2a',
    'res4r_branch2a',
    'res4s_branch2a',
    'res4t_branch2a',
    'res4u_branch2a',
    'res4v_branch2a',
    'res4w_branch2a',

    'res5a_branch2a',
    'res5b_branch2a',
    'res5c_branch2a',
]
train_resnet_conv_b = [
    'res2a_branch2b',
    'res2b_branch2b',
    'res2c_branch2b',

    'res3a_branch2b',
    'res3b_branch2b',
    'res3c_branch2b',
    'res4a_branch2b',
    'res4b_branch2b',
    'res4c_branch2b',
    'res4d_branch2b',
    'res4e_branch2b',
    'res4f_branch2b',
    'res4g_branch2b',
    'res4h_branch2b',
    'res4i_branch2b',
    'res4j_branch2b',
    'res4k_branch2b',
    'res4l_branch2b',
    'res4m_branch2b',
    'res4n_branch2b',
    'res4o_branch2b',
    'res4p_branch2b',
    'res4q_branch2b',
    'res4r_branch2b',
    'res4s_branch2b',
    'res4t_branch2b',
    'res4u_branch2b',
    'res4v_branch2b',
    'res4w_branch2b',

    'res5a_branch2b',
    'res5b_branch2b',
    'res5c_branch2b',
]
train_resnet_conv_c = [
    'res2a_branch2c',
    'res2b_branch2c',
    'res2c_branch2c',

    'res3a_branch2c',
    'res3b_branch2c',
    'res3c_branch2c',

    'res4a_branch2c',
    'res4b_branch2c',
    'res4c_branch2c',
    'res4d_branch2c',
    'res4e_branch2c',
    'res4f_branch2c',
    'res4g_branch2c',
    'res4h_branch2c',
    'res4i_branch2c',
    'res4j_branch2c',
    'res4k_branch2c',
    'res4l_branch2c',
    'res4m_branch2c',
    'res4n_branch2c',
    'res4o_branch2c',
    'res4p_branch2c',
    'res4q_branch2c',
    'res4r_branch2c',
    'res4s_branch2c',
    'res4t_branch2c',
    'res4u_branch2c',
    'res4v_branch2c',
    'res4w_branch2c',

    'res5a_branch2c',
    'res5b_branch2c',
    'res5c_branch2c',
]
train_resnet_conv_r = ['conv1',
                       'res2a_branch1',
                       'res3a_branch1',
                       'res4a_branch1',
                       'res5a_branch1',
                       ]

# coco classes ids (use dog, cake, bed for base footprints)
coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports', 'ball', 'kite',
                'baseball', 'bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote keyboard', 'cell phone', 'microwave oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

DEFAULT_LOGS_DIR = coco.CocoConfig().DEFAULT_LOGS_DIR
SAVE_MODEL_DIR = os.path.join(DEFAULT_LOGS_DIR, 'exp_generated')


def save_data(data, path):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(path, index=False, header=False)


def read_csv(path):
    csv_file = open(path, "r")
    reader = csv.reader(csv_file)
    result = []
    for item in reader:
        result.append(item)
    csv_file.close()
    return result


def load_weight(path, pass_config):
    # path to save model trained that will to be deleted
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

    # Create model then load weight
    model1 = mlib.MaskRCNN(mode="training", config=pass_config, model_dir=SAVE_MODEL_DIR)

    # changes made adding exclude arguments for samples mode when load and train
    model1.load_weights(path, by_name=True)
    return model1


def calculate_wd_layers(m_array1, m_array2, name):
    wd = 0
    weight1 = m_array1.keras_model.get_layer(name).get_weights()
    weight2 = m_array2.keras_model.get_layer(name).get_weights()
    arw1 = np.array(weight1)
    arw2 = np.array(weight2)
    for i in range(arw1.size):
        a1 = arw1[i].reshape(arw1[i].size)
        a2 = arw2[i].reshape(arw2[i].size)
        wd = wd + ss.wasserstein_distance(a1, a2)
    return wd


def calculate_wd_bnlayers(m_array1, m_array2, name):
    wd = 0
    weight1 = m_array1.keras_model.get_layer(name).get_weights()
    weight2 = m_array2.keras_model.get_layer(name).get_weights()
    arw1 = np.array(weight1)
    arw2 = np.array(weight2)
    wd = ss.wasserstein_distance(arw1.reshape(arw1.size), arw2.reshape(arw2.size))
    return wd


def calculate_wd_models_all(model1, model2):
    wd_rpn = calculate_wd_layers(model1, model2, train_rpn_model)
    wd_conv = 0
    wd_dense = 0
    wd_normal = 0
    for name in train_conv_layers:
        wd_conv = wd_conv + calculate_wd_layers(model1, model2, name)
    for name in train_dense_layers:
        wd_dense = wd_dense + calculate_wd_layers(model1, model2, name)
    for name in train_normal_layers:
        wd_normal = wd_normal + calculate_wd_bnlayers(model1, model2, name)

    wd_resnet_conv = 0
    for name in train_resnet_conv:
        wd_resnet_conv = wd_resnet_conv + calculate_wd_layers(model1, model2, name)

    final_wd = wd_rpn + wd_conv + wd_dense + wd_normal + wd_resnet_conv

    return final_wd


# TODO: implement comparison of layers' weights (all layers) between 2 models
def compare_layers(model1, model2):
    pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Construct the dataset used for experiment.')
    parser.add_argument('--dataPath', required=True,
                        metavar="<folder_name_of_weights>",
                        help="Folder name in logs containing the weights")
    parser.add_argument('--lr', required=True,
                        metavar="<learning_rate>",
                        help="Learning rate of the models",
                        type=float)
    parser.add_argument('--modelAmount', required=False,
                        default=10,
                        metavar="<m_amount>",
                        help="The first n models to compare before and after (default is 10)",
                        type=int)
    parser.add_argument('--start', required=False,
                        default=1,
                        metavar="<starting_model_index>",
                        help="Start from which model",
                        type=int)
    args = parser.parse_args()

    lr = args.lr
    folder_name = args.dataPath
    m_amount = args.modelAmount
    start = args.start

    # variables
    decimal_place = 10
    model_name = 'mask_rcnn_coco_'
    save_csv_dir = '../drive/My Drive/cfs_' + folder_name + '/compare_bf_af/'
    pathlib.Path(save_csv_dir).mkdir(parents=True, exist_ok=True)
    save_wd_path = save_csv_dir + 'wd_first' + m_amount + '.csv'
    temp_save_wd = save_csv_dir + 'wd_first' + m_amount + '_temp.csv'
    eps = 1

    model_path = os.path.join(DEFAULT_LOGS_DIR, folder_name)
    config = coco.CocoConfig()
    config.LEARNING_RATE = lr

    wd = []
    if start != 1:
        previous = np.asarray(read_csv(temp_save_wd), dtype=np.float64)
        temp_size = previous.shape[0]
        previous = np.reshape(previous, temp_size)
        for index in range(temp_size):
            wd.append(previous[index])

    coco_path = './coco'
    # choose dataset and label
    dataset_train = coco.CocoDataset()
    dataset_train.load_coco(coco_path, "samples", shift='1')
    dataset_train.prepare()

    dataset_val = coco.CocoDataset()
    dataset_val.load_coco(coco_path, "val2017")
    dataset_val.prepare()

    # compare layers' weights and calculate wd between before and after models
    for i in range(start, (m_amount + 1)):
        before_model_name = model_name + (str(i).zfill(4)) + '.h5'
        before_model_path = os.path.join(model_path, before_model_name)
        before_model = load_weight(before_model_path, coco.CocoConfig())

        after_model_name = model_name + (str(i).zfill(4)) + '.h5'
        after_model_path = os.path.join(model_path, after_model_name)
        after_model = load_weight(after_model_path, coco.CocoConfig())

        after_model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=eps, layers='all')

        wd_between = calculate_wd_models_all(after_model, before_model)
        wd_between = round(wd_between, decimal_place)
        wd.append(wd_between)
        print('wd' + str(i) + ' is computed')
        K.clear_session()
        shutil.rmtree(SAVE_MODEL_DIR, ignore_errors=True)

        if (i % 5 == 0) and (i != m_amount):
            temp_wd = np.asarray(wd, dtype=np.float64)
            np.savetxt(temp_save_wd, temp_wd, delimiter=',', fmt='%f')

    wd = np.asarray(wd, dtype=np.float64)
    np.savetxt(save_wd_path, wd, delimiter=',', fmt='%f')
    os.remove(temp_save_wd)
    print('The wd of each before and after model is calculated and saved.\n')

    print('\nFinish comparing first', m_amount, 'models.')
