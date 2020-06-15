import os
import sys
import numpy as np
import mrcnn.model as modellib
import train_coco1_dog_only as train_coco1_dog
import train_coco80_all_dog as train_coco80_all

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

coco_config1 = train_coco1_dog.CocoConfig()
coco_config80 = train_coco80_all.CocoConfig()
MODEL_DIR1 = os.path.join(ROOT_DIR, "logs1")
MODEL_DIR80 = os.path.join(ROOT_DIR, "logs80")

# variables
print('Enter the weight file path of the specified coco model accordingly.')
print('Example 1, ../drive/My Drive/mrcnn_coco_A_weights/logs/mask_rcnn_coco_0001.h5')
print('Example 2, ../drive/My Drive/mrcnn_coco_B_weights/logs/mask_rcnn_coco_0001.h5')
weight1_path = input('The model with config.NUM_CLASSES=1+1 which train and infer on only dog data: ')
weight80_path = input('The model with config.NUM_CLASSES=1+80 which train and infer on all data: ')


class InferenceConfig1(coco_config1.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class InferenceConfig80(coco_config80.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def resize_layer(layer_w):
    length = 1
    shape = np.shape(layer_w)
    for i in range(len(shape)):
        length = length * shape[i]
    # print('length =', length)
    flat_array = np.reshape(layer_w, length)
    # print(flat_array, dtype=float)
    return flat_array


def extend_to_array(weights):
    network_array = []
    length = 0
    for weight in weights:
        flatten_weight = resize_layer(weight)
        network_array.extend(flatten_weight)
        length += len(flatten_weight)
    # oa = np.array(network_array).flatten(order='C')
    oa = network_array
    return oa


if __name__ == '__main__':
    config1 = InferenceConfig1()
    config80 = InferenceConfig80()

    model1 = modellib.MaskRCNN(mode="inference", config=config1, model_dir=MODEL_DIR1)
    model80 = modellib.MaskRCNN(mode="inference", config=config80, model_dir=MODEL_DIR80)

    model1.load_weights(weight1_path, by_name=True)
    model80.load_weights(weight80_path, by_name=True)

    # get the weights array of each model
    model1_weights = model1.get_weights()
    model80_weights = model80.get_weights()

    model1_network_flatten_array = extend_to_array(model1_weights)
    model80_network_flatten_array = extend_to_array(model80_weights)

    # compare shape/size of the two networks?
    print(len(model1_network_flatten_array))
    print(len(model80_network_flatten_array))

    print('Finish process.')
