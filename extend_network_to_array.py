import os
import sys
import numpy as np
import mrcnn.model as modellib
import train_coco1_dog_only as train_coco1_dog
import train_coco80_all as train_coco80_all
import train_coco80_dog_only as train_coco80_dog

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

coco_config1 = train_coco1_dog.CocoConfig()
coco_config80B = train_coco80_all.CocoConfig()
coco_config80A = train_coco80_dog.CocoConfig()
MODEL_DIR1 = os.path.join(ROOT_DIR, "logs1")
MODEL_DIR80B = os.path.join(ROOT_DIR, "logs80")
MODEL_DIR80A = os.path.join(ROOT_DIR, "Alogs80")


class InferenceConfig1(coco_config1.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class InferenceConfig80B(coco_config80B.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class InferenceConfig80A(coco_config80A.__class__):
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
    networks = []
    while True:
        network = input('Enter the name of network if you want to extend it (coco1/coco80B/coco80A/n)')
        if network == 'coco1' or network == 'coco80B' or network == 'coco80A':
            networks.append(network)
            networks = list(set(networks))
            if len(networks) == 3:
                break
        else:
            break

    flatten_networks = {}
    print('Enter the weight file path of the specified coco model accordingly.')
    print('For example, ../drive/My Drive/A1vsB80/logs1/mask_rcnn_coco_0001.h5')
    for network in networks:
        if network == 'coco1':
            weight_path = input('The model with config.NUM_CLASSES=1+1 which train and infer on only dog data: ')
            config = InferenceConfig1()
            model_dir = MODEL_DIR1
        elif network == 'coco80B':
            weight_path = input('The model with config.NUM_CLASSES=1+80 which train and infer on all data: ')
            config = InferenceConfig80B()
            model_dir = MODEL_DIR80B
        else:
            weight_path = input('The model with config.NUM_CLASSES=1+80 which train and infer on only dog data: ')
            config = InferenceConfig80A()
            model_dir = MODEL_DIR80A

        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
        model.load_weights(weight_path, by_name=True)
        model_weights = model.get_weights()
        model_network_flatten_array = extend_to_array(model_weights)
        # print(len(model_network_flatten_array))
        flatten_networks[network] = len(model_network_flatten_array)

    # In summary, print each network name and its size
    for name, size in flatten_networks.items():
        print((name + ':'), size)

    print('Finish process.')
