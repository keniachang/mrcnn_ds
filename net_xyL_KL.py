import pandas as pd
import numpy as np
import scipy.stats as ss
import mrcnn.model as modellib
from calculate import train_conv_layers, train_dense_layers, train_normal_layers, train_rpn_model, \
    train_resnet_conv_a, train_resnet_conv_b, train_resnet_conv_c, train_resnet_conv_r
from train_xyL_bus import CocoConfig
coco_config = CocoConfig()

# MRCNN network structure
mrcnn_head = {
    'train_conv_layers': train_conv_layers,
    'train_dense_layers': train_dense_layers,
    'train_normal_layers': train_normal_layers,
    'train_rpn_model': [train_rpn_model]
}
mrcnn_backbone = {
    'train_resnet_conv_a': train_resnet_conv_a,
    'train_resnet_conv_b': train_resnet_conv_b,
    'train_resnet_conv_c': train_resnet_conv_c,
    'train_resnet_conv_r': train_resnet_conv_r
}
mrcnn_structure = [mrcnn_head, mrcnn_backbone]


class InferenceConfig(coco_config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # DETECTION_MIN_CONFIDENCE = 0


def softmax(a):
    ex_a = np.exp(a)
    # print('Sum of np.exp(a) or ex_a:', sum(ex_a))
    return ex_a / sum(ex_a)


def rel_entr(a, b):
    array_a = np.array(a, dtype=np.float64)
    array_b = np.array(b, dtype=np.float64)
    # print(array_a)
    # print(array_b)
    v = array_a / array_b
    lg = np.log(v)
    return array_a * lg


def entropy(pk, qk=None, base=None, axis=0):
    pk = np.array(pk, dtype=np.float64)
    # print('Softmax of pk')
    pk = softmax(pk)
    if qk is None:
        vec = ss.entr(pk)
    else:
        qk = np.array(qk, dtype=np.float64)
        if qk.shape != pk.shape:
            raise ValueError("qk and pk must have same shape.")
        # print('Softmax of qk')
        qk = softmax(qk)
        vec = rel_entr(pk, qk)
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    return S


def KL_div(array1, array2):
    return entropy(array1, array2)


def resize_layer(layer_w):
    length = 1
    shape = np.shape(layer_w)
    for i in range(len(shape)):
        length = length * shape[i]
    # print('length = ',length)
    flat_array = np.reshape(layer_w, length)
    # print(flat_array,dtype=float)
    return flat_array


def resize_model(weights):
    output = []
    length = 0
    for w in weights:
        f_w = resize_layer(w)
        output.extend(f_w)
        length = length + len(f_w)
    # oa = np.array(output).flatten(order='C')
    oa = output
    return oa


def compare_network_head_weights(network_model, weight_path1, weight_path2, metric, metric_name):
    weight_paths = [weight_path1, weight_path2]
    vs = []
    for weight_path in weight_paths:
        network_model.load_weights(weight_path, by_name=True)
        v = []
        for name, layers in mrcnn_head.items():
            for layer_name in layers:
                layer_weights = network_model.keras_model.get_layer(layer_name).get_weights()
                flatten_layer = resize_model(layer_weights)
                v.extend(flatten_layer)
        vs.append(v)
    print('Computing', metric_name, 'on the two weights...')
    dis = metric(vs[0], vs[1])
    return dis


def compare_net_flw_lbl(network_model1, network_model2, metric):
    dis = 0
    neg = 0
    for i, mrcnn_struct in enumerate(mrcnn_structure):
        if i == 0:
            print('Processing the layers of head...')
        else:
            print('Processing the layers of backbone...')
        for var_name, layers in mrcnn_struct.items():
            print('Current: ' + var_name + '...')
            for layer_name in layers:
                layer_weights1 = network_model1.keras_model.get_layer(layer_name).get_weights()
                flatten_layer1 = resize_model(layer_weights1)

                layer_weights2 = network_model2.keras_model.get_layer(layer_name).get_weights()
                flatten_layer2 = resize_model(layer_weights2)

                layer_KL = metric(flatten_layer1, flatten_layer2)
                if layer_KL < 0:
                    neg += 1
                    print(neg, 'negative KL(s),', layer_KL, 'and it is treated as 0')
                    layer_KL = 0

                dis += layer_KL
    return dis


def compare_network_whole_weights(network_model, weight_path1, weight_path2, metric, metric_name):
    weight_paths = [weight_path1, weight_path2]
    vs = []
    for weight_path in weight_paths:
        network_model.load_weights(weight_path, by_name=True)
        v = []
        for struct in mrcnn_structure:
            for name, layers in struct.items():
                print('Processing', name)
                for layer_name in layers:
                    layer_weights = network_model.keras_model.get_layer(layer_name).get_weights()
                    flatten_layer = resize_model(layer_weights)
                    v.extend(flatten_layer)
        vs.append(v)
    print('Computing', metric_name, 'on the two weights...')
    dis = metric(vs[0], vs[1])
    return dis


def save_data(data, path):
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(path, index=False, header=False)


if __name__ == '__main__':
    from find_catID import coco_categories

    import argparse
    parser = argparse.ArgumentParser(description='Calculate KL_div on a network.')
    parser.add_argument('--label', required=True,
                        metavar="<bus|train|mix67>",
                        help="The label of the network to be computed KL_div on.")
    parser.add_argument('--mode', required=True,
                        metavar="<head|lbl|whole>",
                        help="The KL_div computation mode.")
    args = parser.parse_args()
    # assert args.label == 'bus' or args.label == 'train' or args.label == 'mix67'
    assert args.label in coco_categories[:10]
    assert args.mode == 'head' or args.mode == 'lbl' or args.mode == 'whole'

    config = InferenceConfig()
    model_dir = "./logs"
    label = args.label
    # w_path_template = '../drive/My Drive/mrcnn_{}_head_weights/logs/mask_rcnn_coco_{}.h5'
    # w_path_template = '../drive/My Drive/mrcnn_{}_weights/logs/mask_rcnn_coco_{}.h5'
    w_path_template = '../drive/My Drive/mrcnn_cl10_{}_weights/logs/mask_rcnn_coco_{}.h5'

    start = "%04d" % 1
    end = "%04d" % 50

    weight1_path = w_path_template.format(label, start)
    weight2_path = w_path_template.format(label, end)

    if args.mode == 'head':
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
        kl_distance = compare_network_head_weights(model, weight1_path, weight2_path, KL_div, 'KL_div')

    elif args.mode == 'lbl':
        model1 = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
        model2 = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
        model1.load_weights(weight1_path, by_name=True)
        model2.load_weights(weight2_path, by_name=True)
        kl_distance = compare_net_flw_lbl(model1, model2, KL_div)

    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
        kl_distance = compare_network_whole_weights(model, weight1_path, weight2_path, KL_div, 'KL_div')

    kl_ds = [kl_distance]
    print()
    print(kl_distance)

    output_path = '../drive/My Drive/mrcnn_cl10_{}_weights/{}_w{}_w{}_{}_KL.csv'.format(label,
                                                                                        label, start, end, args.mode)
    save_data((np.asarray(kl_ds, dtype=np.float64)), output_path)

    print('\nFinish Process.')
