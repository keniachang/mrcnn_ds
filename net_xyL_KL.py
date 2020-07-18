import os
import pandas as pd
import numpy as np
import scipy.stats as ss
import mrcnn.model as modellib
from calculate import train_conv_layers, train_dense_layers, train_normal_layers, train_rpn_model, \
    train_resnet_conv_a, train_resnet_conv_b, train_resnet_conv_c, train_resnet_conv_r
import train_network_xyL
from net_xyL_acc import InferenceConfig

ROOT_DIR = os.path.abspath("./")

# MRCNN network structure
mrcnn_head = {
    'train_conv_layers': train_conv_layers,
    'train_dense_layers': train_dense_layers,
    'train_normal_layers': train_normal_layers,
    'train_rpn_model': train_rpn_model
}
mrcnn_backbone = {
    'train_resnet_conv_a': train_resnet_conv_a,
    'train_resnet_conv_b': train_resnet_conv_b,
    'train_resnet_conv_c': train_resnet_conv_c,
    'train_resnet_conv_r': train_resnet_conv_r
}
mrcnn_structure = [mrcnn_head, mrcnn_backbone]


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


def calc_network_weight(network_model1, network_model2, weight_path1, weight_path2, metric):
    network_model1.load_weights(weight_path1, by_name=True)
    network_model2.load_weights(weight_path2, by_name=True)
    dis = 0
    for var_name, layers in mrcnn_head.items():
        for layer_name in layers:
            layer_weights1 = network_model1.keras_model.get_layer(layer_name).get_weights()
            flatten_layer1 = resize_model(layer_weights1)

            layer_weights2 = network_model2.keras_model.get_layer(layer_name).get_weights()
            flatten_layer2 = resize_model(layer_weights2)

            dis += metric(flatten_layer1, flatten_layer2)
    return dis


def save_data(data, path):
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(path, index=False, header=False)


if __name__ == '__main__':
    config = InferenceConfig()
    model_dir = os.path.join(ROOT_DIR, "logs")
    model1 = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
    model2 = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)

    label = train_network_xyL.label
    w_path_template = '../drive/My Drive/mrcnn_{}_head_weights/logs/mask_rcnn_coco_{}.h5'

    start = "%04d" % 1
    end = "%04d" % 50

    weight1_path = w_path_template.format(label, start)
    weight2_path = w_path_template.format(label, end)

    # kl_distance = compare_network_head_weights(model, weight1_path, weight2_path, KL_div, 'KL_div')
    # kl_ds = [kl_distance]
    kl_distance = calc_network_weight(model1, model2, weight1_path, weight2_path, KL_div)

    print(kl_distance)

    # output_path = '../drive/My Drive/mrcnn_{}_head_weights/{}_w{}_w{}_KL.csv'.format(label, label, start, end)
    # save_data((np.asarray(kl_ds, dtype=np.float64)), output_path)

    print('\nFinish Process.')
