import os
import pandas as pd
import numpy as np
import scipy.stats as ss
import mrcnn.model as modellib
import net_xyL_acc
from net_xyL_acc import InferenceConfig

ROOT_DIR = os.path.abspath("./")

# MRCNN network head structure
mrcnn_head = {
    'train_rpn_model': ['rpn_model'],
    'train_conv_layers': ['fpn_c5p5', 'fpn_c4p4', 'fpn_c3p3', 'fpn_c2p2', 'fpn_p5', 'fpn_p2', 'fpn_p3', 'fpn_p4'],
    'train_dense_layers': ['mrcnn_mask_conv1', 'mrcnn_mask_conv2', 'mrcnn_mask_conv3', 'mrcnn_mask_conv4',
                           'mrcnn_bbox_fc', 'mrcnn_mask_deconv', 'mrcnn_class_logits', 'mrcnn_mask'],
    'train_normal_layers': ['mrcnn_mask_bn1', 'mrcnn_mask_bn2', 'mrcnn_mask_bn3', 'mrcnn_mask_bn4']
}


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


def save_data(data, path):
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(path, index=False, header=False)


if __name__ == '__main__':
    config = InferenceConfig()
    model_dir = os.path.join(ROOT_DIR, "logs")
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)

    label = net_xyL_acc.label
    w_path_template = '../drive/My Drive/mrcnn_{}_weights/logs/mask_rcnn_coco_{}.h5'

    start = "%04d" % 1
    if label == 'mix':
        end = "%04d" % 49
    else:
        end = "%04d" % 50

    weight1_path = w_path_template.format(label, start)
    weight2_path = w_path_template.format(label, end)

    kl_distance = compare_network_head_weights(model, weight1_path, weight2_path, KL_div, 'KL_div')
    kl_ds = [kl_distance]
    print(kl_distance)

    output_path = '../drive/My Drive/mrcnn_{}_weights/{}_w{}_w{}_KL.csv'.format(label, label, start, end)
    save_data((np.asarray(kl_ds, dtype=np.float64)), output_path)

    print('\nFinish Process.')
