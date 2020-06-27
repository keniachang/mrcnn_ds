import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as ss
import csv
import mrcnn.model as modellib
import extend_network_to_array as ext_net_arr
from eval_networksL import InferenceConfig

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

# MRCNN network head structure
mrcnn_head = {
    'train_rpn_model': ['rpn_model'],
    'train_conv_layers': ['fpn_c5p5', 'fpn_c4p4', 'fpn_c3p3', 'fpn_c2p2', 'fpn_p5', 'fpn_p2', 'fpn_p3', 'fpn_p4'],
    'train_dense_layers': ['mrcnn_mask_conv1', 'mrcnn_mask_conv2', 'mrcnn_mask_conv3', 'mrcnn_mask_conv4',
                           'mrcnn_bbox_fc', 'mrcnn_mask_deconv', 'mrcnn_class_logits', 'mrcnn_mask'],
    'train_normal_layers': ['mrcnn_mask_bn1', 'mrcnn_mask_bn2', 'mrcnn_mask_bn3', 'mrcnn_mask_bn4']
}

# train_rpn_model = 'rpn_model'
# train_conv_layers = ['fpn_c5p5', 'fpn_c4p4', 'fpn_c3p3', 'fpn_c2p2', 'fpn_p5', 'fpn_p2', 'fpn_p3', 'fpn_p4']
# train_dense_layers = ['mrcnn_mask_conv1', 'mrcnn_mask_conv2', 'mrcnn_mask_conv3', 'mrcnn_mask_conv4',
#                       'mrcnn_bbox_fc', 'mrcnn_mask_deconv', 'mrcnn_class_logits', 'mrcnn_mask']
# train_normal_layers = ['mrcnn_mask_bn1', 'mrcnn_mask_bn2', 'mrcnn_mask_bn3', 'mrcnn_mask_bn4']


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


def compare_two_weights(network_model, weight_path1, weight_path2, metric):
    network_model.load_weights(weight_path1, by_name=True)
    v1 = resize_model(network_model.get_weights())

    network_model.load_weights(weight_path2, by_name=True)
    v2 = resize_model(network_model.get_weights())

    dis = metric(v1, v2)
    return dis


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

    for v in vs:
        print('Flatten network head:', len(v))
        print('Min:', np.min(v))
        print('Max:', np.max(v))
        print()

    print('Computing', metric_name, 'on the two weights...')
    dis = metric(vs[0], vs[1])
    return dis


def save_data(data, path):
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(path, index=False, header=False)


def read_previous_output_csv_file(path):
    csv_file = open(path, "r")
    reader = csv.reader(csv_file)
    result = []
    for item in reader:
        result.append(float(item))
    csv_file.close()
    return result


"""
Current Modes Supported:
    1. 1m2w: calculate KL between two weights of the same model/network.
    2. 1mfsw: calculate KL between a model's final weight and its other weights of the same training sequence.
"""
if __name__ == '__main__':
    # # calculate KL between two models/networks
    # a_weight_path = input('The weight file path of network A80: ')
    # a_config = ext_net_arr.InferenceConfig80A()
    # a_model_dir = ext_net_arr.MODEL_DIR80A
    # a_model = modellib.MaskRCNN(mode="inference", config=a_config, model_dir=a_model_dir)
    # a_model.load_weights(a_weight_path)
    #
    # b_weight_path = input('The weight file path of network B80: ')
    # b_config = ext_net_arr.InferenceConfig80B()
    # b_model_dir = ext_net_arr.MODEL_DIR80B
    # b_model = modellib.MaskRCNN(mode="inference", config=b_config, model_dir=b_model_dir)
    # b_model.load_weights(b_weight_path)
    #
    # kl_distance = compare_two_model(a_model, b_model, KL_div)
    # print(kl_distance)

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate the KL between two weights.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'1m2w' or '1mfsw'")
    parser.add_argument('--network', required=True,
                        metavar="<A80|B80|C80>",
                        help="Network type")
    parser.add_argument('--path1', required=True,
                        metavar="/path/to/first/weights",
                        help='File path to the first weight file')
    parser.add_argument('--path2', required=True,
                        metavar="/path/to/second/weights",
                        help='File path to the second or final weight file')
    parser.add_argument('--save_output', required=False,
                        default=True,
                        metavar="<True|False>",
                        help='Save the result of the calculation for 1m2w mode (default=True)',
                        type=bool)
    parser.add_argument('--prev_output', required=False,
                        default='None',
                        metavar="<Previous output file path|None>",
                        help='To resume the calculation process for 1mfsw mode (default=None)',
                        type=str)
    args = parser.parse_args()
    print("Mode: ", args.command)
    print("Netowrk: ", args.network)
    print("First weight file path: ", args.path1)
    print("Second or final weight file path: ", args.path2)
    print("Save output: ", args.save_output)
    print("Previous output path: ", args.prev_output)

    mode = args.command
    network = args.network
    weight1_path = args.path1
    weight2_path = args.path2
    save_output = args.save_output
    prev_output = args.prev_output

    if network == 'A80':
        config = ext_net_arr.InferenceConfig80A()
        model_dir = ext_net_arr.MODEL_DIR80A
    elif network == 'B80':
        config = ext_net_arr.InferenceConfig80B()
        model_dir = ext_net_arr.MODEL_DIR80B
    elif network == 'C80':
        config = ext_net_arr.InferenceConfig80C()
        model_dir = ext_net_arr.MODEL_DIR80C
    elif network == 'L':
        config = InferenceConfig()
        model_dir = os.path.join(ROOT_DIR, "Network_L")
    else:
        sys.exit('Unsupported network!')

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)

    if mode == '1m2w':
        if str(os.path.dirname(weight1_path)) != str(os.path.dirname(weight2_path)):
            sys.exit('The two weights provided to calculate KL is not of the same model!')

        # kl_distance = compare_two_weights(model, weight1_path, weight2_path, KL_div)
        kl_distance = compare_network_head_weights(model, weight1_path, weight2_path, KL_div, 'KL_div')
        kl_ds = [kl_distance]

        if save_output:
            weights_dir = os.path.dirname(weight1_path)

            weight1_filename = str(((os.path.basename(weight1_path)).split('.'))[0])
            weight1_index = (weight1_filename.split('_'))[-1]
            weight2_filename = str(((os.path.basename(weight2_path)).split('.'))[0])
            weight2_index = (weight2_filename.split('_'))[-1]

            output_path = os.path.join(os.path.dirname(weights_dir),
                                       "{}_w{}_w{}_KL.csv".format(network, weight1_index, weight2_index))
            save_data((np.asarray(kl_ds, dtype=np.float64)), output_path)
            print(str(weight1_index) + '-' + str(weight2_index) + ':', kl_distance)
        else:
            print(kl_distance)

    elif mode == '1mfsw':
        if str(os.path.dirname(weight1_path)) != str(os.path.dirname(weight2_path)):
            sys.exit('The two weights provided to calculate KL is not of the same model!')

        weight1_filename = str(((os.path.basename(weight1_path)).split('.'))[0])
        weight1_index = (weight1_filename.split('_'))[-1]

        weight2_filename = str(((os.path.basename(weight2_path)).split('.'))[0])
        weight2_index = (weight2_filename.split('_'))[-1]

        weights_dir = os.path.dirname(weight1_path)
        weights_prefix = '_'.join((weight1_filename.split('_'))[:-1])
        weights_path_prefix = str(weights_dir) + '/' + weights_prefix

        while True:
            if weight1_index[0] == '0':
                weight1_index = weight1_index[1:]
            else:
                break

        while True:
            if weight2_index[0] == '0':
                weight2_index = weight2_index[1:]
            else:
                break

        start = int(weight1_index)
        end = int(weight2_index) + 1

        if prev_output != 'None':
            prev_start = int((((str(((os.path.basename(prev_output)).split('.'))[0])).split('_'))[1])[1:])
            prev_end = int((((str(((os.path.basename(prev_output)).split('.'))[0])).split('_'))[2])[1:])

            if (prev_end + 1) != start:
                sys.exit('Resuming is not properly done!')

            kl_ds = read_previous_output_csv_file(prev_output)
            out_start = prev_start
        else:
            kl_ds = []
            out_start = start

        temp_path = ''
        for index in range(start, end):
            epoch = str(index).zfill(4)
            current_weight_path = weights_path_prefix + '_{}.h5'.format(epoch)

            # kl_distance = compare_two_weights(model, current_weight_path, weight2_path, KL_div)
            kl_distance = compare_network_head_weights(model, current_weight_path, weight2_path, KL_div, 'KL_div')
            kl_ds.append(kl_distance)
            print(str(index) + '-' + str(weight2_index) + ':', kl_distance)
            print('\n')

            if index % 5 == 0 and index + 1 != end:
                temp_path = os.path.join(os.path.dirname(weights_dir),
                                         "{}_w{}_w{}_KL.csv".format(network, out_start, index))
                save_data((np.asarray(kl_ds, dtype=np.float64)), temp_path)

                if index > 5:
                    prev_temp = os.path.join(os.path.dirname(weights_dir),
                                             "{}_w{}_w{}_KL.csv".format(network, out_start, (index - 5)))
                    os.remove(prev_temp)

        output_path = os.path.join(os.path.dirname(weights_dir),
                                   "{}_w{}_w{}_KL.csv".format(network, out_start, weight2_index))
        save_data((np.asarray(kl_ds, dtype=np.float64)), output_path)
        os.remove(temp_path)

    else:
        print('The mode passed is not implemented.')

    print('\nFinish Process.')
