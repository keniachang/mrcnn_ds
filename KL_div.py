import numpy as np
import scipy.stats as ss
import scipy.special as sp
import random


def rel_entr(a, b):
    array_a = np.array(a)
    array_b = np.array(b)
    v = array_a / array_b
    lg = np.log(v)
    return array_a * lg


def softmax(a):
    ex_a = np.exp(a)
    return ex_a / sum(ex_a)


def entropy(pk, qk=None, base=None, axis=0):
    pk = np.array(pk)
    pk = softmax(pk)
    if qk is None:
        vec = ss.entr(pk)
    else:
        qk = np.array(qk)
        if qk.shape != pk.shape:
            raise ValueError("qk and pk must have same shape.")
        qk = softmax(qk)
        vec = rel_entr(pk, qk)
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    return S


def KL_div(array1, array2):
    return entropy(array1, array2)


def JS_div(array1, array2):
    p = np.array(array1)
    q = np.array(array2)
    m = (p + q) / 2
    return KL_div(p, m) / 2 + KL_div(q, m) / 2


# a = list(range(6000000))[1:]
# b = list(range(6000000))[1:]
# random.shuffle(a)
# print(sum(rel_entr([1/2, 1/2],[9/10, 1/10])))
import cnn_model as cm


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

# cm.model.load_weights('cnn_standard/0E413b.h5')
# targetv = resize_model(cm.model.get_weights())
#
# print(KL_div(targetv, targetv))
# print(np.log(-12))
