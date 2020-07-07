import KL_div as KL
import dataReader as dr
import cnn_model as cm
import numpy as np
import random
import matplotlib.pyplot as plt


def reletive_information(dataset):
    s = KL.resize_layer(sum(dataset) / len(dataset))
    ri = 0
    for si in dataset:
        s_flat = KL.resize_layer(si)
        ri = ri + KL.KL_div(s_flat, s) + KL.KL_div(s, s_flat)
    return ri


def information_cleanliness(xtrain, ytrain, label_set):
    RI = reletive_information(xtrain)
    labelsRI = []
    for label in label_set:
        idx = ytrain == label
        x = xtrain[idx]
        labelsRI.append(reletive_information(x))
    return labelsRI, sum(labelsRI) / RI


def multilabel_RI():
    kl = []
    for i in range(10):
        idx = cm.y_index_train <= i
        test = cm.x_train[idx]
        RI = reletive_information(test)
        print(RI)
        kl.append(RI)
    return kl


def half_half_RI():
    A = cm.x_train[cm.y_index_train == 2][:2000, :, :]
    B = cm.x_train[cm.y_index_train == 0][:2000, :, :]
    A_half = A[:1000, :, :]
    B_half = B[:1000, :, :]
    C = np.array([A_half, B_half])
    C.resize([2000, 28, 28, 1])
    # C = np.array((.to).flatten()
    print(reletive_information(A), reletive_information(B), reletive_information(C))


def mix_dataset(xtrains, ytrains, dataset_size, label_size):
    if len(xtrains) == 1:
        return xtrains[0][:dataset_size, :, :], ytrains[0][:dataset_size, :]
    mix_length = int(dataset_size / label_size)
    x_train_mix = []
    y_train_mix = []

    for xtrain in xtrains:
        x_train_mix.extend(xtrain[:mix_length, :, :])  # for 2D images
    for ytrain in ytrains:
        y_train_mix.extend(ytrain[:mix_length, :])

    d_samples = dataset_size - mix_length * len(xtrains)
    if d_samples > 0:
        x_train_mix.extend(xtrains[0][-d_samples:, :, :])
        y_train_mix.extend(ytrains[0][-d_samples:, :])
    x_train_mix = np.array(x_train_mix)
    x_train_mix.resize(xtrains[0].shape)
    y_train_mix = np.array(y_train_mix)
    y_train_mix.resize(ytrains[0].shape)

    return x_train_mix, y_train_mix


def mixed_RI(label_set, test_length, output_address):
    xtrains = []
    ytrains = []
    xtrain_mix = []
    ytrain_mix = []
    for label in label_set:
        idx1 = (cm.y_index_train == label)
        x_train1 = cm.x_train[idx1][:test_length, :, :]
        y_train1 = cm.y_train[idx1][:test_length, :]
        xtrains.append(x_train1)
        ytrains.append(y_train1)
        x_mix, y_mix = mix_dataset(xtrains, ytrains, test_length, len(xtrains))
        xtrain_mix.append(x_mix)
        ytrain_mix.append(y_mix)
    RIs = []
    for dataset in xtrain_mix:
        ri = reletive_information(dataset)
        RIs.append(ri)

    dr.save_data(RIs, output_address)


def average_num(output_address):
    kl = []
    for i in range(10):
        idx = cm.y_index_train == i
        test = cm.x_train[idx]
        RI = reletive_information(test) / len(idx)
        print(RI)
        kl.append(RI)
    dr.save_data(kl, output_address)
    return kl


def quality_evaluate(error_rate):
    xtrain = cm.x_train
    ytrain = cm.y_index_train
    ICs = []
    label_ri = []
    for error in error_rate:
        x = xtrain
        y = np.array(ytrain)
        y_error = y[:int(len(y) * error)]
        y_keep = y[int(len(y) * error):]
        random.shuffle(y_error)

        yerrorlist = list(y_error)
        ykeeplist = list(y_keep)
        yerrorlist.extend(ykeeplist)

        y_new = np.array(yerrorlist)
        print(sum(y == ytrain) / len(y))
        label_ris, IC = information_cleanliness(x, y_new, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ICs.append(IC)
        label_ri.append(label_ris)
        print(label_ris)
        print(IC)
    # dr.save_data(ICs,'data_evaluation_minst/error_label_detection.csv')
    # plt.plot(error_rate,ICs)
    # plt.savefig('data_evaluation_minst/error_label_detection.png')


# kl = multilabel_RI()
# print(kl)
# dr.save_data(kl,'data_evaluation_minst/multilabel_RI.csv')

# half_half_RI()
# mixed_RI([0,1,2,3,4,5,6,7,8,9],5000,'data_evaluation_minst/mixed_label_RI.csv')
# average_num('data_evaluation_minst/averagenumber_label_RI.csv')

quality_evaluate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
