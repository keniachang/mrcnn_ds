import os
import cnn_model as cm
import scipy.stats as ss
import scipy
import numpy as np
import matplotlib.pyplot as plt
import dataReader as dr
import random
import KL_div as kl


def sort_key(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0]) * 468 + int(batch_str[0])


def sort_key_zero(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0]) * 46 + int(batch_str[0])


def sort_key_one(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0]) * 52 + int(batch_str[0])


def sort_key_two(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0]) * 46 + int(batch_str[0])


def sort_key_three(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0]) * 47 + int(batch_str[0])


def sort_key_four(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0]) * 45 + int(batch_str[0])


def sort_key_five(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0]) * 42 + int(batch_str[0])


def sort_key_six(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0]) * 46 + int(batch_str[0])


def sort_key_seven(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0]) * 48 + int(batch_str[0])


def sort_key_eight(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0]) * 45 + int(batch_str[0])


def sort_key_nine(e):
    epoch_str = e.split('E')
    batch_str = epoch_str[1].split('b')
    return int(epoch_str[0]) * 46 + int(batch_str[0])


def sum_layer(array):
    length = 1
    shape = np.shape(array)
    for i in range(len(shape)):
        length = length * shape[i]
    # print('length = ',length)
    flat_array = np.reshape(array, [length, 1])
    return sum(flat_array)[0], length


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


def draw_plot(list, name_list, xlabel='', ylabel='', title='', fz=(8, 4)):
    plt.figure(figsize=fz)
    for i in range(len(list)):
        plt.plot(range(len(list[i])), list[i], label=name_list[i])
    plt.title(title)  # 标题
    plt.xlabel(xlabel)  # x轴的标签
    plt.ylabel(ylabel)  # y轴的标签
    plt.legend()
    plt.show()


def normalize(array, ratio):
    sumv = sum(array)
    ntarget = [a * ratio for a in array]
    ntarget = [a / sumv for a in ntarget]
    return ntarget


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"

    if sum(x) == 0 or sum(y) == 0:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def accuracy(model):
    score = model.evaluate(cm.x_test, cm.y_test, verbose=0)
    return score[0], score[1]


def single_accuracy_test():
    file_address = input('enter output path:')
    file_ls = os.listdir(file_address)
    file_ls.sort(key=sort_key)
    # file_ls.sort(key=sort_key)
    losses = []
    accuracies = []
    # vm.vae.load_weights(output_path+'/0E0b.h5')
    # cm.model.load_weights(output_path+'/4E467b.h5') #end of normal
    # cm.model.load_weights(file_address + '/4E45b.h5')  # end of sample_selection

    for file in file_ls:
        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue
        cm.model.load_weights(file_address + '/' + file)
        print(file)
        loss, ac = accuracy(cm.model)
        print(file)
        losses.append(loss)
        accuracies.append(ac)
        # if i>10:
        #     break
    # list = []
    # list.append(eds)
    # list.append(wds)
    # name_list = ['ed','wd']
    # draw_plot(list,name_list)

    plt.plot(range(len(losses)), losses)
    dr.save_data(losses, 'record/' + file_address + '/losses.csv')
    plt.savefig('record/' + file_address + '/losses.png')
    plt.close()

    plt.plot(range(len(accuracies)), accuracies)
    dr.save_data(accuracies, 'record/' + file_address + '/accuracy.csv')
    plt.savefig('record/' + file_address + '/accuracy.png')
    plt.close()


def group_test(length):
    for i in range(length):
        if i == 0:
            file_address = 'model_cnn_zero'
            selection = sort_key_zero
            cm.model.load_weights(file_address + '/4E45b.h5')
        elif i == 1:
            file_address = 'model_cnn_one'
            selection = sort_key_one
            cm.model.load_weights(file_address + '/4E51b.h5')
        elif i == 2:
            file_address = 'model_cnn_two'
            selection = sort_key_two
            cm.model.load_weights(file_address + '/4E45b.h5')
        elif i == 3:
            file_address = 'model_cnn_three'
            selection = sort_key_three
            cm.model.load_weights(file_address + '/4E46b.h5')

        elif i == 4:
            file_address = 'model_cnn_four'
            selection = sort_key_four
            cm.model.load_weights(file_address + '/4E44b.h5')

        elif i == 5:
            file_address = 'model_cnn_five'
            selection = sort_key_five
            cm.model.load_weights(file_address + '/4E41b.h5')

        elif i == 6:
            file_address = 'model_cnn_six'
            selection = sort_key_six
            cm.model.load_weights(file_address + '/4E45b.h5')

        elif i == 7:
            file_address = 'model_cnn_seven'
            selection = sort_key_seven
            cm.model.load_weights(file_address + '/4E47b.h5')

        elif i == 8:
            file_address = 'model_cnn_eight'
            selection = sort_key_eight
            cm.model.load_weights(file_address + '/4E44b.h5')

        elif i == 9:
            file_address = 'model_cnn_nine'
            selection = sort_key_nine
            cm.model.load_weights(file_address + '/4E45b.h5')
        else:
            file_address = 'model_cnn_default'
            selection = sort_key
            cm.model.load_weights(file_address + '/4E45b.h5')
        print('start: ', i)
        file_ls = os.listdir(file_address)
        file_ls.sort(key=selection)
        # file_ls.sort(key=sort_key)

        # vm.vae.load_weights(output_path+'/0E0b.h5')
        # cm.model.load_weights(output_path+'/4E467b.h5') #end of normal
        # cm.model.load_weights(file_address + '/4E45b.h5')  # end of sample_selection
        # cm.model.load_weights(output_path+'/1E467b.h5') #end of normal

        vmodel = cm.model.get_weights()
        RATIO = 100000
        target = resize_model(vmodel)
        ntarget = normalize(target, RATIO)

        eds = []
        wds = []
        cds = []
        i = 0

        pre_previous = []
        previous = []
        cds_dif = []

        for file in file_ls:
            extend = os.path.splitext(file)[-1][1:]
            if (extend != 'h5'):
                continue

            print(file_address + '/' + file)
            cm.model.load_weights(file_address + '/' + file)
            vcurrent = resize_model(cm.model.get_weights())

            E_dis = np.linalg.norm(np.array(target) - np.array(vcurrent), ord=2)
            eds.append(E_dis)

            C_dis = cosine_similarity(vcurrent, target)
            cds.append(C_dis)

            nvcurrent = normalize(vcurrent, RATIO)
            W_dis = ss.wasserstein_distance(np.array(ntarget).transpose(), np.array(nvcurrent).transpose())
            wds.append(W_dis)

            # print(C_dis)
            # print(E_dis)
            # print(W_dis)
            print(file_address + '/' + file)
            if len(pre_previous) == 0:
                pre_previous = vcurrent
                continue
            elif len(previous) == 0:
                previous = vcurrent
                continue
            else:
                pp_dif = np.array(previous) - np.array(pre_previous)
                p_dif = np.array(vcurrent) - np.array(previous)
                C_dis_diff = cosine_similarity(pp_dif, p_dif)
                cds_dif.append(C_dis_diff)
                pre_previous = previous
                previous = vcurrent

            # print(C_dis_diff)
            i = i + 1
            print('*********************************')
            # if i>10:
            #     break
        # list = []
        # list.append(eds)
        # list.append(wds)
        # name_list = ['ed','wd']
        # draw_plot(list,name_list)

        plt.plot(range(len(eds)), eds)
        dr.save_data(eds, 'record/' + file_address + '/E_dis_random.csv')
        plt.savefig('record/' + file_address + '/E_dis_random.png')
        plt.close()

        plt.plot(range(len(wds)), wds)
        dr.save_data(wds, 'record/' + file_address + '/W_dis_random.csv')
        plt.savefig('record/' + file_address + '/W_dis_random.png')
        plt.close()

        plt.plot(range(len(cds)), cds)
        dr.save_data(cds, 'record/' + file_address + '/C_dis_random.csv')
        plt.savefig('record/' + file_address + '/C_dis_random.png')
        plt.close()

        plt.plot(range(len(cds_dif)), cds_dif)
        dr.save_data(cds_dif, 'record/' + file_address + '/C_dis_diff_random.csv')
        plt.savefig('record/' + file_address + '/C_dis_dif_random.png')
        plt.close()


def single_test():
    file_address = input('enter output path:')
    file_ls = os.listdir(file_address)
    file_ls.sort(key=sort_key)
    # file_ls.sort(key=sort_key)

    # vm.vae.load_weights(output_path+'/0E0b.h5')
    # cm.model.load_weights(output_path+'/4E467b.h5') #end of normal
    # cm.model.load_weights(file_address + '/4E45b.h5')  # end of sample_selection
    cm.model.load_weights(file_address + '/4E467b.h5')  # end of normal

    vmodel = cm.model.get_weights()
    RATIO = 100000
    target = resize_model(vmodel)
    ntarget = normalize(target, RATIO)

    eds = []
    wds = []
    cds = []
    i = 0

    pre_previous = []
    previous = []
    cds_dif = []

    for file in file_ls:
        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue
        cm.model.load_weights(file_address + '/' + file)
        print(file)
        vcurrent = resize_model(cm.model.get_weights())

        E_dis = np.linalg.norm(np.array(target) - np.array(vcurrent), ord=2)
        eds.append(E_dis)

        C_dis = cosine_similarity(vcurrent, target)
        cds.append(C_dis)

        nvcurrent = normalize(vcurrent, RATIO)
        W_dis = ss.wasserstein_distance(np.array(ntarget).transpose(), np.array(nvcurrent).transpose())
        wds.append(W_dis)

        # print(C_dis)
        # print(E_dis)
        # print(W_dis)
        print(file)
        if len(pre_previous) == 0:
            pre_previous = vcurrent
            continue
        elif len(previous) == 0:
            previous = vcurrent
            continue
        else:
            pp_dif = np.array(previous) - np.array(pre_previous)
            p_dif = np.array(vcurrent) - np.array(previous)
            C_dis_diff = cosine_similarity(pp_dif, p_dif)
            cds_dif.append(C_dis_diff)
            pre_previous = previous
            previous = vcurrent

        # print(C_dis_diff)
        i = i + 1
        print('*********************************')
        # if i>10:
        #     break
    # list = []
    # list.append(eds)
    # list.append(wds)
    # name_list = ['ed','wd']
    # draw_plot(list,name_list)

    plt.plot(range(len(eds)), eds)
    dr.save_data(eds, 'record/' + file_address + '/E_dis_random.csv')
    plt.savefig('record/' + file_address + '/E_dis_random.png')
    plt.close()

    plt.plot(range(len(wds)), wds)
    dr.save_data(wds, 'record/' + file_address + '/W_dis_random.csv')
    plt.savefig('record/' + file_address + '/W_dis_random.png')
    plt.close()

    plt.plot(range(len(cds)), cds)
    dr.save_data(cds, 'record/' + file_address + '/C_dis_random.csv')
    plt.savefig('record/' + file_address + '/C_dis_random.png')
    plt.close()

    plt.plot(range(len(cds_dif)), cds_dif)
    dr.save_data(cds_dif, 'record/' + file_address + '/C_dis_diff_random.csv')
    plt.savefig('record/' + file_address + '/C_dis_dif_random.png')
    plt.close()


def multilabel_test(index, length):
    file_address = 'multi_label' + str(index)
    file_ls = os.listdir(file_address)

    def sort_key_multi_label(e):
        epoch_str = e.split('E')
        batch_str = epoch_str[1].split('b')
        return int(epoch_str[0]) * length + int(batch_str[0])

    file_ls.sort(key=sort_key_multi_label)
    # file_ls.sort(key=sort_key)

    # vm.vae.load_weights(output_path+'/0E0b.h5')
    # cm.model.load_weights(output_path+'/4E467b.h5') #end of normal
    # cm.model.load_weights(file_address + '/4E45b.h5')  # end of sample_selection
    # cm.model.load_weights(output_path+'/1E467b.h5') #end of normal
    cm.model.load_weights(file_address + '/4E' + str(length - 1) + 'b.h5')
    vmodel = cm.model.get_weights()
    RATIO = 100000
    target = resize_model(vmodel)
    ntarget = normalize(target, RATIO)

    eds = []
    wds = []
    cds = []
    i = 0

    pre_previous = []
    previous = []
    cds_dif = []

    for file in file_ls:
        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue

        print(file_address + '/' + file)
        cm.model.load_weights(file_address + '/' + file)
        vcurrent = resize_model(cm.model.get_weights())

        E_dis = np.linalg.norm(np.array(target) - np.array(vcurrent), ord=2)
        eds.append(E_dis)

        C_dis = cosine_similarity(vcurrent, target)
        cds.append(C_dis)

        nvcurrent = normalize(vcurrent, RATIO)
        W_dis = ss.wasserstein_distance(np.array(ntarget).transpose(), np.array(nvcurrent).transpose())
        wds.append(W_dis)

        # print(C_dis)
        # print(E_dis)
        # print(W_dis)
        if len(pre_previous) == 0:
            pre_previous = vcurrent
            continue
        elif len(previous) == 0:
            previous = vcurrent
            continue
        else:
            pp_dif = np.array(previous) - np.array(pre_previous)
            p_dif = np.array(vcurrent) - np.array(previous)
            C_dis_diff = cosine_similarity(pp_dif, p_dif)
            cds_dif.append(C_dis_diff)
            pre_previous = previous
            previous = vcurrent

        # print(C_dis_diff)
        i = i + 1
        print('*********************************')
        # if i>10:
        #     break
    # list = []
    # list.append(eds)
    # list.append(wds)
    # name_list = ['ed','wd']
    # draw_plot(list,name_list)

    plt.plot(range(len(eds)), eds)
    dr.save_data(eds, 'record/' + file_address + '/E_dis_random.csv')
    plt.savefig('record/' + file_address + '/E_dis_random.png')
    plt.close()

    plt.plot(range(len(wds)), wds)
    dr.save_data(wds, 'record/' + file_address + '/W_dis_random.csv')
    plt.savefig('record/' + file_address + '/W_dis_random.png')
    plt.close()

    plt.plot(range(len(cds)), cds)
    dr.save_data(cds, 'record/' + file_address + '/C_dis_random.csv')
    plt.savefig('record/' + file_address + '/C_dis_random.png')
    plt.close()

    plt.plot(range(len(cds_dif)), cds_dif)
    dr.save_data(cds_dif, 'record/' + file_address + '/C_dis_diff_random.csv')
    plt.savefig('record/' + file_address + '/C_dis_dif_random.png')
    plt.close()


def Euclidean_distance(array1, array2):
    return np.linalg.norm(array1 - array2, ord=2)


def compare_two_sequence(path1, path2, metric, metric_name):
    file_address = 'record/' + path1 + '_' + path2

    if not os.path.exists(file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(file_address)  # makedirs 创建文件时如果路径不存在会创建这个路径
        os.makedirs('record/' + file_address)
        print
        "---  new folder...  ---"
    else:
        print
        "---  There is this folder!  ---"

    file_ls1 = os.listdir(path1)
    file_ls1.sort(key=sort_key)
    file_ls2 = os.listdir(path2)
    file_ls2.sort(key=sort_key)
    distance = []
    for file in file_ls1:
        if file not in file_ls2:
            break
        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue
        print(path1 + '/' + file, '  ', path2 + '/' + file)

        cm.model.load_weights(path1 + '/' + file)
        vcurrent1 = resize_model(cm.model.get_weights())
        cm.model.load_weights(path2 + '/' + file)
        vcurrent3 = resize_model(cm.model.get_weights())

        dis = metric(np.array(vcurrent1), np.array(vcurrent3))
        print(dis)
        distance.append(dis)
    dr.save_data(distance, file_address + '/' + metric_name + '.csv')


def analyse_sequence(path1, target_path, metric, metric_name):
    file_address = 'record/' + path1

    if not os.path.exists(file_address):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(file_address)  # makedirs 创建文件时如果路径不存在会创建这个路径
        os.makedirs('record/' + file_address)
        print
        "---  new folder...  ---"
    else:
        print
        "---  There is this folder!  ---"

    file_ls1 = os.listdir(path1)
    file_ls1.sort(key=sort_key)

    distance = []
    cm.model.load_weights(target_path)
    targetv = resize_model(cm.model.get_weights())
    for file in file_ls1:
        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue

        cm.model.load_weights(path1 + '/' + file)
        vcurrent1 = resize_model(cm.model.get_weights())

        dis = metric(vcurrent1, targetv)
        distance.append(dis)
        print(dis)

    dr.save_data(distance, file_address + '/' + metric_name + '.csv')


def compare_two_sequence_accuracy(path1, path2):
    file_address = 'record/' + path1 + '_' + path2
    os.makedirs(file_address)
    file_ls1 = os.listdir(path1)
    file_ls1.sort(key=sort_key)
    file_ls2 = os.listdir(path2)
    file_ls2.sort(key=sort_key)
    loss_1 = []
    loss_2 = []
    ac_1 = []
    ac_2 = []
    for file in file_ls1:
        if file not in file_ls2:
            break

        extend = os.path.splitext(file)[-1][1:]
        if (extend != 'h5'):
            continue
        print(path1 + '/' + file, '  ', path2 + '/' + file)

        cm.model.load_weights(path1 + '/' + file)
        loss1, accuracy1 = accuracy(cm.model)
        print(loss1, ' ', accuracy1)
        cm.model.load_weights(path2 + '/' + file)
        loss2, accuracy2 = accuracy(cm.model)
        print(loss2, ' ', accuracy2)
        loss_1.append(loss1)
        loss_2.append(loss2)
        ac_1.append(accuracy1)
        ac_2.append(accuracy2)
    dr.save_data(ac_1, file_address + '/accuracy_1.csv')
    dr.save_data(ac_2, file_address + '/accuracy_2.csv')
    dr.save_data(loss_1, file_address + '/loss1.csv')
    dr.save_data(loss_2, file_address + '/loss2.csv')


# group_test(10)
# single_test()
# multi = [[9,422]]
# for i in multi:
#     multilabel_test(i[0],i[1])

# single_accuracy_test()

# analyse_sequence('cnn_standard1','cnn_standard1/0E467b.h5',kl.KL_div,'KL_div')
# a = list(range(60000))
analyse_sequence('cnn_error_label', 'cnn_error_label/0E467b.h5', kl.KL_div, 'KL_div')
# b = list(range(60000))
# random.shuffle(a)
# print(KL_div(np.transpose(a), np.transpose(b)))
# compare_two_sequence('cnn_standard','cnn_standard1',KL_div,'KL_div')
