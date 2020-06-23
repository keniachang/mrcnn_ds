import csv
import numpy as np
import matplotlib.pyplot as plt


def read_csv(path):
    csv_file = open(path, "r")
    reader = csv.reader(csv_file)
    result = []
    for item in reader:
        result.append(item)
    csv_file.close()
    return result


if __name__ == '__main__':
    # plot Networks A & B accuracies on dog data
    file_dir = './cocoABC_dog_ap/'

    # filename = 'cocoA1_dog150_full_acc.csv'
    # file_path = file_dir + filename
    # acc = (np.array(read_csv(file_path))).astype(float)
    # plt.plot(acc, label='A1_dog_acc')

    # filename = 'cocoA80_dog150_full_acc.csv'
    # file_path = file_dir + filename
    # acc = (np.array(read_csv(file_path))).astype(float)
    # plt.plot(acc, label='A80_dog_acc')
    #
    # filename = 'cocoB80_dog150_full_acc2.csv'
    # file_path = file_dir + filename
    # acc = (np.array(read_csv(file_path))).astype(float)
    # plt.plot(acc, label='B80_dog_mix_acc')
    #
    # filename = 'cocoB80_dog150_pure_full_acc2.csv'
    # file_path = file_dir + filename
    # acc = (np.array(read_csv(file_path))).astype(float)
    # plt.plot(acc, label='B80_dog_pure_acc')

    filename = 'cocoA80_dog150_all_pure_full_acc.csv'
    file_path = file_dir + filename
    acc = (np.array(read_csv(file_path))).astype(float)
    plt.plot(acc, label='A80_apdog_acc')

    filename = 'cocoB80_dog150_all_pure_full_acc.csv'
    file_path = file_dir + filename
    acc = (np.array(read_csv(file_path))).astype(float)
    plt.plot(acc, label='B80_apdog_acc')

    filename = 'cocoC80_dog150_all_pure_full_acc.csv'
    file_path = file_dir + filename
    acc = (np.array(read_csv(file_path))).astype(float)
    plt.plot(acc, label='C80_apdog_acc')

    # # plot Networks A & B KL_div on their own
    # file_dir = 'cocoABC_KL/'
    #
    # filename = 'A80_head_w1_w150_KL.csv'
    # file_path = file_dir + filename
    # acc = (np.array(read_csv(file_path))).astype(float)
    # plt.plot(acc, label='A80_1-150_KL')
    #
    # filename = 'B80_head_w1_w150_KL.csv'
    # file_path = file_dir + filename
    # acc = (np.array(read_csv(file_path))).astype(float)
    # plt.plot(acc, label='B80_1-150_KL')
    #
    # filename = 'C80_head_w1_w150_KL.csv'
    # file_path = file_dir + filename
    # acc = (np.array(read_csv(file_path))).astype(float)
    # plt.plot(acc, label='C80_1-150_KL')

    plt.legend()
    plt.show()

    print('Finish process.')
