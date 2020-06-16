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
    file_dir = './coco_dog_ap/'

    filename = 'cocoA1_dog150_full_acc.csv'
    file_path = file_dir + filename
    wd = (np.array(read_csv(file_path))).astype(float)
    plt.plot(wd, label='A1_dog_acc')

    filename = 'cocoA80_dog150_full_acc.csv'
    file_path = file_dir + filename
    wd = (np.array(read_csv(file_path))).astype(float)
    plt.plot(wd, label='A80_dog_acc')

    plt.legend()
    plt.show()
