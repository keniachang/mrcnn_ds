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


filename = 'wd.csv'
file_path = './cfs0215T1912_exps/' + filename
v = (np.array(read_csv(file_path))).astype(float)
plt.plot(v, label='wd')

# filename = 'v.csv'
# file_path = './cfs0215T1912_exps/' + filename
# v = (np.array(read_csv(file_path))).astype(float)
# plt.plot(v, label='v')
#
# filename = 'vp3.1.csv'
# file_path = './cfs0215T1912_exps/' + filename
# vp = (np.array(read_csv(file_path))).astype(float)
# vp_sum = np.sum(vp)
# print('sum of vp:', vp_sum)
# vp = vp[:(v.shape[0])]
# plt.plot(vp, label='vp')

# dv = np.subtract(vp, v)
# dv_sum = np.sum(dv)
# print('sum of dv_random:', dv_sum)
# plt.plot(dv, label='delta velocity')

plt.legend()
plt.show()
