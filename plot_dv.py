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


file_dir = './cfs0215T1912_interrupt_exps/'

filename = 'wd_complete.csv'
file_path = file_dir + filename
wd = (np.array(read_csv(file_path))).astype(float)
plt.plot(wd, label='wd_complete')

filename = 'wdp31_temp_complete.csv'
file_path = file_dir + filename
wdp3_1 = (np.array(read_csv(file_path))).astype(float)
plt.plot(wdp3_1, label='wdp3.1_temp_complete')

# filename = 'v.csv'
# file_path = file_dir + filename
# v = (np.array(read_csv(file_path))).astype(float)
# plt.plot(v, label='v')
#
# filename = 'vp3.1.csv'
# file_path = file_dir + filename
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
