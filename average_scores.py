# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn

import numpy as np
import pdb

def valid():
    files_scores = ['/home/mcg/cxk/action-recognition-zoo/results/tsn-flow/output/flow.npz', '/home/mcg/cxk/action-recognition-zoo/results/tsn-rgb/output/rgb.npz']

    allsum = np.zeros([11522, 174])
    labels = []
    for filename in files_scores:
        print(filename)
        data = np.load(filename)
        scores = data['scores']
        # print(scores.shape)
        ss = scores[:, 0]
        ll = scores[:, 1]
        labels.append(ll)
        ss = [x.reshape(174) for x in ss]

        allsum += ss

    preds = np.argmax(allsum,axis=1)

    num_correct = np.sum(preds == labels)
    acc = num_correct * 1.0 / preds.shape[0]
    print('acc=%.3f' % (acc))



if __name__ == '__main__':
    valid()
