# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn

import numpy as np
import pdb
from sklearn.metrics import confusion_matrix

def valid():
    files_scores = ['./flow.npz', './rgb.npz']

    allsum = np.zeros([3783, 101])
    labels = []
    for filename in files_scores:
        print(filename)
        data = np.load(filename)
        scores = data['scores']
        ll = data['labels']
        # print(scores.shape)
        ss = scores[:, 0]

        labels = ll
        ss = [x.reshape(101) for x in ss]
        ss = np.array(ss)
        allsum += ss


    preds = np.argmax(allsum,axis=1)
    num_correct = np.sum(preds == labels)
    acc = num_correct * 1.0 / preds.shape[0]
    a=preds==labels
    print(preds.shape, labels.shape)
    print('acc=%.4f' % (acc))

def mean_class_accuracy():

    data = np.load('rgb.npz')
    scores = data['scores'][:, 0]
    ss = [x.reshape(101) for x in scores]
    ss = np.array(ss)
    pred = np.argmax(ss, axis=1)
    labels = data['labels']
    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    #print("%.3f" % (sum(labels==pred)*1.0/len(labels)*100))

    cls_acc = cls_hit / cls_cnt
    with open('/home/mcg/cxk/dataset/UCF-101/classInd.txt') as f:
        lines = f.readlines()
    categories = [item.strip().split()[1] for item in lines]
 #   print('\n'.join(['%.3f' % (cls_acc[i]) for i in range(len(categories))]))
 #    print('\n'.join(['%s %.3f' % (categories[i], cls_acc[i]) for i in range(len(categories))]))
    for i in range(101):
        cf[i][i]=0
    #print(cf.argmax(axis=1))
    cm = cf.argmax(axis=1)
    print('\n'.join(['%s %s' % (categories[idx], categories[i]) for idx,i in enumerate(cm)]))
    return np.mean(cls_hit/cls_cnt)


if __name__ == '__main__':
    pass
    # valid()
    # mean_class_accuracy()