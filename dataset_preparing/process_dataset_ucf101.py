# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn

# processing the raw data of the UCF101 dataset
# From original files:classInd.txt,trainlist01.txt,testlist01.txt generate the meta files:
#   category.txt:               the list of categories.
#   trainlist_mid01.txt:      each row contains [videoname num_frames classIDX]
#   testlist_mid01.txt:       each row contains [videoname num_frames classIDX]
#

import os

rootdir = '/home/mcg/cxk/dataset/UCF101-frames-TSN/'
classInddir = os.path.join(rootdir, 'classInd.txt')
with open(classInddir) as f:
    lines = f.readlines()

categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line.split(' ')[-1])

with open(rootdir + 'category.txt', 'w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i

# for train
for idx in [1, 2, 3]:
    with open(os.path.join(rootdir, 'trainlist0%s.txt' % str(idx))) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        items = line.split(' ')
        folders.append(items[0].split('/')[-1])
        idx_categories.append(int(items[1]) - 1)
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curFolder = curFolder.split('.')[0]
        curIDX = idx_categories[i]
        # counting the number of frames in each video folders
        dir_files = os.listdir(os.path.join(rootdir, curFolder))
        imgs = []
        for dd in dir_files:
            if dd.find('img')!=-1:
                imgs.append(dd)
        output.append('%s %d %d' % (curFolder, len(imgs), curIDX))
        print('%d/%d' % (i, len(folders)))
    with open(os.path.join(rootdir, 'trainlist_mid0%s.txt' % str(idx)), 'w') as f:
        f.write('\n'.join(output))

# for test
for idx in [1, 2, 3]:
    with open(os.path.join(rootdir, 'testlist0%s.txt' % str(idx))) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        items = line.split('/')
        assert len(items) == 2
        folders.append(line.split('/')[-1])
        idx_categories.append(dict_categories[items[0]])
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curFolder = curFolder.split('.')[0]
        curIDX = idx_categories[i]
        # counting the number of frames in each video folders
        dir_files = os.listdir(os.path.join(rootdir, curFolder))
        imgs = []
        for dd in dir_files:
            if dd.find('img') != -1:
                imgs.append(dd)
        output.append('%s %d %d' % (curFolder, len(imgs), curIDX))
        print('%d/%d' % (i, len(folders)))
    with open(os.path.join(rootdir, 'testlist_mid0%s.txt' % str(idx)), 'w') as f:
        f.write('\n'.join(output))

