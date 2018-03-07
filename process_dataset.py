# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn

# processing the raw data of the video datasets (Something-something)
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#

import os


def train_and_val_test(modality='rgb'):
    dataset_name = ''
    if modality == 'rgb':
        dataset_name = 'something-rgb'
    elif modality == 'flow':
        dataset_name = 'something-optical-flow'
    else:
        print('error in modal type!')
        exit()
    rootdir = '/home/mcg/cxk/dataset/somthing-something/'

    prefix_name = 'something-something-v1'
    with open(rootdir + '%s-labels.csv' % prefix_name) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)
    categories = sorted(categories)
    with open(rootdir + 'category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = [rootdir + '%s-validation.csv' % prefix_name, rootdir + '%s-train.csv' % prefix_name]
    files_output = [rootdir + 'val_videofolder_%s.txt' % modality, rootdir + 'train_videofolder_%s.txt' % modality]
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(';')
            folders.append(items[0])
            idx_categories.append(os.path.join(dict_categories[items[1]]))
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join(rootdir, dataset_name, curFolder))
            output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
        # -----------test set-----------
        with open(rootdir + 'something-something-v1-test.csv') as f:
            lines = f.readlines()
        output = []
        for idx, i in enumerate(lines):
            floder = i.strip()
            files = os.listdir(rootdir + dataset_name + '/%s' % floder)
            output.append('%s %d' % (floder, len(files)))
            print('%d/%d' % (idx, len(lines)))
        with open(rootdir + 'test_videofolder_%s.txt' % modality, 'w') as f:
            f.write('\n'.join(output))


if __name__ == '__main__':
    train_and_val_test(modality='flow')
    train_and_val_test(modality='rgb')
