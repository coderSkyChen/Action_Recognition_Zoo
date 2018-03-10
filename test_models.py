# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn
# Note that when testing TSN, num_segments=1, and num_segments>1 only on traing phrase.

import os
import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from models import *
from transforms import *
from dataset import *
import pdb
from torch.nn import functional as F


# options
parser = argparse.ArgumentParser(description="testing on the full validation set")
parser.add_argument('--model', type=str, choices=['TwoStream', 'TSN'])
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--weights', type=str)
parser.add_argument('--train_id', type=str)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='TSN-DI',
                    choices=['avg', 'TRN','TRNmultiscale', 'TSN-DI'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--softmax', type=int, default=0)

args = parser.parse_args()

def return_something_path(modality):
    filename_categories = '/home/mcg/cxk/dataset/somthing-something/category.txt'
    if modality == 'RGB':
        root_data = '/home/mcg/cxk/dataset/somthing-something/something-rgb'
        filename_imglist_train = '/home/mcg/cxk/dataset/somthing-something/train_videofolder_rgb.txt'
        filename_imglist_val = '/home/mcg/cxk/dataset/somthing-something/val_videofolder_rgb.txt'

        prefix = '{:05d}.jpg'
    else:
        root_data = '/home/mcg/cxk/dataset/somthing-something/something-optical-flow'
        filename_imglist_train = '/home/mcg/cxk/dataset/somthing-something/train_videofolder_flow.txt'
        filename_imglist_val = '/home/mcg/cxk/dataset/somthing-something/val_videofolder_flow.txt'

        prefix = '{:s}_{:05d}.jpg'

    with open(filename_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    return categories, filename_imglist_train, filename_imglist_val, root_data, prefix

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 0, True, True)
    # pred = pred.t()
    # print(target)
    # print(pred)
    correct = pred.eq(target.view(-1).expand(pred.size()))
    # print(correct)
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / (batch_size)))
    return res


# time.sleep(3400)

categories, args.train_list, args.val_list, args.root_path, prefix = return_something_path(args.modality)
num_class = len(categories)

if args.model == 'TwoStream':
    net = TwoStream(num_class, args.modality, base_model=args.arch)
elif args.model == 'TSN':
    net = TSN(num_class, 1, args.modality, base_model=args.arch)

checkpoint = torch.load(os.path.join('/home/mcg/cxk/action-recognition-zoo/results', args.train_id, 'model', args.weights))
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))
if args.model == 'TwoStream':
    data_loader = torch.utils.data.DataLoader(
            TwoStreamDataSet(args.root_path, args.val_list, num_segments=args.test_segments,
                       new_length=1 if args.modality == "RGB" else 5,
                       modality=args.modality,
                       image_tmpl=prefix,
                       test_mode=True,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                           ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                       ])),
            batch_size=1, shuffle=False,
            num_workers=args.workers * 2, pin_memory=True)
elif args.model == 'TSN':
    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, args.val_list, num_segments=args.test_segments,
                       new_length=1 if args.modality == "RGB" else 5,
                       modality=args.modality,
                       image_tmpl=prefix,
                       test_mode=True,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                           ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                       ])),
            batch_size=1, shuffle=False,
            num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))


#net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net = torch.nn.DataParallel(net.cuda())
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []


def eval_video(video_data):
    i, data, label = video_data

    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    # data: bs* channelss * w *h   channelss=channels*frames

    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                        volatile=True)
    rst = net(input_var)
    rst = rst.data.cpu().numpy().copy()

    rst = rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, num_class)).mean(axis=0).reshape((num_class))

    return i, rst, label[0]


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

top1 = AverageMeter()
top5 = AverageMeter()

for i, (data, label) in data_gen:
    if i >= max_num:
        break
    rst = eval_video((i, data, label))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    prec1, prec5 = accuracy(torch.from_numpy(rst[1]), label, topk=(1, 5))
    top1.update(prec1[0], 1)
    top5.update(prec5[0], 1)
    print('video {} done, total {}/{}, average {:.3f} sec/video, moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i, i+1,
                                                                    len(data_loader),
                                                                    float(cnt_time) / (i+1), top1.avg, top5.avg))

video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

video_labels = [x[1] for x in output]


cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print('-----Evaluation is finished------')
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))

if args.save_scores is not None:

    if args.modality=='RGB':
        test_list = '/home/mcg/cxk/dataset/somthing-something/val_videofolder_rgb.txt'
    elif args.modality=='Flow':
        test_list = '/home/mcg/cxk/dataset/somthing-something/val_videofolder_flow.txt'
    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(test_list)]

    assert len(output) == len(name_list)

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}
    reorder_output = [None] * len(name_list)
    # reorder_label = [None] * len(output)
    reorder_pred = [None] * len(name_list)
    output_csv = []
    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        # reorder_label[idx] = video_labels[i]
        reorder_pred[idx] = video_pred[i]
        output_csv.append('%s;%s'%(name_list[i], categories[video_pred[i]]))

    np.savez(os.path.join('/home/mcg/cxk/action-recognition-zoo/results', args.train_id, 'output', args.save_scores), scores=reorder_output, predictions=reorder_pred)

    # with open(os.path.join('/home/mcg/cxk/action-recognition-zoo/results', args.train_id, 'output', args.save_scores+'.csv'), 'w') as f:
    #     f.write('\n'.join(output_csv))


