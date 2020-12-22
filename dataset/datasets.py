import os
import os.path as osp
import numpy as np
import cv2
from torch.utils import data
from PIL import Image
from scipy.misc import imread
from torchvision import transforms

class Voc2012(data.Dataset):
    def __init__(self, data_path, trainval="train_aug", transform=None, max_iter=40000):
        self.data_path = data_path
        self.transform = transform
        self.trainval = trainval
        self.max_iter = max_iter

        self.__init_classes()
        self.names = self.__dataset_info()

    def __getitem__(self, index):
        x = imread(self.data_path + '/JPEGImages/' + self.names[index] + '.jpg', mode='RGB')
        x = Image.fromarray(x)  # PIL

        if self.trainval != 'test':
            x_mask = imread(self.data_path + '/SegmentationClassAug/' + self.names[index] + '.png', mode='L')
            x_mask = Image.fromarray(x_mask)  # PIL

            sample = {'image': x, 'label': x_mask}
            if self.transform is not None:
                sample = self.transform(sample)
            x, x_mask = sample['image'], sample['label']
            return x, x_mask
        else:
            sample = {'image': x}
            if self.transform is not None:
                sample = self.transform(sample)
            x = sample['image']
            return x

    def __len__(self):
        return len(self.names)

    def __dataset_info(self):
        with open(self.data_path + '/list/' + self.trainval + '.txt') as f:
            annotations = f.readlines()  # 2913 num pic
        annotations = [n[:-1] for n in annotations]  # delete '\n'
        names = []
        for name in annotations:
            names.append(name)
        if not self.max_iter==None:
            names = names * int(np.ceil(float(self.max_iter) / len(names)))

        return names

    def __init_classes(self):
        self.classes = ('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))  # assign class_num:0,1,2

class Person_Part(data.Dataset):
    def __init__(self, data_path, train="train_convert", transform=None, max_iter=10000):
        self.data_path = data_path
        self.transform = transform
        self.train = train
        self.max_iter = max_iter

        self.__init_classes()
        self.names = self.__dataset_info()

    def __getitem__(self, index):
        x = imread('./dataset/data/voc/VOCdevkit/VOC2012' + '/JPEGImages/' + self.names[index] + '.jpg', mode='RGB')
        x = Image.fromarray(x)  # PIL

        x_mask = imread(self.data_path + '/pascal_person_part_gt_raw/' + self.names[index] + '.png', mode='L')
        x_mask = Image.fromarray(x_mask)  # PIL

        sample = {'image': x, 'label': x_mask}
        if self.transform is not None:
            sample = self.transform(sample)
        x, x_mask = sample['image'], sample['label']
        # y = torch.from_numpy(y)  # 20 classes
        return x, x_mask

    def __len__(self):
        return len(self.names)

    def __dataset_info(self):
        with open(self.data_path + '/list/' + self.train + '.txt') as f:
            annotations = f.readlines()  # 2913 num pic
        annotations = [n[:-1] for n in annotations]  # delete '\n'
        names = []
        for name in annotations:
            names.append(name)
        if not self.max_iter==None:
            names = names * int(np.ceil(float(self.max_iter) / len(names)))

        return names

    def __init_classes(self):
        self.classes = ('background', 'head', 'torso', 'arm', 'hand', 'leg', 'foot')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))  # assign class_num:0,1,2

class VOCDataTestSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(505, 505), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            self.files.append({
                "img": img_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean
        
        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
        image = image.transpose((2, 0, 1))
        return image, name, size


class ADE20K_train(data.Dataset):
    def __init__(self, data_path, transform=None, max_iter=40000):
        self.data_path = data_path
        self.transform = transform
        self.normalize = transforms.Normalize(
            mean = [0,485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
        self.max_iter = max_iter
        self.names = self.__dataset_info()

    def __getitem__(self, index):
        x = imread(self.data_path + '/images/training/' + self.names[index], mode='RGB')
        x = Image.fromarray(x)  # PIL

        x_mask = imread(self.data_path + '/annotations/training/' + self.names[index][:-4] + '.png', mode='L')
        x_mask = Image.fromarray(x_mask)  # PIL

        sample = {'image': x, 'label': x_mask}
        if self.transform is not None:
            sample = self.transform(sample)
        x, x_mask = sample['image'], sample['label']

        return x, x_mask

    def __len__(self):
        return len(self.names)

    def __dataset_info(self):

        names = []
        for filename in os.listdir(self.data_path + '/images/training/'):
            names.append(filename)
        if not self.max_iter==None:
            names = names * int(np.ceil(float(self.max_iter) / len(names)))
        return names