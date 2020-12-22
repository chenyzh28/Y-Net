import argparse
from scipy import ndimage
import numpy as np
import json

import torch
from torch.autograd import Variable
from networks.ynet import Res_YNet
import os
from math import ceil
from PIL import Image as PILImage

import torch.nn as nn
import cv2
from scipy.misc import imread
from utils.dcrf import dense_crf

import timeit

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

DATA_DIRECTORY = './dataset/data/voc/VOCdevkit/VOC2012/'
DATA_LIST_PATH = './dataset/list/voc/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 21
NUM_STEPS = 500
INPUT_SIZE = '600, 600'
RESTORE_FROM = './snapshots_voc/VOC_scenes_40000.pth'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='4',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="choose the number of recurrence.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--whole", type=bool, default=True,
                        help="use whole input size.")
    parser.add_argument("--use_crf", type=bool, default=False,
                        help="use densecrf or not.")
    parser.add_argument("--os", type=int, default=16,
                        help="output stride.")
    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_sliding(net, image, tile_size, classes):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    overlap = 1 / 3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            print("Predicting tile %i" % tile_counter)
            padded_prediction = net(Variable(torch.from_numpy(padded_img).float(), volatile=True).cuda())
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[1]
            padded_prediction = interp(padded_prediction).cpu().data[0].numpy().transpose(1, 2, 0)
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    full_probs /= count_predictions
    return full_probs


def predict_whole(net, image, tile_size):
    image = torch.from_numpy(image).float()
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    prediction = net(image.cuda())
    # pdb.set_trace()
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)
    return prediction


def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation, recurrence):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image = image.data
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((H_, W_, classes))
    final_output = []
    for scale in scales:
        scale = float(scale)
        print("Predicting image scaled by %f" % scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        scaled_probs = predict_whole(net, scale_image, tile_size, recurrence)
        if flip_evaluation == True:
            flip_scaled_probs = predict_whole(net, scale_image[:, :, :, ::-1].copy(), tile_size, recurrence)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:, ::-1, :])
        final_output.append(scaled_probs)
        # Avg Merging
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # gpu0 = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    h, w = map(int, args.input_size.split(','))
    if args.whole:
        input_size = (h, w)
    else:
        input_size = (h, w)

    model = Res_YNet(num_classes=args.num_classes, os=args.os)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    img_list = open(args.data_list).readlines()

    data_list = []
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    # palette = get_palette(256)

    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
               128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
               64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    output_dir = "./outputs_voc/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = timeit.default_timer()

    for i in img_list:
        imgs = imread(args.data_dir + '/JPEGImages/' + i[:-1] + '.jpg', mode='RGB')
        imgs = PILImage.fromarray(imgs)  # PIL
        mask_labels = imread(args.data_dir + '/SegmentationClassAug/' + i[:-1] + '.png', mode='L')

        imgs = np.array(imgs).astype(np.float32)
        imgs -= IMG_MEAN

        imgs_size_0, imgs_size_1 = imgs.shape[0], imgs.shape[1]
        imgs = cv2.resize(imgs, input_size).astype(np.float32)
        img_bc = np.zeros((input_size[0], input_size[1], 3))
        img_bc[:imgs.shape[0], :imgs.shape[1], :] = imgs

        image = img_bc[np.newaxis, :].transpose(0, 3, 1, 2)

        with torch.no_grad():
            if args.whole:
                output = predict_multiscale(model, image, input_size, [1.0],
                                            args.num_classes, False, args.recurrence)
            else:
                output = predict_sliding(model, image, input_size, args.num_classes, True, args.recurrence)
        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        seg_pred = cv2.resize(seg_pred, (imgs_size_1, imgs_size_0))

        if args.use_crf:
            crf = dense_crf(21)
            imgs_cpu_crf = imgs
            imgs_cpu_crf += IMG_MEAN
            imgs_cpu_crf = cv2.resize(imgs_cpu_crf, (imgs_size_1, imgs_size_0)).astype(np.float32)
            full_img = PILImage.fromarray(np.uint8(imgs_cpu_crf))
            seg_pred = np.uint8(crf(np.array(full_img).astype(np.uint8), seg_pred))

        output_im = PILImage.fromarray(seg_pred).convert('P')
        output_im.putpalette(palette)
        output_im.save(output_dir + i[:-1] + '.png')

        seg_gt = np.asarray(mask_labels[:imgs_size_0, :imgs_size_1], dtype=np.int)
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)

    end = timeit.default_timer()
    print(end - start, 'seconds')

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()
    print({'meanIU': mean_IU, 'IU_array': IU_array})
    with open('result.txt', 'w') as f:
        f.write(json.dumps({'meanIU': mean_IU, 'IU_array': IU_array.tolist()}))

if __name__ == '__main__':
    main()
