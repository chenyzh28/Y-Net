import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from networks.ynet import Res_YNet
from dataset.datasets import Voc2012

import timeit
from tensorboardX import SummaryWriter
from utils.criterion import CriterionDSN, CriterionOhemDSN, CriterionExtDSN
from utils.encoding import DataParallelModel, DataParallelCriterion
import ss_transfroms as tr
from torchvision import transforms

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 16
DATA_DIRECTORY = './dataset/data/voc/VOCdevkit/VOC2012/'
DATA_LIST_PATH = './dataset/list/voc/train_aug.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '600, 600'
LEARNING_RATE = 7e-3
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 40000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_init.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots_voc/'
WEIGHT_DECAY = 0.0005
SE_Weight = 0.2


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Y-Net Training")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default='1,2,6,7',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="choose the number of recurrence.")
    parser.add_argument("--ft", type=bool, default=False,
                        help="fine-tune the model with large input size.")
    parser.add_argument("--os", type=int, default=16,
                        help="output stride.")
    parser.add_argument("--se_weight", type=float, default=SE_Weight,
                        help="weight for extra loss.")

    parser.add_argument("--ohem", type=str2bool, default='False',
                        help="use hard negative mining")
    parser.add_argument("--ohem-thres", type=float, default=0.6,
                        help="choose the samples with correct probability under the threshold.")
    parser.add_argument("--ohem-keep", type=int, default=200000,
                        help="choose the samples with correct probability under the threshold.")
    parser.add_argument("--ext", type=str2bool, default='False',
                        help="use hard negative mining")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def main():
    """Create the model and start the training."""
    writer = SummaryWriter(args.snapshot_dir)

    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    train_transforms = transforms.Compose([tr.RandomSized(input_size),
                                           tr.RandomRotate(15),
                                           tr.RandomHorizontalFlip(),
                                           tr.Normalize(mean=(104.00698793, 116.66876762, 122.67891434)),
                                           tr.ToTensor()
                                           ])
    training_data = Voc2012('./dataset/data/voc/VOCdevkit/VOC2012', 'train_aug', transform=train_transforms, max_iter=args.num_steps * args.batch_size)
    trainloader = data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)


    # Create network.
    res_ynet = Res_YNet(num_classes=args.num_classes, os=args.os)

    # # Loading parameters from pretrained model, but change the module name
    saved_state_dict = torch.load(args.restore_from)
    new_params = res_ynet.state_dict().copy()
    for k, v in res_ynet.state_dict().items():
        if k in saved_state_dict.keys():
            v = saved_state_dict[k]
            new_params[k] = v
        elif '_1' in k:
            name = k.split('_1')[0] + k.split('_1')[1]
            v = saved_state_dict[name]
            new_params[k] = v
        elif '_2' in k:
            name = k.split('_2')[0] + k.split('_2')[1]
            v = saved_state_dict[name]
            new_params[k] = v
    assert len(new_params.keys()) == len(res_ynet.state_dict().keys())
    res_ynet.load_state_dict(new_params)

    model = DataParallelModel(res_ynet)
    model.train()
    model.float()
    model.cuda()

    if args.ohem:
        criterion = CriterionOhemDSN(thresh=args.ohem_thres, min_kept=args.ohem_keep)
    elif args.ext:
        criterion = CriterionExtDSN(se_weight=args.se_weight)
    else:
        criterion = CriterionDSN()
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, res_ynet.parameters()), 'lr': args.learning_rate}],
        lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()


    for i_iter, batch in enumerate(trainloader):
        i_iter += args.start_iters
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
        if torch_ver == "0.3":
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()

        lr = adjust_learning_rate(optimizer, i_iter)
        preds = model(images)

        loss = criterion(preds, labels.squeeze(1))
        loss.backward()
        optimizer.step()

        if i_iter % 100 == 0:
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

        print('iter = {} of {} completed, loss = {}'.format(i_iter, args.num_steps, loss.data.cpu().numpy()))

        if i_iter >= args.num_steps - 1:
            print('save model ...')
            torch.save(res_ynet.state_dict(), osp.join(args.snapshot_dir, 'VOC_scenes_' + str(args.num_steps) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save(res_ynet.state_dict(), osp.join(args.snapshot_dir, 'VOC_scenes_' + str(i_iter) + '.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
