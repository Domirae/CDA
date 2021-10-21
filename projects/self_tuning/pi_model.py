"""
@author: Yifei Ji, Baixu Chen
@contact: jiyf990330@163.com, cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import sys
import argparse
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

sys.path.append('../..')
from ssllib.pi_model import sigmoid_rampup, SoftmaxMSELoss
from common.modules.classifier import Classifier
from common.vision.transforms import ResizeImage, MultipleApply
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.data import ForeverDataIterator
from common.utils.logger import CompleteLogger

sys.path.append('.')
import datasets
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_augmentation = [
        T.RandomResizedCrop(224, scale=(0.2, 1.)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ]

    train_transform = MultipleApply([T.Compose(train_augmentation), T.Compose(train_augmentation)])

    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]

    # labeled dataset
    labeled_train_dataset = dataset(root=args.root, split='train', sample_rate=args.sample_rate,
                                    download=True, transform=train_transform)
    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=True)
    labeled_train_iter = ForeverDataIterator(labeled_train_loader)

    # unlabeled dataset
    unlabeled_train_dataset = dataset(root=args.root, split='train', sample_rate=args.sample_rate,
                                      download=True, transform=train_transform, unlabeled=True)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True)
    unlabeled_train_iter = ForeverDataIterator(unlabeled_train_loader)

    val_dataset = dataset(root=args.root, split='test', sample_rate=100, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch)
    num_classes = labeled_train_dataset.num_classes
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = Classifier(backbone, num_classes, pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier = torch.nn.DataParallel(classifier)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.module.get_parameters(args.lr), args.lr, momentum=0.9, weight_decay=args.wd,
                    nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epochs, gamma=args.lr_gamma)

    # resume from the best checkpoint
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc1 = utils.validate(val_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.0
    for epoch in range(args.epochs):
        # print lr
        print(lr_scheduler.get_lr())

        # train for one epoch
        train(labeled_train_iter, unlabeled_train_iter, classifier, optimizer, epoch, args)

        # update lr
        lr_scheduler.step()

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))
    logger.close()


def train(labeled_train_iter: ForeverDataIterator, unlabeled_train_iter: ForeverDataIterator, model: nn.DataParallel,
          optimizer: SGD, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    con_losses = AverageMeter('Con Loss', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, con_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    consistency_criterion = SoftmaxMSELoss().to(device)
    end = time.time()
    print('sigmoid rampup: ', sigmoid_rampup(epoch, args.rampup_length))
    for i in range(args.iters_per_epoch):
        labeled_x, labels = next(labeled_train_iter)
        labeled_x0, labeled_x1 = labeled_x[0], labeled_x[1]
        labeled_x0 = labeled_x0.to(device)
        labeled_x1 = labeled_x1.to(device)
        batch_size = labeled_x0.shape[0]
        labels = labels.to(device)

        unlabeled_x, _ = next(unlabeled_train_iter)
        unlabeled_x0, unlabeled_x1 = unlabeled_x[0], unlabeled_x[1]
        unlabeled_x0 = unlabeled_x0.to(device)
        unlabeled_x1 = unlabeled_x1.to(device)

        concat_x = torch.cat([labeled_x0, labeled_x1, unlabeled_x0, unlabeled_x1], dim=0)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y, f = model(concat_x)
        # cross entropy loss
        cls_loss = F.cross_entropy(y[:batch_size], labels)

        labeled_y0, labeled_y1, unlabeled_y0, unlabeled_y1 = torch.chunk(y, 4, dim=0)
        y0 = torch.cat([labeled_y0, unlabeled_y0], dim=0)
        y1 = torch.cat([labeled_y1, unlabeled_y1], dim=0)
        # consistency loss
        consistency_loss = consistency_criterion(y0, y1) * args.trade_off * sigmoid_rampup(epoch, args.rampup_length)
        loss = cls_loss + consistency_loss

        # measure accuracy and record loss
        losses.update(loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        con_losses.update(consistency_loss.item(), batch_size)
        cls_acc = accuracy(y[:batch_size], labels)[0]
        cls_accs.update(cls_acc.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='Pi Model for Semi Supervised Learning')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA',
                        help='dataset: ' + ' | '.join(dataset_names))
    parser.add_argument('-sr', '--sample-rate', default=100, type=int,
                        metavar='N',
                        help='sample rate of training dataset (default: 100)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=0.1, type=float,
                        help='trade-off for consistency loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N',
                        help='mini-batch size (default: 48)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay-epochs', type=int, default=(12,), nargs='+', help='epochs to decay lr')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--rampup-length', default=10, type=int, help='rampup length')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    args = parser.parse_args()
    main(args)
