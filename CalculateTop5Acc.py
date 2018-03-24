
import torch
from torchvision import datasets
import os
import AverageMeter as am
import Augmentation as ag
from torch.autograd import Variable

class CalculateTop5Acc():

    def __init__(self, aug, model, use_gpu = True):
       #Define augmentation strategy
        self.augmentation_strategy = ag.Augmentation(aug);
        self.data_transforms = self.augmentation_strategy.applyTransforms();
        self.model = model;
        self.model.train(False)
        self.use_gpu = use_gpu

    def validate(self, datapath, batch_size):
        top1 = am.AverageMeter()
        top5 = am.AverageMeter()

        # Root directory
        data_dir = datapath;

        ######### Data Loader ###########
        dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), self.data_transforms[x])
                 for x in ['val']}
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                       shuffle=True, num_workers=16)
                        # set num_workers higher for more cores and faster data loading
                        for x in ['val']}

        for i, (input, labels) in enumerate(dset_loaders['val']):

            # wrap them in Variable
            if self.use_gpu:
                input_var, target_var = torch.autograd.Variable(input.cuda(), volatile=True), \
                                        torch.autograd.Variable(labels.cuda(), volatile=True)
            else:
                input_var, target_var = torch.autograd.Variable(input), Variable(labels)

            # compute output
            output = self.model(input_var)
            # loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = self.accuracy(output.data, target_var.data, topk=(1, 5))

            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))


        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        #return top5.avg

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



    def calculateTop5Acc(self, datapath, batch_size = 10):


       self.validate(datapath, batch_size)