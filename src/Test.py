import os

#pytorch modules
import torch
from torchvision import datasets
from torch.autograd import Variable

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import Augmentation as ag
class Test():

    def __init__(self, aug, model, use_gpu = True):
       #Define augmentation strategy
        self.augmentation_strategy = ag.Augmentation(aug);
        self.data_transforms = self.augmentation_strategy.applyTransforms();
        self.model = model;
        self.model.train(False)
        self.use_gpu = use_gpu
        
    def testfromdir(self,datapath,batch_size = 128):
        #Root directory
        data_dir = datapath;
        ##
        
        ######### Data Loader ###########
        dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), self.data_transforms[x])
                     for x in ['val']}
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                       shuffle=True, num_workers=16)# set num_workers higher for more cores and faster data loading
                     for x in ['val']}

        scores = torch.cuda.FloatTensor();
        id = 0

        for count, data in enumerate(dset_loaders['val']):

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if self.use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                    Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)


            # forward
            outputs = self.model(inputs)
            scores = torch.cat((scores,torch.nn.functional.softmax(outputs).data),0);
            id += 1
                
        return(scores.cpu().numpy());


    def test_ytest_ypred(self, datapath, batch_size):
        # Root directory
        data_dir = datapath;
        ##

        ######### Data Loader ###########
        dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), self.data_transforms[x])
                 for x in ['val']}
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                       shuffle=True, num_workers=16)
                        # set num_workers higher for more cores and faster data loading
                        for x in ['val']}

        dset_sizes = {x: len(dsets[x]) for x in ['val']}
        dset_classes = dsets['val'].classes

        y_pred = []
        y_true = []

        for count, data in enumerate(dset_loaders['val']):

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if self.use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                                 Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)

            for i in range(inputs.size()[0]):
                y_pred.append(preds[i])
                y_true.append(labels.data[i])


        return (dset_classes, y_true, y_pred)


    def plot_confusion_matrix(self, cm, classes, normalize=False,title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    def calculate_confusion_matrix(self, datapath, batch_size):

        dset_classes, y_true, y_pred = self.test_ytest_ypred(datapath, batch_size);


        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_true, y_pred)
        cnf_matrix = cnf_matrix[:20, :20]
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        # plt.figure()
        # self.plot_confusion_matrix(cnf_matrix, classes=y_test,
        #                       title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=dset_classes[:20], normalize=True,
                              title='Normalized confusion matrix')

        plt.show()

