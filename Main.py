#Demo for CS7-GV1

#general modules
from __future__ import print_function, division
import os
import argparse
import time
import copy
import numpy as np

#pytorch modules
import torch
import torch.nn as nn
from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.autograd import Variable

# TensorBoardX
from tensorboardX import SummaryWriter

#user defined modules
import Augmentation as ag
import Models
from ExtReduceLROnPlateau import ExtReduceLROnPlateau
from Test import Test
from CalculateTop5Acc import CalculateTop5Acc
from RenderConfusionMatrix import RenderConfusionMatrix

parser = argparse.ArgumentParser(description='CS7-GV1 Final Project');



#add/remove arguments as required. It is useful when tuning hyperparameters from bash scripts
parser.add_argument('--aug', type=str, default = '', help='data augmentation strategy')
parser.add_argument('--datapath', type=str, default='', 
               help='root folder for data.It contains two sub-directories train and val')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')               
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')
parser.add_argument('--batch_size', type=int, default = 128,
                    help='batch size')
parser.add_argument('--model', type=str, default = None, help='Specify model to use for training.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limlsit')
parser.add_argument('--tag', type=str, default=None,
                    help='unique_identifier used to save results')
args = parser.parse_args();

if not args.tag:
    print('Please specify tag...')
    exit()

print (args)

#Define augmentation strategy
augmentation_strategy = ag.Augmentation(args.aug);
data_transforms = augmentation_strategy.applyTransforms();
##

#Root directory
data_dir = args.datapath;
##


# TensorBoardX
summary_writer = SummaryWriter('../logging/'+args.tag)

######### Data Loader ###########
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
             for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
                                               shuffle=True, num_workers=16) # set num_workers higher for more cores and faster data loading
             for x in ['train', 'val']}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes
#################################

#set GPU flag
use_gpu = args.cuda;
##

#Load model . Once you define your own model in Models.py, you can call it from here. 
if args.model == "ResNet18":
    current_model = Models.resnet18(args.pretrained)
    num_ftrs = current_model.fc.in_features
    current_model.fc = nn.Linear(num_ftrs, len(dset_classes));

elif args.model == "ResNet50":
    current_model = Models.restnet50(args.pretrained)

elif args.model == "VGG19":
    current_model = Models.vgg19(args.pretrained)

elif args.model == "Alexnet":
    current_model = Models.alexnet(args.pretrained)
    
elif args.model == 'Demo':
    current_model = Models.demo_model();

elif args.model == 'Model1':
    current_model = Models.model1();

elif args.model == 'Model2':
    current_model = Models.model2();

elif args.model == 'Model3':
    current_model = Models.model3();

elif args.model == 'Model4':
    current_model = Models.model4();

elif args.model == 'Model5':
    current_model = Models.model5();

elif args.model == 'Model6':
    current_model = Models.model6();

else :
    print ("Model %s not found"%(args.model))
    exit();    


if use_gpu:
    current_model = current_model.cuda();
    
# uses a cross entropy loss as the loss function
# http://pytorcorks for any number of images in the folder. Check the kaggle competition page for downloading the test datah.org/docs/master/nn.html#
criterion = nn.CrossEntropyLoss()

#uses stochastic gradient descent for learning
# http://pytorch.org/docs/master/optim.html
optimizer_ft = optim.SGD(current_model.parameters(), lr=args.lr, momentum=0.9)

#the learning rate condition. The ReduceLROnPlateau class reduces the learning rate by 'factor' after 'patience' epochs.
scheduler_ft = ReduceLROnPlateau(optimizer_ft, 'min', factor = 0.5,patience = 3, verbose = True)



def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for count, data in enumerate(dset_loaders[phase]):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                if count%10 == 0:
                    print('Batch %d || Running Loss = %0.6f || Running Accuracy = %0.6f'%(count+1,running_loss/(args.batch_size*(count+1)),running_corrects/(args.batch_size*(count+1))))
                #print('Running Loss = %0.6f'%(running_loss/(args.batch_size*(count+1))))

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            

            print('Epoch %d || %s Loss: %.4f || Acc: %.4f'%(epoch,
                phase, epoch_loss, epoch_acc),end = ' || ')
            #pdb.set_trace();
            if phase == 'val':
                print ('\n', end='');
                lr_scheduler.step(epoch_loss);
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)


            if phase == 'val':
                # Plotting to TensorBoardX
                summary_writer.add_scalar('val-loss',epoch_loss, epoch)
                summary_writer.add_scalar('val-acc', epoch_acc, epoch)
            elif phase == 'train':
                summary_writer.add_scalar('train-loss', epoch_loss, epoch)
                summary_writer.add_scalar('train-acc', epoch_acc, epoch)


            summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)


    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    summary_writer.close();
    return best_model



# #comment the block below if you are not training i.e while testing
# ######################
# trained_model = train_model(current_model, criterion, optimizer_ft, scheduler_ft,
#                       num_epochs=args.epochs);
# with open('../serialized-models/'+args.tag+'.model', 'wb') as f:
#     torch.save(trained_model, f);
# ######################

## uncomment the lines blow while testing.
trained_model = torch.load('../serialized-models/'+args.tag+'.model');
testDataPath = '../../data/'
# t = Test(args.aug,trained_model);
# scores = t.testfromdir(testDataPath, args.batch_size);
#
# np.savetxt('../testing-results/'+args.tag+'.txt', scores, fmt='%0.5f',delimiter=',')


# t.calculate_confusion_matrix(testDataPath, batch_size=10);

rcm = RenderConfusionMatrix(args.aug,trained_model)
rcm.renderConfusionMatrix(testDataPath, batch_size=10)

top5Acc = CalculateTop5Acc(args.aug,trained_model)
top5Acc.calculateTop5Acc(testDataPath, batch_size=10)

