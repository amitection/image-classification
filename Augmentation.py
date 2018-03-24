''' Module for data augmentation. Two strategies have been demonstrated below. 
You can check for more strategies at 
http://pytorch.org/docs/master/torchvision/transforms.html '''

from torchvision import transforms


class Augmentation:   
    def __init__(self,strategy):
        print ("Data Augmentation Initialized with strategy %s"%(strategy));
        self.strategy = strategy;
        
        
    def applyTransforms(self):
        if self.strategy == "H_FLIP": # horizontal flip with a probability of 0.5
            data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        elif self.strategy == "SCALE_H_FLIP": # resize to 224*224 and then do a random horizontal flip.
            data_transforms = {
            'train': transforms.Compose([
                transforms.Scale([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        elif self.strategy == "FIVE_CROP_FLIP":
            data_transforms = {
                'train': transforms.Compose([
                    transforms.FiveCrop([56, 56]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.FiveCrop([56, 56]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        }

        elif self.strategy == "RANDOM_CROP_FLIP":
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(300),
                    transforms.RandomCrop(258),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(300),
                    transforms.RandomCrop(258),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        }
        elif self.strategy == "SCALE":
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(258),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(300),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        }


        else :
            print ("Please specify correct augmentation strategy : %s not defined"%(self.strategy));
            exit();
            
        return data_transforms;

