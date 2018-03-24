from torchvision import models
import DemoModel
import Model1
import Model2
import Model3
import Model4
import Model5
import Model6

def resnet18(pretrained = True):
    return models.resnet18(pretrained)

def restnet50(pretrained = True):
    return models.resnet50(pretrained)

def vgg19(pretrained = True):
    return models.vgg19(pretrained)

def alexnet(pretrained = True):
    return models.alexnet(pretrained)

def demo_model():
    return DemoModel.Demo_Model();

def model1():
    return Model1.Model1();

def model2():
    return Model2.Model2();

def model3():
    return Model3.Model3();

def model4():
    return Model4.Model4();

def model5():
    return Model5.Model5();

def model6():
    return Model6.Model6();