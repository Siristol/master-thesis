from .ResNet import *
from .VGG import *
from .MobileNetV1 import *

def modelpool(MODELNAME, DATANAME):
    if 'imagenet' in DATANAME.lower():
        num_classes = 1000
    elif '100' in DATANAME.lower():
        num_classes = 100
        num_filters = 32
        strideFistConv = 1
    elif '10' in DATANAME.lower():
        num_classes = 10
        num_filters = 32
        strideFistConv = 1
    elif 'coco' in DATANAME.lower():
        num_classes = 2
        num_filters = 8
        strideFistConv = 2
    if MODELNAME.lower() == 'vgg16':
        return vgg16(num_classes=num_classes)
    if MODELNAME.lower() == 'vgg19':
        return vgg19(num_classes=num_classes)
    if MODELNAME.lower() == 'vgg13':
        return vgg13(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet20':
        return resnet20(num_classes=num_classes)
    elif MODELNAME.lower() == 'mobilenet':
        return MobileNet(num_classes=num_classes, num_filters=num_filters, strideFistConv=strideFistConv)
    else:
        print("Error:only support vgg16, vgg19, vgg13, resnet18, resnet20, resnet34, mobilenet, mobilenetv1")
        exit(0)
