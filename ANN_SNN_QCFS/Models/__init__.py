from .ResNet import *
from .VGG import *
from .layer import *
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
    elif MODELNAME.lower() == 'vgg19':
        return vgg19(num_classes=num_classes)
    elif MODELNAME.lower() == 'vgg16_wobn':
        return vgg16_wobn(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet20':
        return resnet20(num_classes=num_classes)
    elif MODELNAME.lower() == 'mobilenet':
        return MobileNet(num_classes=num_classes, num_filters=num_filters, strideFistConv=strideFistConv)
    else:
        print("still not support this model")
        exit(0)