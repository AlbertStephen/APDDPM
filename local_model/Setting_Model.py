from torchvision import models
from .preactresnet import *
from .wideresnet import *

save_name = ["ResNet18.pth", "ResNet50.pth", "PreactResNet18.pth", "WideResNet28_10.pth"]

def PreactResNet(num_classes= 10):
    model = PreActResNet18(num_classes=num_classes)
    return model

def WideResNet_28(num_classes= 10, depth=28, widen_factor=10, dropRate=0.0):
    model = WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen_factor, dropRate=dropRate)
    return model

def ResNet18(num_classes, pretrained= True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return model

def ResNet50(num_classes, pretrained= True):
    model = models.resnet50(pretrained=pretrained)
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    return model

def setup_model(index, num_class, pretrained= True):
    if index == 0:
        model = ResNet18(num_classes= num_class, pretrained= pretrained)
        image_size = 32
        # image_size = 224
    elif index == 1:
        model = ResNet50(num_classes=num_class, pretrained=pretrained)
        image_size = 32
    elif index == 2:
        model = PreActResNet18(num_classes=num_class)
    elif index == 3:
        model = WideResNet_28(num_classes=num_class)
    else:
        raise ValueError(f"{index} model isn't exist!")
    return model

def load_model(data_name, index, pretrained = True):
    if data_name.lower() == "cifar10":
        model= setup_model(index= index, num_class= 10, pretrained= pretrained)
    elif data_name.lower() == "cifar100":
        model= setup_model(index=index, num_class=100, pretrained=pretrained)
    else:
        raise ValueError(f"{data_name} dataset isn't exist!")
    return model



