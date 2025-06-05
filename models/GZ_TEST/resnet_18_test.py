import torch.nn as nn
import torchvision
from torchvision import models
import clip
from PIL import Image
class Resnet18Backbone(nn.Module):
    def __init__(self):
        super(Resnet18Backbone, self).__init__()

        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Sequential()
        # model.
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self._maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        # x = self.model.avgpool(x)
        # resnet_18 = models.resnet18(pretrained=True)
        # modules = list(resnet_18.children())[:-2]
        # self.backbone = nn.Sequential(*modules)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self._maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # x = self.avgpool(x)
        # x = self.backbone(x)
        return x4
_, preprocess = clip.load("RN50", device='cpu')

img = preprocess(Image.open("/home/gaozhao/graspnet/scenes/scene_0000/kinect/rgb/0000.png").convert('RGB')).unsqueeze(0)
net = Resnet18Backbone()
x = net(img)