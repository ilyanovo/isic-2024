import timm
import torch
from torch import nn


class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, drop_path_rate=0, drop_rate=0, pretrained=True, checkpoint_path=None):
        super(ISICModel, self).__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            heckpoint_path=checkpoint_path,
            drop_rate=drop_rate, 
            drop_path_rate=drop_path_rate)

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid() if num_classes == 1 else nn.Softmax()

    def forward(self, images):
        return self.sigmoid(self.model(images))


class ISICModelEdgnet(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None, *args, **kwargs):
        super(ISICModelEdgnet, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, global_pool='avg')
        self.sigmoid = nn.Sigmoid()
    def forward(self, images):
        return self.sigmoid(self.model(images))



def setup_model(model_name, checkpoint_path=None, num_classes=1, drop_path_rate=0, drop_rate=0, device: str = 'cuda', model_maker=ISICModel):
    model = model_maker(model_name, pretrained=True, num_classes=num_classes, drop_path_rate=drop_path_rate, drop_rate=drop_rate)

    return model.to(device)

