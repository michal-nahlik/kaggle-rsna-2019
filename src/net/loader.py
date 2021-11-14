import torch
import pretrainedmodels
import torchvision.models as torchvision_models
from efficientnet_pytorch import EfficientNet


def get_model(n_classes, origin, model_name):
    """
    Load / create model from different sources and unfreeze layers
    :param n_classes: number of classes
    :param origin: one of ['efficientnet', 'torchhub', 'torchvision', 'pretrainedmodels', 'saved']
    :param model_name: name of the model or file in case of loading from saved models
    :return: model with unfrozen layers
    """

    # from lib
    print('Loading model {} from {}'.format(origin, model_name))

    if origin == 'efficientnet':
        model = EfficientNet.from_pretrained(model_name)
        num_in_features = model._fc.in_features
        model._fc = torch.nn.Linear(num_in_features, n_classes)

    # from torch.hub
    elif origin == 'torchhub':
        model = torch.hub.load('facebookresearch/WSL-Images', model_name)
        num_in_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_in_features, n_classes)

    # from torchvision
    elif origin == 'torchvision':
        model = torchvision_models.resnext50_32x4d(pretrained=True)
        num_in_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_in_features, n_classes)

    # or from pretrainedmodels
    elif origin == 'pretrainedmodels':
        model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        dim_feats = model.last_linear.in_features
        # model.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        model.last_linear = torch.nn.Linear(dim_feats, n_classes)
        print(model.input_size, model.input_space, model.input_range)

    # or from my saved model
    elif origin == 'saved':
        model_path = '../output/models/{}'.format(model_name)
        model = torch.load(model_path)

    for param in model.parameters():
        param.requires_grad = True    
        
    return model
