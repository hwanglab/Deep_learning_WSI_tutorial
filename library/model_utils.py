import os
import torch
from torch import nn
from torchvision import models


def get_model_class_from_path(model_path):
    model_path = os.path.basename(model_path).lower()
    if 'densenet201' in model_path:
        return models.densenet201
    elif 'resnet18' in model_path:
        return models.resnet18
    elif 'shufflenet_v2_x1_0' in model_path:
        return models.shufflenet_v2_x1_0
    else:
        raise ValueError('Cannot automatically determine model type given model_path: ' + model_path)


def load_model_arch(model_init_func, pretrained, num_classes):
    if type(model_init_func) == str:
        raise TypeError('model_init_func should not be a string: "{}"'.format(model_init_func))
    if model_init_func.__name__ == 'googlenet':
        # Auxiliary GoogleNet outputs are not currently supported
        model = model_init_func(pretrained=pretrained, aux_logits=False)
    else:
        model = model_init_func(pretrained=pretrained)
    # Adjust the output layer to have the correct number of output nodes
    if isinstance(model, models.DenseNet):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model, models.GoogLeNet):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model, models.MobileNetV2):
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model, models.ResNet):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model, models.ShuffleNetV2):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model, models.SqueezeNet):
        num_ftrs = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(num_ftrs, num_classes, 1)
    else:
        raise RuntimeError('Unsupported model class: {}'.format(model.__class__.__name__))

    return model


def load_saved_model(model_path, num_classes,
                     model_class='auto', include_classification_layer=True):
    if model_class == 'auto':
        model_class = get_model_class_from_path(model_path)
    model = load_model_arch(model_class, False,
                            num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    if include_classification_layer == False:
        if isinstance(model, models.DenseNet):
            model.classifier = nn.Identity()
        elif isinstance(model, models.GoogLeNet):
            model.fc = nn.Identity()
        elif isinstance(model, models.MobileNetV2):
            model.classifier[1] = nn.Identity()
        elif isinstance(model, models.ResNet):
            model.fc = nn.Identity()
        elif isinstance(model, models.ShuffleNetV2):
            model.fc = nn.Identity()
        elif isinstance(model, models.SqueezeNet):
            model.classifier[1] = nn.Identity()
        else:
            raise RuntimeError('Unsupported model class: {}'.format(model.__class__.__name__))
    
    model.eval()
    return model


def load_saved_model_for_inference(*args, **kwargs):
    model = load_saved_model(*args, **kwargs)
    # Simply appends softmax layer
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model.eval()

    return model


def load_saved_model_for_feature_extraction(*args, **kwargs):
    assert 'include_classification_layer' not in kwargs.keys()
    model = load_saved_model(*args, include_classification_layer=False, **kwargs)
    model.eval()

    return model