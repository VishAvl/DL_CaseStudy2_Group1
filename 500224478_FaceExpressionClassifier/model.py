import torch as T
from torch import nn, optim
import torch.nn.functional as F
import torchvision.models as models

from typing import Union, List

def get_model(num_classes:int, unfreeze_layers:Union[None, List[int]] = None, drop_rate: Union[None, float] = None):

    model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False


    if unfreeze_layers is not None and len(unfreeze_layers) > 0:
        # Now unfreeze the layers in the unfreeze layer/ list
        for layer_num in unfreeze_layers:
            for name, child in model.features[layer_num].named_modules():
                if not isinstance(child, nn.BatchNorm2d) and \
                    not isinstance(child, nn.Sequential) and \
                    not hasattr(child, 'block'):

                    for param in child.parameters():

                        param.requires_grad = True

 

    if drop_rate is not None:
        model.classifier[0] = nn.Dropout(drop_rate)

 

    # Chagne the classifier head as per our need

    model.classifier[1] = nn.Linear(2560, num_classes)

    return model

if __name__ == "__main__":
    # model = get_model(6, [-1], 0.1)
    # script = T.jit.script(model)
    # script.save("model_script.pt")

    model = T.jit.load('model_script.pt')
    print("Model Loaded Succesfully")