import torch.nn.functional as F
import torchvision.models as models
from torch import nn

class audio_model(
    nn.Module
    ):
    
    def __init__(
        self, 
        num_classes
        ):
      
        super().__init__() 
        self.model = models.inception_v3(pretrained=False, aux_logits=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(
        self,
        x
        ):
      
        x = self.model(x)
        return F.log_softmax(x, dim=1)