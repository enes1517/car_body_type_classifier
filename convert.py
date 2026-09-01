import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
import torch
import torch.nn as nn
import torchvision.models as models

# model_egitimi.py ile AYNI mimari kullanılmalı (Dropout=0.5)
m = models.mobilenet_v2(weights=None)
m.classifier = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(m.classifier[1].in_features, 8)
)
sd = torch.load('c:/Users/HP/Desktop/yazlab3/araba_modeli_pytorch.pth', map_location='cpu')
m.load_state_dict(sd)
m.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    m,
    dummy_input,
    'c:/Users/HP/Desktop/yazlab3/araba_modeli.onnx',
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Conversion successful.")
