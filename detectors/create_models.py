from detectors.models.efficientdet import EfficientDet
from detectors.ssd.config import cfg
from detectors.ssd.modeling.detector import build_detection_model

efficientdet_params = {'input_size': 512,
                        'backbone': 'B0',
                        'W_bifpn': 64,
                        'D_bifpn': 2,
                        'D_class': 3}


def create_efficientdet_model():
    model = EfficientDet(num_classes=1,
        network='efficientdet-d0',
        W_bifpn=efficientdet_params['W_bifpn'],
        D_bifpn=efficientdet_params['D_bifpn'],
        D_class=efficientdet_params['D_class']
    )
    for name, param in model.named_parameters():
        if param.requires_grad and 'backbone' in name:
            param.requires_grad = False
    return model

def create_ssd_model():
    model = build_detection_model(cfg)
    for name, param in model.named_parameters():
        if param.requires_grad and 'backbone' in name:
            param.requires_grad = False
    return model
