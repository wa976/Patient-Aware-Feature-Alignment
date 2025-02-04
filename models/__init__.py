from .ast import ASTModel
from .beats import BEATsTransferLearningModel   
from .cnn import CNN6, CNN10, CNN14


_backbone_class_map = {
    'ast': ASTModel,
    'beats': BEATsTransferLearningModel,
    'cnn6': CNN6,
    'cnn10': CNN10,
    'cnn14': CNN14,
}


def get_backbone_class(key):
    if key in _backbone_class_map:
        return _backbone_class_map[key]
    else:
        raise ValueError('Invalid backbone: {}'.format(key))