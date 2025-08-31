# ResNet Paper Code Implementation tests
import torch

from src.abilities.resnet_paper_code_implementation import (
    BasicBlock,
    BottleneckBlock,
    ResNet,
    create_paper_code_implementation,
)


def test_resnet_creation():
    model = create_paper_code_implementation('resnet18')
    assert model is not None
    assert isinstance(model, ResNet)

def test_basic_block():
    block = BasicBlock(64, 64)
    x = torch.randn(2, 64, 56, 56)
    output = block(x)
    assert output.shape == x.shape

def test_bottleneck_block():
    import torch.nn as nn
    # Create downsample for dimension mismatch (64 -> 256)
    downsample = nn.Sequential(
        nn.Conv2d(64, 256, kernel_size=1, bias=False),
        nn.BatchNorm2d(256)
    )
    block = BottleneckBlock(64, 64, downsample=downsample)
    x = torch.randn(2, 64, 56, 56)
    output = block(x)
    assert output.shape == (2, 256, 56, 56)  # expansion = 4

def test_resnet_forward():
    model = create_paper_code_implementation('resnet18', num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    assert output.shape == (1, 10)
