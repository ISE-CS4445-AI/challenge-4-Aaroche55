# DO NOT CHANGE THIS FILE! If any code is changed, the instructor will be notified on Github classroom's assignment dashboard.

# Common imports

import pytest
import torch
import challenge_4_export

def test_create_random_tensor():
    """
    Test that create_random_tensor returns a torch.Tensor with the expected shape and value range.
    """
    shape = (2, 3, 4)
    tensor = challenge_4_export.create_random_tensor(shape)
    assert isinstance(tensor, torch.Tensor), "Output is not a torch.Tensor."
    assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}."
    # Verify that the values are in the range [0, 1]
    assert torch.all(tensor >= 0) and torch.all(tensor <= 1), "Tensor values should be in range [0, 1]."


def test_move_tensor_to_device():
    """
    Test that move_tensor_to_device correctly moves a tensor to the requested device.
    """
    sample_tensor = torch.tensor([1, 2, 3])
    # Use an available device: 'cuda' if available, else 'cpu'
    available_device = "cuda" if torch.cuda.is_available() else "cpu"
    moved_tensor = challenge_4_export.move_tensor_to_device(sample_tensor)
    assert moved_tensor.device.type == available_device, (
        f"Expected tensor on device '{available_device}', but got '{moved_tensor.device.type}'."
    )


def test_build_LeNet5():
    """
    Test that build_LeNet5 returns a LeNet-5 model.
    The test checks for:
      - The model is an instance of nn.Module.
      - Presence of at least one average pooling layer (nn.AvgPool2d).
      - Presence of at least one Sigmoid activation (nn.Sigmoid).
      - The first convolutional layer has 1 input channel, 6 output channels, and kernel size 5x5.
    """
    model = challenge_4_export.build_LeNet5()
    # Check that the model is an instance of torch.nn.Module.
    assert isinstance(model, torch.nn.Module), "Returned model is not an instance of torch.nn.Module."
    
    # Check for at least one average pooling layer.
    has_avgpool = any(isinstance(m, torch.nn.AvgPool2d) for m in model.modules())
    assert has_avgpool, "Model does not contain an Average Pooling layer (nn.AvgPool2d)."
    
    # Check for at least one Sigmoid activation layer.
    has_sigmoid = any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())
    assert has_sigmoid, "Model does not contain a Sigmoid activation layer (nn.Sigmoid)."
    
    # Check for convolutional layers and inspect the first one.
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(conv_layers) >= 2, "Model should have at least two convolutional layers."
    
    first_conv = conv_layers[0]
    assert first_conv.in_channels == 1, f"First Conv2d layer expected 1 input channel, got {first_conv.in_channels}."
    assert first_conv.out_channels == 6, f"First Conv2d layer expected 6 output channels, got {first_conv.out_channels}."
    assert first_conv.kernel_size == (5, 5), f"First Conv2d layer kernel_size expected (5, 5), got {first_conv.kernel_size}."