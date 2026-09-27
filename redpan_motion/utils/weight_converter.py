"""
TensorFlow to PyTorch weight converter for RED-PAN.

Converts pre-trained TensorFlow weights to PyTorch format.
"""

import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)


# Mapping from TensorFlow layer names to PyTorch layer names
# Format: (tf_pattern, pytorch_pattern, transpose_required)
TF_TO_PYTORCH_MAPPING = [
    # Initial RRConv
    (r'RR_conv_init/conv1d', 'init_rrconv.init_conv.conv', True),
    (r'RR_conv_init/batch_normalization', 'init_rrconv.init_conv.bn', False),
    (r'RR_conv_init/conv1d_1', 'init_rrconv.conv_1x1', True),
    (r'RR_conv_init/conv1d_2', 'init_rrconv.recurrent_conv.conv', True),
    (r'RR_conv_init/batch_normalization_1', 'init_rrconv.recurrent_conv.bn', False),
    
    # Encoder blocks (pattern: RR_conv_enc_{i})
    # Expansion
    (r'RR_conv_enc_(\d+)_exp/conv1d', r'enc_exp_convs.\1.init_conv.conv', True),
    (r'RR_conv_enc_(\d+)_exp/batch_normalization', r'enc_exp_convs.\1.init_conv.bn', False),
    (r'RR_conv_enc_(\d+)_exp/conv1d_1', r'enc_exp_convs.\1.conv_1x1', True),
    (r'RR_conv_enc_(\d+)_exp/conv1d_2', r'enc_exp_convs.\1.recurrent_conv.conv', True),
    (r'RR_conv_enc_(\d+)_exp/batch_normalization_1', r'enc_exp_convs.\1.recurrent_conv.bn', False),
    
    # Downsampling
    (r'RR_conv_enc_(\d+)/conv1d', r'enc_down_convs.\1.init_conv.conv', True),
    (r'RR_conv_enc_(\d+)/batch_normalization', r'enc_down_convs.\1.init_conv.bn', False),
    (r'RR_conv_enc_(\d+)/conv1d_1', r'enc_down_convs.\1.conv_1x1', True),
    (r'RR_conv_enc_(\d+)/conv1d_2', r'enc_down_convs.\1.recurrent_conv.conv', True),
    (r'RR_conv_enc_(\d+)/batch_normalization_1', r'enc_down_convs.\1.recurrent_conv.bn', False),
    
    # Bottleneck
    (r'RR_conv_bottleneck/conv1d', 'bottleneck.init_conv.conv', True),
    (r'RR_conv_bottleneck/batch_normalization', 'bottleneck.init_conv.bn', False),
    (r'RR_conv_bottleneck/conv1d_1', 'bottleneck.conv_1x1', True),
    (r'RR_conv_bottleneck/conv1d_2', 'bottleneck.recurrent_conv.conv', True),
    (r'RR_conv_bottleneck/batch_normalization_1', 'bottleneck.recurrent_conv.bn', False),
    
    # Output heads
    (r'out_PS', 'picker_head', True),
    (r'out_M', 'detector_head', True),
]


def load_tensorflow_weights(model_path: str) -> Dict[str, np.ndarray]:
    """
    Load weights from a TensorFlow SavedModel or .h5 file.
    
    Args:
        model_path: Path to TensorFlow model
    
    Returns:
        Dictionary of {layer_name: weight_array}
    """
    import tensorflow as tf
    
    weights_dict = {}
    
    if model_path.endswith('.h5'):
        # Keras H5 format
        model = tf.keras.models.load_model(model_path, compile=False)
        for layer in model.layers:
            for weight in layer.weights:
                name = weight.name.replace(':0', '')
                weights_dict[name] = weight.numpy()
    else:
        # SavedModel format
        model = tf.saved_model.load(model_path)
        # Try to get weights from variables
        for var in model.variables:
            name = var.name.replace(':0', '')
            weights_dict[name] = var.numpy()
    
    logger.info(f"Loaded {len(weights_dict)} weight tensors from TensorFlow model")
    return weights_dict


def convert_tf_to_pytorch_weights(
    tf_weights: Dict[str, np.ndarray],
) -> Dict[str, torch.Tensor]:
    """
    Convert TensorFlow weight dictionary to PyTorch format.
    
    Handles:
    - Conv1D: TF (kernel_size, in_ch, out_ch) -> PyTorch (out_ch, in_ch, kernel_size)
    - BatchNorm: TF (gamma, beta, mean, var) -> PyTorch (weight, bias, running_mean, running_var)
    
    Args:
        tf_weights: Dictionary from load_tensorflow_weights
    
    Returns:
        Dictionary of PyTorch state_dict format
    """
    import re
    
    pytorch_weights = OrderedDict()
    
    for tf_name, tf_weight in tf_weights.items():
        pytorch_name = None
        needs_transpose = False
        
        # Try to match patterns
        for tf_pattern, pytorch_pattern, transpose in TF_TO_PYTORCH_MAPPING:
            match = re.match(tf_pattern, tf_name)
            if match:
                # Handle indexed patterns
                if r'\1' in pytorch_pattern:
                    pytorch_name = re.sub(tf_pattern, pytorch_pattern, tf_name)
                else:
                    pytorch_name = pytorch_pattern
                needs_transpose = transpose
                break
        
        if pytorch_name is None:
            # Try generic matching
            pytorch_name = _generic_tf_to_pytorch_name(tf_name)
            needs_transpose = 'conv' in tf_name.lower() and 'kernel' in tf_name.lower()
        
        # Convert weight
        if 'kernel' in tf_name.lower():
            # Conv kernel: TF (K, In, Out) -> PyTorch (Out, In, K)
            if len(tf_weight.shape) == 3:
                pytorch_weight = np.transpose(tf_weight, (2, 1, 0))
            else:
                pytorch_weight = tf_weight.T if needs_transpose else tf_weight
            param_name = pytorch_name + '.weight'
            
        elif 'bias' in tf_name.lower():
            pytorch_weight = tf_weight
            param_name = pytorch_name + '.bias'
            
        elif 'gamma' in tf_name.lower():
            # BatchNorm gamma -> weight
            pytorch_weight = tf_weight
            param_name = pytorch_name.replace('gamma', '') + '.weight'
            
        elif 'beta' in tf_name.lower():
            # BatchNorm beta -> bias
            pytorch_weight = tf_weight
            param_name = pytorch_name.replace('beta', '') + '.bias'
            
        elif 'moving_mean' in tf_name.lower():
            pytorch_weight = tf_weight
            param_name = pytorch_name.replace('moving_mean', '') + '.running_mean'
            
        elif 'moving_variance' in tf_name.lower():
            pytorch_weight = tf_weight
            param_name = pytorch_name.replace('moving_variance', '') + '.running_var'
            
        else:
            pytorch_weight = tf_weight
            param_name = pytorch_name
        
        pytorch_weights[param_name] = torch.from_numpy(pytorch_weight.copy())
    
    logger.info(f"Converted {len(pytorch_weights)} weight tensors to PyTorch format")
    return pytorch_weights


def _generic_tf_to_pytorch_name(tf_name: str) -> str:
    """Generic name conversion for unmapped layers."""
    # Replace TF naming conventions with PyTorch conventions
    name = tf_name
    name = name.replace('/', '.')
    name = name.replace('kernel', 'weight')
    name = name.replace('gamma', 'weight')
    name = name.replace('beta', 'bias')
    name = name.replace('moving_mean', 'running_mean')
    name = name.replace('moving_variance', 'running_var')
    return name


def convert_and_load_weights(
    pytorch_model: torch.nn.Module,
    tf_model_path: str,
    strict: bool = False,
) -> Tuple[torch.nn.Module, Dict]:
    """
    Convert TensorFlow weights and load into PyTorch model.
    
    Args:
        pytorch_model: PyTorch model instance
        tf_model_path: Path to TensorFlow model
        strict: If True, raise error on missing/unexpected keys
    
    Returns:
        Tuple of (model with loaded weights, load_info dict)
    """
    # Load TF weights
    tf_weights = load_tensorflow_weights(tf_model_path)
    
    # Convert to PyTorch format
    pytorch_weights = convert_tf_to_pytorch_weights(tf_weights)
    
    # Load into model
    missing, unexpected = pytorch_model.load_state_dict(pytorch_weights, strict=strict)
    
    load_info = {
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'loaded_keys': len(pytorch_weights) - len(unexpected),
    }
    
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
    
    logger.info(f"Loaded {load_info['loaded_keys']} weights into PyTorch model")
    
    return pytorch_model, load_info


def verify_weight_equivalence(
    tf_model_path: str,
    pytorch_model: torch.nn.Module,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Dict:
    """
    Verify that converted PyTorch model produces same outputs as TensorFlow model.
    
    Args:
        tf_model_path: Path to TensorFlow model
        pytorch_model: PyTorch model with loaded weights
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    
    Returns:
        Dictionary with verification results
    """
    import tensorflow as tf
    
    # Load TF model
    tf_model = tf.keras.models.load_model(tf_model_path, compile=False)
    
    # Create test input
    np.random.seed(42)
    test_input = np.random.randn(1, 6000, 3).astype(np.float32)  # TF format
    
    # TensorFlow prediction
    tf_output = tf_model.predict(test_input, verbose=0)
    if isinstance(tf_output, list):
        tf_picker, tf_detector = tf_output
    else:
        tf_picker = tf_output
        tf_detector = None
    
    # PyTorch prediction
    pytorch_model.eval()
    pytorch_input = torch.from_numpy(test_input.transpose(0, 2, 1))  # (B, C, T)
    with torch.no_grad():
        pytorch_picker, pytorch_detector = pytorch_model(pytorch_input)
    
    # Convert to numpy for comparison
    pytorch_picker_np = pytorch_picker.numpy().transpose(0, 2, 1)  # (B, T, C)
    pytorch_detector_np = pytorch_detector.numpy().transpose(0, 2, 1)
    
    # Compare
    picker_match = np.allclose(tf_picker, pytorch_picker_np, rtol=rtol, atol=atol)
    picker_max_diff = np.max(np.abs(tf_picker - pytorch_picker_np))
    
    results = {
        'picker_match': picker_match,
        'picker_max_diff': float(picker_max_diff),
    }
    
    if tf_detector is not None:
        detector_match = np.allclose(tf_detector, pytorch_detector_np, rtol=rtol, atol=atol)
        detector_max_diff = np.max(np.abs(tf_detector - pytorch_detector_np))
        results['detector_match'] = detector_match
        results['detector_max_diff'] = float(detector_max_diff)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert TF weights to PyTorch')
    parser.add_argument('--tf-model', '-t', required=True, help='Path to TensorFlow model')
    parser.add_argument('--output', '-o', required=True, help='Output PyTorch checkpoint path')
    parser.add_argument('--verify', action='store_true', help='Verify equivalence')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    from redpan_motion.models.mtan_r2unet import MTAN_R2UNet
    
    # Create PyTorch model
    model = MTAN_R2UNet()
    
    # Convert and load weights
    model, info = convert_and_load_weights(model, args.tf_model)
    
    print(f"\nConversion results:")
    print(f"  Loaded: {info['loaded_keys']} parameters")
    print(f"  Missing: {len(info['missing_keys'])} parameters")
    print(f"  Unexpected: {len(info['unexpected_keys'])} parameters")
    
    # Save PyTorch checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'conversion_info': info,
    }, args.output)
    print(f"\nSaved PyTorch checkpoint to: {args.output}")
    
    # Verify if requested
    if args.verify:
        print("\nVerifying equivalence...")
        results = verify_weight_equivalence(args.tf_model, model)
        for k, v in results.items():
            print(f"  {k}: {v}")
