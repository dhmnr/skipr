import torch
import hashlib

def fingerprint_tensor(tensor: torch.Tensor) -> str:
    """
    Generate a fingerprint (hash) for a given PyTorch tensor.
    Args:
        tensor (torch.Tensor): The input tensor to be hashed.
    Returns:
        str: A hexadecimal string representing the hash of the tensor.
    """
    tensor_cpu = tensor.cpu().detach().numpy()
    tensor_bytes = tensor_cpu.tobytes()
    hasher = hashlib.sha256()
    hasher.update(tensor_bytes)
    return hasher.hexdigest()