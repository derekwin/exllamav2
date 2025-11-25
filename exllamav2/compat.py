from __future__ import annotations
import torch
import itertools
from exllamav2.device import get_device_stream

_GLOBAL_UHM_INITIALIZED = False

def init_uhm_runtime(
    gpu_id: int = 0,
    rank: int | None = None,
    world_size: int = 1,
    cache_capacity: int = 8,
    prefetch_count: int = 1,
    cache_type: str = "LRU",
    mem_type: str = "NVIDIA_GPU"
):

    global _GLOBAL_UHM_INITIALIZED
    if _GLOBAL_UHM_INITIALIZED:
        return

    try:
        import uhm_tensor as ut
    except ImportError:
        raise ImportError("uhm_tensor module not found. Please ensure it is installed.")

    cfg = ut.RuntimeConfig()
    cfg.cache_capacity = cache_capacity
    cfg.prefetch_count = prefetch_count
    cfg.cache_type = getattr(ut.CacheType, cache_type)
    cfg.mem_type = getattr(ut.MemoryType, mem_type)
    cfg.gpu_id = gpu_id
    cfg.rank = gpu_id if rank is None else rank
    cfg.world_size = world_size

    ut.init(cfg, connect_to=0)

    print(f"[UHM] Global Runtime initialized (GPU {cfg.gpu_id}, rank={cfg.rank}/{cfg.world_size})")

    _GLOBAL_UHM_INITIALIZED = True


def uhm_allocate_tensor_like(tensor: torch.Tensor):
    try:
        import uhm_tensor as ut
    except ImportError:
        raise ImportError("uhm_tensor module not found. Please ensure it is installed.")

    shape = list(tensor.shape)

    if not _GLOBAL_UHM_INITIALIZED:
        raise RuntimeError("UHM runtime not initialized. Call init_uhm_runtime(...) first.")
    uhmt = ut.allocate(shape, tensor.dtype)
    uhmt.copy_(tensor, non_blocking=True)
    return uhmt


# Emulate pairwise on Python <3.10

try:
    pairwise = itertools.pairwise
except AttributeError:
    def pairwise(iterable):
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

# On some setups Torch will attempt to use GPU peer-to-peer copies even when they are not supported. This is either
# a driver issue, a bug in Torch, or both. Either way, the result is that .to() will create an empty tensor on the
# target device and silently fail to copy any data into it. This is a workaround.

tested_peer_copy = None

def test_gpu_peer_copy(
    device_a: torch.Device,
    device_b: torch.Device
):
    global tested_peer_copy

    if tested_peer_copy is None:
        num_dev = torch.cuda.device_count()
        tested_peer_copy = [[0 for _ in range(num_dev)] for _ in range(num_dev)]

    idx_a = device_a.index
    idx_b = device_b.index
    if idx_a > idx_b: idx_a, idx_b = idx_b, idx_a

    t = tested_peer_copy[idx_a][idx_b]
    if t == -1: return False
    if t == 1: return True

    dev_i = f"cuda:{idx_a}"
    dev_j = f"cuda:{idx_b}"
    a = torch.randn(5, device = dev_i) + 123.0
    b = a.to(dev_j)
    c = b.to(dev_i)
    if torch.all(a == c):
        tested_peer_copy[idx_a][idx_b] = 1
        return True
    else:
        tested_peer_copy[idx_a][idx_b] = -1
        return False


def safe_move_tensor(
    tensor: torch.Tensor | tuple[torch.Tensor],
    device: torch.Device | str | int,
    non_blocking = False
):

    # Accept tensor or tuple of tensors

    if isinstance(tensor, tuple):
        return tuple(safe_move_tensor(x, device) for x in tensor)

    # Accept torch.device, string or int

    device = torch.device(device)
    from_index = tensor.device.index
    to_index = device.index

    # No move

    if tensor.device == device:
        return tensor

    # Copies to/from system RAM are always fine

    if tensor.device.type == "cpu":
        stream = get_device_stream(to_index)
        if stream is not None:
            with torch.cuda.stream(stream):
                r = tensor.to(device, non_blocking = True)
                torch.cuda.synchronize(to_index)
            return r
        else:
            return tensor.to(device, non_blocking = non_blocking)

    if device.type == "cpu":
        stream = get_device_stream(from_index)
        if stream is not None:
            with torch.cuda.stream(stream):
                r = tensor.to(device, non_blocking = True)
                torch.cuda.synchronize(from_index)
            return r
        else:
            return tensor.to(device, non_blocking = non_blocking)

    # Source and dest are distinct CUDA devices
    # Test tensor.to (once) and if it seems to be working, let Torch decide

    if test_gpu_peer_copy(tensor.device, device):
        from_stream = get_device_stream(from_index)
        to_stream = get_device_stream(to_index)

        if from_stream is not None and to_stream is not None:
            with torch.cuda.stream(from_stream):
                with torch.cuda.stream(to_stream):
                    r = tensor.to(device, non_blocking = True)
        elif from_stream is not None:
            with torch.cuda.stream(from_stream):
                r = tensor.to(device, non_blocking = True)
        elif to_stream is not None:
            with torch.cuda.stream(to_stream):
                r = tensor.to(device, non_blocking = True)
        else:
            r = tensor.to(device, non_blocking = True)

        if not non_blocking:
            torch.cuda.synchronize(to_index)
        return r

    # Force move tensor via CPU

    from_stream = get_device_stream(from_index)
    to_stream = get_device_stream(to_index)

    if from_stream is not None:
        with torch.cuda.stream(from_stream):
            tensor_cpu = tensor.to("cpu", non_blocking = True)
            torch.cuda.synchronize(from_index)
    else:
        tensor_cpu = tensor.cpu()

    if to_stream is not None:
        with torch.cuda.stream(to_stream):
            r = tensor_cpu.to(device, non_blocking = True)
            torch.cuda.synchronize(to_index)
            return r
    else:
        return tensor_cpu.to(device)


def move_tensor_preferring_uhm(
    tensor: torch.Tensor,
    device: torch.device | str | int
) -> torch.Tensor:
    dev = torch.device(device)
    if tensor.device == dev:
        return tensor
    
    if _GLOBAL_UHM_INITIALIZED:
        return uhm_allocate_tensor_like(tensor) 
    return safe_move_tensor(tensor, dev)
