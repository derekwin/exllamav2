from __future__ import annotations
import torch
import itertools
from exllamav2.device import get_device_stream

_GLOBAL_UT = None

def init_uhm_runtime(
    gpu_id: int = 0,
    rank: int | None = None,
    world_size: int = 1,
    cache_capacity: int = 8,
    prefetch_count: int = 1,
    cache_type: str = "LRU",
    mem_type: str = "NVIDIA_GPU"
):
    """
    初始化全局 UHM runtime。
    若已存在 _GLOBAL_UT，则直接跳过。

    参数:
        gpu_id        - 当前 GPU ID
        rank          - 当前 rank（默认等于 gpu_id）
        world_size    - 总进程数
        cache_capacity- 缓存容量
        prefetch_count- 预取数量
        cache_type    - 缓存策略 ("LRU" 或 "MARKOV")
        mem_type      - 内存类型 ("NVIDIA_GPU" 或 "CPU")
    """
    global _GLOBAL_UT

    if _GLOBAL_UT is not None:
        return _GLOBAL_UT

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

    _GLOBAL_UT = ut.create_runtime(cfg)
    print(f"[UHM] Global runtime initialized (GPU {cfg.gpu_id}, rank={cfg.rank}/{cfg.world_size})")
    return _GLOBAL_UT


def _normalize_device(dev) -> str:
    # 统一成字符串：'cuda' / 'cpu' / 'runtime'
    if isinstance(dev, torch.device):
        return dev.type
    elif isinstance(dev, str):
        return dev
    else:
        raise ValueError(f"Unsupported device spec: {dev}")


def uhm_allocate_tensor_like(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """使用 uhm_tensor 分配一个与 tensor 形状相同的新张量，并复制数据
       device: str : "cuda"|"cpu"|"runtime"
    """
    try:
        import uhm_tensor as ut
    except ImportError:
        raise ImportError("uhm_tensor module not found. Please ensure it is installed.")
    
    shape = list(tensor.shape)

    dtype_map = {
        torch.float16: ut.ScalarType.F16,
        torch.bfloat16: ut.ScalarType.BF16,
        torch.float32: ut.ScalarType.F32,
        torch.int8: ut.ScalarType.INT8,
    }

    dtype_enum = dtype_map.get(tensor.dtype)
    if dtype_enum is None:
        raise ValueError(f"Unsupported dtype {tensor.dtype}")

    if device == "cuda":
        new_tensor = ut.allocate_gpu_tensor(shape, dtype_enum)
    elif device == "cpu":
        new_tensor = ut.allocate_cpu_tensor(shape, dtype_enum)
    elif device == "runtime":
        if _GLOBAL_UT:
            new_tensor = ut.allocate_runtime_tensor(_GLOBAL_UT, shape, dtype_enum)
        else:
            raise ValueError(f"Unsupported device type: {device}, invalid runtime")
    else:
        raise ValueError(f"Unsupported device type: {device}")

    new_tensor.copy_(tensor, non_blocking=True)
    return new_tensor


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
    """
    优先使用 UHM runtime 的分配 + copy_ 实现张量迁移；
    若 _GLOBAL_UT 不存在，则退回 safe_move_tensor。
    """
    dev = torch.device(device)

    # 如果目标设备一致，直接返回
    if tensor.device == dev:
        return tensor

    # # 如果存在 UHM runtime，优先使用
    if _GLOBAL_UT is not None:
        if dev.type == "cpu":
            return uhm_allocate_tensor_like(tensor, "cpu")
        elif dev.type == "cuda":
            # 单机场景可直接使用 runtime 分配
            return uhm_allocate_tensor_like(tensor, "runtime")
        else:
            return safe_move_tensor(tensor, dev)

    # 否则退回原有逻辑
    return safe_move_tensor(tensor, dev)
