# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Torch distributed utilities."""

import typing as tp

import torch


def _dist_module():
    return getattr(torch, "distributed", None)


def _dist_attr(name: str):
    dist = _dist_module()
    return getattr(dist, name, None) if dist is not None else None


def _dist_is_initialized() -> bool:
    is_initialized = _dist_attr("is_initialized")
    if not callable(is_initialized):
        return False
    try:
        return bool(is_initialized())
    except Exception:
        return False


def rank():
    get_rank = _dist_attr("get_rank")
    if _dist_is_initialized() and callable(get_rank):
        return get_rank()
    return 0


def world_size():
    get_world_size = _dist_attr("get_world_size")
    if _dist_is_initialized() and callable(get_world_size):
        return get_world_size()
    return 1


def is_distributed():
    return _dist_is_initialized() and world_size() > 1


def all_reduce(tensor: torch.Tensor, op=None, async_op: bool = False):
    reduce = _dist_attr("all_reduce")
    if not is_distributed() or not callable(reduce):
        return None
    if op is None:
        reduce_op = _dist_attr("ReduceOp")
        op = getattr(reduce_op, "SUM", None) if reduce_op is not None else None
    if op is None:
        return None
    return reduce(tensor, op=op, async_op=async_op)


def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def _check_number_of_params(params: tp.List[torch.Tensor]):
    # utility function to check that the number of params in all workers is the same,
    # and thus avoid a deadlock with distributed all reduce.
    if not is_distributed() or not params:
        return
    # print('params[0].device ', params[0].device)
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        # If not all the workers have the same number, for at least one of them,
        # this inequality will be verified.
        raise RuntimeError(
            f"Mismatch in number of params: ours is {len(params)}, at least one worker has a different one."
        )


def broadcast_tensors(tensors: tp.Iterable[torch.Tensor], src: int = 0):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    if not is_distributed():
        return
    broadcast = _dist_attr("broadcast")
    if not callable(broadcast):
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()


def sync_buffer(buffers, average=True):
    """
    Sync grad for buffers. If average is False, broadcast instead of averaging.
    """
    if not is_distributed():
        return
    broadcast = _dist_attr("broadcast")
    handles = []
    for buffer in buffers:
        if torch.is_floating_point(buffer.data):
            if average:
                reduce_op = _dist_attr("ReduceOp")
                sum_op = getattr(reduce_op, "SUM", None) if reduce_op is not None else None
                if sum_op is None:
                    continue
                handle = all_reduce(buffer.data, op=sum_op, async_op=True)
            else:
                if not callable(broadcast):
                    continue
                handle = broadcast(buffer.data, src=0, async_op=True)
            if handle is None:
                continue
            handles.append((buffer, handle))
    for buffer, handle in handles:
        handle.wait()
        if average:
            buffer.data /= world_size()


def sync_grad(params):
    """
    Simpler alternative to DistributedDataParallel, that doesn't rely
    on any black magic. For simple models it can also be as fast.
    Just call this on your model parameters after the call to backward!
    """
    if not is_distributed():
        return
    reduce_op = _dist_attr("ReduceOp")
    sum_op = getattr(reduce_op, "SUM", None) if reduce_op is not None else None
    if sum_op is None:
        return
    handles = []
    for p in params:
        if p.grad is not None:
            handle = all_reduce(p.grad.data, op=sum_op, async_op=True)
            if handle is None:
                continue
            handles.append((p, handle))
    for p, handle in handles:
        handle.wait()
        p.grad.data /= world_size()


def average_metrics(metrics: tp.Dict[str, float], count=1.0):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unormalized weight.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))
