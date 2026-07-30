"""CUDA stream-based DataLoader prefetcher (Step 5D, Session Speed).

Overlaps data transfer (CPU pinned → GPU) with the previous batch's
forward/backward pass using a secondary CUDA stream.  On CPU-only systems
it degrades gracefully to a standard iterator.

Usage::

    prefetcher = CUDAPrefetcher(loader, device)
    for batch in prefetcher:
        x, y = batch
        # train...

The prefetcher wraps any iterable DataLoader and provides an identical
iteration interface. Output tensors are already on the target device.
"""
from __future__ import annotations

from typing import Any, Iterator


class CUDAPrefetcher:
    """Prefetch DataLoader batches onto the GPU using a secondary CUDA stream.

    Attributes
    ----------
    loader : iterable DataLoader
        Source DataLoader. Must yield tuples/lists of tensors (or plain tensors).
    device : torch.device
        Target device. If not CUDA, the prefetcher is a transparent passthrough.
    """

    def __init__(self, loader, device) -> None:
        import torch
        self.loader = loader
        self.device = device
        self._is_cuda = device.type == "cuda" if hasattr(device, "type") else str(device).startswith("cuda")
        self._stream = torch.cuda.Stream() if self._is_cuda else None

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Any]:
        import torch

        if not self._is_cuda:
            yield from self.loader
            return

        loader_iter = iter(self.loader)

        def _move_to_device(batch):
            if isinstance(batch, torch.Tensor):
                return batch.to(self.device, non_blocking=True)
            if isinstance(batch, (list, tuple)):
                moved = [_move_to_device(b) for b in batch]
                return type(batch)(moved)
            return batch  # non-tensor (e.g. string labels)

        # Pre-load first batch on the prefetch stream.
        try:
            next_batch = next(loader_iter)
        except StopIteration:
            return

        with torch.cuda.stream(self._stream):
            next_batch_gpu = _move_to_device(next_batch)

        while True:
            # Synchronise: ensure prefetched batch is ready on the main stream.
            torch.cuda.current_stream().wait_stream(self._stream)
            current_batch = next_batch_gpu

            # Record a CUDA event so the main stream owns the tensors.
            if isinstance(current_batch, torch.Tensor):
                current_batch.record_stream(torch.cuda.current_stream())
            elif isinstance(current_batch, (list, tuple)):
                for t in current_batch:
                    if isinstance(t, torch.Tensor):
                        t.record_stream(torch.cuda.current_stream())

            # Start prefetching the *next* batch while current is yielded.
            try:
                next_batch = next(loader_iter)
                with torch.cuda.stream(self._stream):
                    next_batch_gpu = _move_to_device(next_batch)
            except StopIteration:
                next_batch_gpu = None

            yield current_batch

            if next_batch_gpu is None:
                break

    def __len__(self) -> int:
        return len(self.loader)
