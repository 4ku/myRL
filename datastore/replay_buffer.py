import jax
import numpy as np
import collections
from typing import Any, Dict, Iterable, Optional, Union
from flax.core.frozen_dict import FrozenDict

ArrayTree = Union[np.ndarray, Dict[str, "ArrayTree"]]


class ReplayBuffer:
    """Circular replay buffer supporting sequence sampling.

    Stores transitions in a circular buffer and samples sequences of transitions.
    """

    def __init__(
        self,
        example_transition: Dict[str, Any],
        capacity: int,
    ):
        self._example_transition = example_transition
        self._capacity = int(capacity)
        self._size = 0
        self._insert_index = 0

        def _tree_init_storage_from_example(
            example: ArrayTree, capacity: int
        ) -> ArrayTree:
            if isinstance(example, dict):
                return {
                    k: _tree_init_storage_from_example(v, capacity)
                    for k, v in example.items()
                }
            arr = np.asarray(example)
            return np.empty((capacity, *arr.shape), dtype=arr.dtype)

        self._storage: ArrayTree = _tree_init_storage_from_example(
            example_transition, self._capacity
        )

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def clear(self) -> None:
        """Reset the buffer to empty state."""
        self._size = 0
        self._insert_index = 0

    def insert(self, transition: Dict[str, Any]) -> None:
        """Insert a single transition into the buffer."""

        def _tree_write_single(dst: ArrayTree, src: ArrayTree, index: int) -> None:
            """Write a single transition `src` into `dst` at position `index`."""
            if isinstance(dst, dict):
                assert isinstance(src, dict), f"Expected dict, got {type(src).__name__}"
                for k in dst.keys():
                    assert k in src, f"Key {k} not found in src"
                    _tree_write_single(dst[k], src[k], index)
                return
            src = np.asarray(src)
            dst = np.asarray(dst)
            assert dst.shape == (
                self._capacity,
                *src.shape,
            ), f"Expected shape {dst.shape[1:]}, got {src.shape}"
            dst[index] = src

        _tree_write_single(self._storage, transition, self._insert_index)

        # Circular buffer: wrap around when reaching capacity
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, sequence_length: int = 2) -> FrozenDict:
        """Sample sequences of consecutive transitions.

        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence

        Returns:
            FrozenDict with shape (batch_size, sequence_length, ...) for each field
        """
        if sequence_length <= 0:
            raise ValueError("sequence_length must be >= 1")
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if self._size < sequence_length:
            raise RuntimeError(
                f"Not enough data to sample sequences of length {sequence_length} (size={self._size})."
            )

        # Sample starting indices ensuring sequences don't wrap around buffer boundary
        max_start_idx = self.size - sequence_length + 1
        idxs = np.random.randint(max_start_idx, size=batch_size)
        # Generate all indices for sequences: (batch_size, sequence_length)
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]
        all_idxs = all_idxs.flatten()

        def _tree_gather_and_reshape(storage: ArrayTree) -> ArrayTree:
            """Gather indices and reshape to (batch_size, sequence_length, ...)."""
            if isinstance(storage, dict):
                return {k: _tree_gather_and_reshape(v) for k, v in storage.items()}
            gathered = storage[all_idxs]
            new_shape = (batch_size, sequence_length) + storage.shape[1:]
            return gathered.reshape(new_shape)

        batch_tree = _tree_gather_and_reshape(self._storage)
        return FrozenDict(batch_tree)

    def get_iterator(
        self,
        *,
        queue_size: int = 2,
        device: Optional[Any] = None,
        sample_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Iterable[FrozenDict]:
        """Create an infinite iterator that yields batches.

        Args:
            queue_size: Number of batches to prefetch
            device: JAX device to place batches on
            sample_kwargs: Arguments passed to sample() method

        Yields:
            Batches from the replay buffer
        """
        queue = collections.deque()

        def enqueue(n: int) -> None:
            for _ in range(n):
                batch: FrozenDict
                batch = self.sample(**sample_kwargs)
                if device is not None:
                    batch = jax.device_put(batch, device=device)
                queue.append(batch)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


__all__ = ["ReplayBuffer"]
