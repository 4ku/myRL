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
        seed: int,
    ):
        self._example_transition = example_transition
        self._capacity = int(capacity)
        self._size = 0
        self._insert_index = 0
        self._rng = np.random.RandomState(seed)

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

    def sample(
        self, 
        batch_size: int, 
        sequence_length: int,
        sample_latest: bool,
    ) -> FrozenDict:
        """Sample sequences of consecutive transitions.

        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence
            sample_latest: If True, samples the most recently added sequences 
                          and updates the buffer state to remove the sampled data.

        Returns:
            FrozenDict with shape (batch_size, sequence_length, ...) for each field.
            If sequence_length=1, the shape is (batch_size, ...).
        """
        if sequence_length <= 0:
            raise ValueError("sequence_length must be >= 1")
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if self._size < sequence_length:
            raise RuntimeError(
                f"Not enough data to sample sequences of length {sequence_length} (size={self._size})."
            )

        # Sample valid logical starting indices.
        # Logical index 0 corresponds to the oldest element in the buffer (at self._insert_index if full).
        # There are (size - sequence_length + 1) valid sequences in total.
        num_valid_sequences = self.size - sequence_length + 1
        
        if sample_latest:
            if num_valid_sequences < batch_size:
                 raise RuntimeError(
                     f"Not enough valid sequences ({num_valid_sequences}) to sample {batch_size} latest sequences."
                 )
            # Select the last 'batch_size' logical indices
            logical_idxs = np.arange(num_valid_sequences - batch_size, num_valid_sequences)
        else:
            logical_idxs = self._rng.randint(num_valid_sequences, size=batch_size)
        
        # Map logical indices to physical indices in the circular buffer
        # If buffer is not full, oldest is at 0.
        # If buffer is full, oldest is at self._insert_index.
        start_idx_offset = self._insert_index if self.size == self.capacity else 0
        idxs = (logical_idxs + start_idx_offset) % self._capacity

        # Update buffer state if sampling latest (consume data)
        if sample_latest:
            self._size -= batch_size
            self._insert_index = (self._insert_index - batch_size) % self._capacity

        if sequence_length == 1:

            def _tree_gather(storage: ArrayTree) -> ArrayTree:
                if isinstance(storage, dict):
                    return {k: _tree_gather(v) for k, v in storage.items()}
                return storage[idxs]

            batch_tree = _tree_gather(self._storage)
            return FrozenDict(batch_tree)

        # Generate all indices for sequences: (batch_size, sequence_length)
        # Use modulo arithmetic to handle wrapping at the end of the physical buffer
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]
        all_idxs = all_idxs % self._capacity
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
