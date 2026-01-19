import os
from typing import Sequence

from paddle._typing import PlaceLike
from paddle.io import BatchSampler, Dataset, DataLoader

from data.datasets import BaseDataset


class BaseDataLoader(DataLoader):
    def __init__(
            self, dataset: Dataset, batch_size: int = 1, use_collect_fn=False,
            places: PlaceLike | Sequence[PlaceLike] | None = None,
            drop_last: bool = False, num_workers: int = min(12, os.cpu_count() // 2),
            batch_sampler: BatchSampler | None = None, shuffle: bool = False, collate_fn=None,
            persistent_workers: bool = True, **kwargs
    ) -> None:
        super().__init__(
            dataset=dataset,
            places=places,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers if os.name != 'nt' else 0,  # win 下使用多线程容易崩溃
            collate_fn=dataset.collect_fn if use_collect_fn and isinstance(dataset, BaseDataset) else collate_fn,
            persistent_workers=persistent_workers,
            **kwargs
        )
