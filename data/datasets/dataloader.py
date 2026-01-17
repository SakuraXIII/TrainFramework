from typing import Sequence

from paddle._typing import PlaceLike
from paddle.io import DataLoader, BatchSampler

from data.datasets import BaseDataset


class BaseDataLoader(DataLoader):
    def __init__(
            self, dataset: BaseDataset, batch_size: int = 1, use_collect_fn=False,
            places: PlaceLike | Sequence[PlaceLike] | None = None,
            drop_last: bool = False, num_workers: int = 0,
            batch_sampler: BatchSampler | None = None, shuffle: bool = False,
            persistent_workers: bool = True, **kwargs
    ) -> None:
        super().__init__(
            dataset=dataset,
            places=places,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=dataset.collect_fn if use_collect_fn else None,
            persistent_workers=persistent_workers,
            **kwargs
        )
