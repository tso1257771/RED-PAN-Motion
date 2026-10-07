"""BaseAdapter — protocol every per-dataset adapter must implement."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, Optional
from ..schema import WaveformSample, Category


class BaseAdapter(ABC):
    """Per-dataset adapter that yields WaveformSample objects.

    Subclasses set:
        name      : dataset short name ('STEAD', 'GeoNet', ...).
        category  : the category emitted by this adapter ('singleEQ' | 'noise' | ...).
                    A dataset that produces multiple categories should be
                    factored into multiple adapter classes (or one parameterised
                    by category) — one (name, category) pair per H5 file.

    Subclasses implement:
        __iter__  : yields WaveformSample objects, already validated.
                    The adapter handles native-format reads, sample positioning
                    (P at random offset within window for events), spectrum-
                    matched noise padding for the noise category, and split
                    assignment.

    Adapters MUST NOT do per-channel std normalization or any in-place
    modification of training-time targets — those belong to the dataloader.
    """

    name: str
    category: Category

    def __init__(self, *, max_samples: Optional[int] = None,
                 split_filter: Optional[str] = None):
        """
        Args:
            max_samples : optional cap (across all splits) for fast smoke tests.
            split_filter: if set, only yield samples whose `split` matches.
        """
        self.max_samples = max_samples
        self.split_filter = split_filter

    @abstractmethod
    def __iter__(self) -> Iterator[WaveformSample]:
        """Yield validated WaveformSample objects."""
        raise NotImplementedError

    def describe(self) -> str:
        """Short human-readable summary of what this adapter produces."""
        return f"{self.__class__.__name__}(name={self.name!r}, category={self.category!r})"
