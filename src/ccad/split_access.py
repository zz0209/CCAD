"""Lazy split loading with an explicit record of loader calls, not an OS audit."""
from collections.abc import Mapping


class SplitAccess(Mapping):
    def __init__(self, splits, loader):
        self._splits = tuple(splits)
        self._loader = loader
        self._cache = {}
        self.events = []

    def __getitem__(self, split):
        if split not in self._splits:
            raise KeyError(f"split not allowed: {split}")
        if split not in self._cache:
            event = {"split": split, "status": "attempted"}
            self.events.append(event)
            try:
                self._cache[split] = self._loader(split)
            except Exception:
                event["status"] = "failed"
                raise
            event["status"] = "loaded"
        return self._cache[split]

    def __iter__(self):
        return iter(self._splits)

    def __len__(self):
        return len(self._splits)

    def requested(self, split):
        return any(event["split"] == split for event in self.events)
