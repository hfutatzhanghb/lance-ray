"""Memory-aware compaction must read integer ids from FragmentMetadata."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import lance_ray.compaction as compaction
from lance.optimize import CompactionOptions


class _FragmentMetadata:
    def __init__(self, fragment_id: int) -> None:
        self.id = fragment_id


class _StoredFragment:
    def __init__(self, fragment_id: int, file_size_bytes: int) -> None:
        self.fragment_id = fragment_id
        self._file_size_bytes = file_size_bytes

    def count_rows(self) -> int:
        return 10

    def data_files(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(file_size_bytes=self._file_size_bytes)]


class _Dataset:
    def __init__(self, sizes: dict[int, int]) -> None:
        self.fragments = [
            _StoredFragment(fragment_id, size) for fragment_id, size in sizes.items()
        ]
        self.requested_ids: list[Any] = []

    def get_fragments(self) -> list[_StoredFragment]:
        return self.fragments

    def get_fragment(self, fragment_id: Any) -> _StoredFragment | None:
        self.requested_ids.append(fragment_id)
        if not isinstance(fragment_id, int):
            raise TypeError(
                "argument 'fragment_id': 'FragmentMetadata' object "
                "cannot be interpreted as an integer"
            )
        for fragment in self.fragments:
            if fragment.fragment_id == fragment_id:
                return fragment
        return None


class _Task:
    def __init__(self, fragments: list[Any]) -> None:
        self.fragments = fragments

    def json(self) -> str:
        return json.dumps(
            {
                "read_version": 7,
                "options": {},
                "task": {
                    "fragments": [
                        {
                            "id": compaction._fragment_id(fragment),
                            "files": [],
                            "physical_rows": 10,
                        }
                        for fragment in self.fragments
                    ]
                },
            }
        )


class _Plan:
    def __init__(self, tasks: list[_Task]) -> None:
        self._tasks = tasks

    @property
    def tasks(self) -> list[_Task]:
        return self._tasks

    def num_tasks(self) -> int:
        return len(self._tasks)


def _options() -> CompactionOptions:
    return cast(CompactionOptions, {})


def test_memory_plan_looks_up_integer_ids_from_fragment_metadata(
    monkeypatch: Any,
) -> None:
    dataset = _Dataset({1: 100, 2: 100})
    original = _Task([_FragmentMetadata(1), _FragmentMetadata(2)])
    monkeypatch.setattr(
        "lance_ray.compaction.Compaction.plan",
        lambda dataset, options: _Plan([original]),
    )

    tasks, metrics = compaction._create_memory_aware_compaction_plan(
        cast(Any, dataset), _options(), 10_000
    )

    assert len(tasks) == 1
    assert cast(Any, tasks[0]) is original
    assert dataset.requested_ids
    assert all(isinstance(fragment_id, int) for fragment_id in dataset.requested_ids)
    assert set(dataset.requested_ids) == {1, 2}
    assert metrics["tasks_split"] == 0


def test_memory_plan_splits_fragment_metadata_without_passing_it_to_get_fragment(
    monkeypatch: Any,
) -> None:
    dataset = _Dataset({1: 1_000, 2: 1_000})
    original = _Task([_FragmentMetadata(1), _FragmentMetadata(2)])
    rebuilt: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "lance_ray.compaction.Compaction.plan",
        lambda dataset, options: _Plan([original]),
    )

    def from_json(payload: str) -> Any:
        decoded = json.loads(payload)
        assert isinstance(decoded, dict)
        rebuilt.append(decoded)
        return decoded

    monkeypatch.setattr("lance_ray.compaction.CompactionTask.from_json", from_json)

    tasks, metrics = compaction._create_memory_aware_compaction_plan(
        cast(Any, dataset),
        _options(),
        2_500,
    )
    split_tasks = cast(list[dict[str, Any]], tasks)

    assert metrics["tasks_split"] == 1
    assert len(split_tasks) == 2
    assert [task["task"]["fragments"][0]["id"] for task in split_tasks] == [1, 2]
    assert all(item["read_version"] == 7 for item in rebuilt)
    assert all(isinstance(fragment_id, int) for fragment_id in dataset.requested_ids)


def test_memory_plan_accepts_integer_fragment_ids(monkeypatch: Any) -> None:
    dataset = _Dataset({4: 100})
    original = _Task([4])
    monkeypatch.setattr(
        "lance_ray.compaction.Compaction.plan",
        lambda dataset, options: _Plan([original]),
    )

    tasks, metrics = compaction._create_memory_aware_compaction_plan(
        cast(Any, dataset), _options(), 10_000
    )

    assert len(tasks) == 1
    assert cast(Any, tasks[0]) is original
    assert 4 in dataset.requested_ids
    assert metrics["tasks_split"] == 0
