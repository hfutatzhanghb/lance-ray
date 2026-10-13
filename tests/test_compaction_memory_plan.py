"""Memory-aware compaction must use integer fragment ids and rebuild split tasks."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, cast

import lance_ray.compaction as compaction
import pytest
from lance.optimize import CompactionOptions


class _FragmentMetadata:
    def __init__(self, fragment_id: int) -> None:
        self.id = fragment_id


class _StoredFragment:
    def __init__(
        self,
        fragment_id: int,
        file_size_bytes: int | None,
        *,
        missing_size: bool = False,
    ) -> None:
        self.fragment_id = fragment_id
        self._file_size_bytes = file_size_bytes
        self._missing_size = missing_size

    def count_rows(self) -> int:
        return 10

    def data_files(self) -> list[SimpleNamespace]:
        if self._missing_size:
            return [SimpleNamespace(file_size_bytes=None)]
        return [SimpleNamespace(file_size_bytes=self._file_size_bytes)]


class _Dataset:
    def __init__(
        self, sizes: dict[int, int | None], *, missing: set[int] | None = None
    ) -> None:
        missing = missing or set()
        self.fragments = [
            _StoredFragment(
                fragment_id,
                size,
                missing_size=fragment_id in missing,
            )
            for fragment_id, size in sizes.items()
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
    def __init__(self, fragments: list[Any], *, read_version: int = 7) -> None:
        self.fragments = fragments
        self.read_version = read_version

    def json(self) -> str:
        return json.dumps(
            {
                "read_version": self.read_version,
                "options": {"target_rows_per_fragment": 1024},
                "task": {
                    "fragments": [
                        {
                            "id": compaction._fragment_id(fragment),
                            "files": [
                                {"path": f"f{compaction._fragment_id(fragment)}.lance"}
                            ],
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


def test_fragment_id_accepts_int_and_metadata() -> None:
    assert compaction._fragment_id(3) == 3
    assert compaction._fragment_id(_FragmentMetadata(9)) == 9
    assert compaction._fragment_id(SimpleNamespace(fragment_id=4)) == 4
    assert compaction._fragment_id(SimpleNamespace(metadata=SimpleNamespace(id=5))) == 5


def test_fragment_id_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="FragmentMetadata"):
        compaction._fragment_id("frag-1")


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


def test_memory_plan_splits_via_json_preserving_metadata(
    monkeypatch: Any,
) -> None:
    dataset = _Dataset({1: 1_000, 2: 1_000})
    original = _Task([_FragmentMetadata(1), _FragmentMetadata(2)], read_version=7)
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
    assert all(
        item["options"] == {"target_rows_per_fragment": 1024} for item in rebuilt
    )
    # Full fragment records are preserved, not reduced to bare integer ids.
    assert all(
        "files" in frag and "physical_rows" in frag
        for item in rebuilt
        for frag in item["task"]["fragments"]
    )
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


def test_memory_plan_warns_when_single_fragment_exceeds_budget(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    dataset = _Dataset({1: 5_000, 2: 100})
    original = _Task([_FragmentMetadata(1), _FragmentMetadata(2)])
    monkeypatch.setattr(
        "lance_ray.compaction.Compaction.plan",
        lambda dataset, options: _Plan([original]),
    )
    monkeypatch.setattr(
        "lance_ray.compaction.CompactionTask.from_json",
        lambda payload: json.loads(payload),
    )

    with caplog.at_level(logging.WARNING, logger="lance_ray.compaction"):
        tasks, metrics = compaction._create_memory_aware_compaction_plan(
            cast(Any, dataset),
            _options(),
            1_000,
        )

    assert metrics["tasks_split"] == 1
    assert len(tasks) == 2
    assert any("may still exceed" in record.message for record in caplog.records)
    assert any(
        "do not guarantee" in record.message.lower()
        or "cannot guarantee" in record.message.lower()
        or "may still exceed" in record.message
        for record in caplog.records
    )


def test_memory_plan_unknown_file_size_is_not_treated_as_zero(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # Fragment 1 has unknown size; fragment 2 is small. Packing must not treat
    # the unknown size as zero and keep both fragments in one under-budget task.
    dataset = _Dataset({1: None, 2: 100}, missing={1})
    original = _Task([_FragmentMetadata(1), _FragmentMetadata(2)])
    monkeypatch.setattr(
        "lance_ray.compaction.Compaction.plan",
        lambda dataset, options: _Plan([original]),
    )
    monkeypatch.setattr(
        "lance_ray.compaction.CompactionTask.from_json",
        lambda payload: json.loads(payload),
    )

    with caplog.at_level(logging.WARNING, logger="lance_ray.compaction"):
        tasks, metrics = compaction._create_memory_aware_compaction_plan(
            cast(Any, dataset),
            _options(),
            10_000,
        )

    assert metrics["tasks_split"] == 1
    assert len(tasks) == 2
    assert any("unknown" in record.message.lower() for record in caplog.records)


def test_compact_database_forwards_memory_limit(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class _Namespace:
        def list_tables(self, request: Any) -> SimpleNamespace:
            assert request.id == ["db"]
            return SimpleNamespace(tables=["t1", "t2"], page_token=None)

    monkeypatch.setattr(
        "lance_ray.compaction.get_or_create_namespace",
        lambda namespace_impl, namespace_properties: _Namespace(),
    )

    def fake_compact_files(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "metrics"

    monkeypatch.setattr("lance_ray.compaction.compact_files", fake_compact_files)

    results = compaction.compact_database(
        database=["db"],
        namespace_impl="dir",
        max_memory_per_task_bytes=1024,
    )

    assert [call["table_id"] for call in calls] == [["db", "t1"], ["db", "t2"]]
    assert all(call["max_memory_per_task_bytes"] == 1024 for call in calls)
    assert results == [
        {"table_id": ["db", "t1"], "metrics": "metrics"},
        {"table_id": ["db", "t2"], "metrics": "metrics"},
    ]


def test_estimate_fragment_memory_stores_integer_ids() -> None:
    dataset = _Dataset({8: 50, 9: 50})
    estimate = compaction._estimate_fragment_memory(cast(Any, dataset), [8, 9])
    assert [detail["fragment_id"] for detail in estimate["fragment_details"]] == [8, 9]
    assert all(
        isinstance(detail["fragment_id"], int)
        for detail in estimate["fragment_details"]
    )
