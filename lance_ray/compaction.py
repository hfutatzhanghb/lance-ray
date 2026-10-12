import json
import logging
from collections.abc import Callable
from typing import Any, Optional, cast

import lance
from lance.lance import CompactionMetrics
from lance.optimize import Compaction, CompactionOptions, CompactionTask
from ray.util.multiprocessing import Pool

from .utils import (
    get_namespace_kwargs,
    get_or_create_namespace,
    resolve_namespace_table,
    validate_uri_or_namespace,
)

logger = logging.getLogger(__name__)


def _handle_compaction_task(
    dataset_uri: str,
    storage_options: Optional[dict[str, str]] = None,
    namespace_impl: Optional[str] = None,
    namespace_properties: Optional[dict[str, str]] = None,
    table_id: Optional[list[str]] = None,
) -> Callable[[CompactionTask], dict[str, Any]]:
    """
    Create a function to handle compaction task execution for use with Pool.
    This function returns a callable that can be used with Pool.map_async
    to execute compaction tasks.
    """

    def func(task: CompactionTask) -> dict[str, Any]:
        """
        Execute a compaction task.

        Args:
            task: CompactionTask to execute

        Returns:
            Dictionary with status and result information
        """
        try:
            namespace_kwargs = get_namespace_kwargs(
                namespace_impl, namespace_properties, table_id
            )

            # Load dataset
            dataset = lance.LanceDataset(
                dataset_uri,
                storage_options=storage_options,
                **namespace_kwargs,
            )

            logger.info(f"Executing compaction task for fragments {task.fragments}")

            # Execute the compaction task
            result = task.execute(dataset)

            logger.info(
                f"Compaction task completed successfully for fragments {task.fragments}"
            )

            return {
                "status": "success",
                "fragments": task.fragments,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Compaction task failed for fragments {task.fragments}: {e}")
            return {
                "status": "error",
                "fragments": task.fragments,
                "error": str(e),
            }

    return func


def _fragment_id(fragment: Any) -> int:
    """Return the integer id of a fragment id or ``FragmentMetadata``."""
    if isinstance(fragment, int):
        return fragment
    for name in ("id", "fragment_id"):
        value = getattr(fragment, name, None)
        if isinstance(value, int):
            return value
        if callable(value):
            called = value()
            if isinstance(called, int):
                return called
    raise TypeError(
        "CompactionTask.fragments entries must be fragment ids or "
        f"FragmentMetadata, got {type(fragment).__name__}"
    )


def _estimate_fragment_memory(
    dataset: lance.LanceDataset,
    fragment_ids: list[int],
    avg_row_size_bytes: Optional[float] = None,
) -> dict[str, Any]:
    """Estimate the memory needed to compact ``fragment_ids``."""
    fragment_details: list[dict[str, int]] = []
    total_rows = 0
    total_file_size_bytes = 0

    for frag_id in fragment_ids:
        fragment = dataset.get_fragment(frag_id)
        if fragment is None:
            logger.warning(f"Fragment {frag_id} not found")
            continue

        frag_rows = int(fragment.count_rows())
        total_rows += frag_rows

        frag_file_size = 0
        for data_file in fragment.data_files():
            file_size = getattr(data_file, "file_size_bytes", None)
            if isinstance(file_size, int):
                frag_file_size += file_size

        total_file_size_bytes += frag_file_size
        fragment_details.append(
            {
                "fragment_id": frag_id,
                "num_rows": frag_rows,
                "file_size_bytes": frag_file_size,
            }
        )

    if avg_row_size_bytes is None and total_rows > 0:
        avg_row_size_bytes = total_file_size_bytes / total_rows

    # Compaction materializes Arrow data and temporary rewrite buffers.
    # A 2x file-size estimate covers that overhead for planning.
    memory_multiplier = 2.0
    return {
        "total_rows": total_rows,
        "total_file_size_bytes": total_file_size_bytes,
        "estimated_memory_bytes": int(total_file_size_bytes * memory_multiplier),
        "avg_row_size_bytes": avg_row_size_bytes,
        "fragment_details": fragment_details,
    }


def _subset_compaction_task(
    task: CompactionTask, fragments: list[Any]
) -> CompactionTask:
    """Return ``task`` limited to ``fragments``, keeping its read version and options.

    ``CompactionTask`` has no constructor for a fragment subset. The task JSON
    already contains the serialized ``Fragment`` records, so a split task is the
    original JSON with ``task.fragments`` filtered by id.
    """
    payload = json.loads(task.json())
    if not isinstance(payload, dict):
        raise TypeError("CompactionTask JSON must be an object")
    task_payload = payload.get("task")
    if not isinstance(task_payload, dict):
        raise TypeError("CompactionTask JSON is missing task")
    records = task_payload.get("fragments")
    if not isinstance(records, list):
        raise TypeError("CompactionTask JSON is missing task.fragments")

    by_id: dict[int, Any] = {}
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("id"), int):
            by_id[int(record["id"])] = record

    selected: list[Any] = []
    for fragment in fragments:
        frag_id = _fragment_id(fragment)
        record = by_id.get(frag_id)
        if record is None:
            raise ValueError(f"fragment {frag_id} is not part of the compaction task")
        selected.append(record)
    task_payload["fragments"] = selected
    return CompactionTask.from_json(json.dumps(payload))


def _create_memory_aware_compaction_plan(
    dataset: lance.LanceDataset,
    compaction_options: CompactionOptions,
    max_memory_per_task_bytes: int,
) -> tuple[list[CompactionTask], dict[str, Any]]:
    """Split compaction tasks so each one stays within ``max_memory_per_task_bytes``.

    ``CompactionTask.fragments`` is a list of ``FragmentMetadata``, while
    ``Dataset.get_fragment`` requires an integer id. Ids are extracted before
    any fragment lookup.
    """
    fragments = dataset.get_fragments()
    fragment_ids = [_fragment_id(fragment) for fragment in fragments]
    if not fragment_ids:
        return [], {"message": "No fragments to compact"}

    memory_estimate = _estimate_fragment_memory(dataset, fragment_ids)
    avg_row_size_bytes = memory_estimate["avg_row_size_bytes"]
    logger.info(
        f"Analyzed {len(fragment_ids)} fragments: "
        f"{memory_estimate['total_rows']} rows, "
        f"{memory_estimate['total_file_size_bytes'] / (1024 * 1024):.2f} MB total file size"
    )

    initial_plan = Compaction.plan(dataset, compaction_options)
    if initial_plan.num_tasks() == 0:
        return [], {
            "message": "No compaction needed based on options",
            "initial_plan": initial_plan,
        }

    final_tasks: list[CompactionTask] = []
    tasks_split = 0
    for task in initial_plan.tasks:
        task_fragment_ids = [_fragment_id(fragment) for fragment in task.fragments]
        task_memory_estimate = _estimate_fragment_memory(
            dataset, task_fragment_ids, avg_row_size_bytes
        )
        if task_memory_estimate["estimated_memory_bytes"] <= max_memory_per_task_bytes:
            final_tasks.append(task)
            continue

        tasks_split += 1
        logger.info(
            f"Splitting compaction task for {len(task.fragments)} fragments "
            f"({task_memory_estimate['estimated_memory_bytes'] / (1024 * 1024):.2f} MB) "
            "into smaller chunks"
        )
        fragment_details = task_memory_estimate["fragment_details"]
        if fragment_details:
            largest_fragment_mb = max(
                detail["file_size_bytes"] * 2.0 / (1024 * 1024)
                for detail in fragment_details
            )
            limit_mb = max_memory_per_task_bytes / (1024 * 1024)
            if largest_fragment_mb > limit_mb:
                logger.warning(
                    f"Memory limit ({limit_mb:.2f} MB) is smaller than the largest "
                    f"fragment ({largest_fragment_mb:.2f} MB). Individual fragments "
                    "may still exceed the limit."
                )

        current_batch: list[Any] = []
        current_batch_size = 0
        for fragment in task.fragments:
            frag_id = _fragment_id(fragment)
            frag_detail = next(
                (
                    detail
                    for detail in fragment_details
                    if detail["fragment_id"] == frag_id
                ),
                None,
            )
            if frag_detail is None:
                continue
            frag_memory = int(frag_detail["file_size_bytes"] * 2.0)
            if (
                current_batch
                and current_batch_size + frag_memory > max_memory_per_task_bytes
            ):
                final_tasks.append(_subset_compaction_task(task, current_batch))
                current_batch = [fragment]
                current_batch_size = frag_memory
            else:
                current_batch.append(fragment)
                current_batch_size += frag_memory
        if current_batch:
            final_tasks.append(_subset_compaction_task(task, current_batch))

    planning_metrics = {
        "original_tasks": initial_plan.num_tasks(),
        "final_tasks": len(final_tasks),
        "tasks_split": tasks_split,
        "max_memory_per_task_bytes": max_memory_per_task_bytes,
        "total_estimated_memory_bytes": memory_estimate["estimated_memory_bytes"],
    }
    logger.info(
        f"Created memory-aware compaction plan: {len(final_tasks)} tasks "
        f"(originally {initial_plan.num_tasks()}, split {tasks_split} tasks)"
    )
    return final_tasks, planning_metrics


def compact_files(
    uri: Optional[str] = None,
    *,
    table_id: Optional[list[str]] = None,
    compaction_options: Optional[CompactionOptions] = None,
    num_workers: int = 4,
    storage_options: Optional[dict[str, str]] = None,
    namespace_impl: Optional[str] = None,
    namespace_properties: Optional[dict[str, str]] = None,
    ray_remote_args: Optional[dict[str, Any]] = None,
    max_memory_per_task_bytes: Optional[int] = None,
) -> Optional[CompactionMetrics]:
    """
    Compact files in a Lance dataset using distributed Ray workers.

    This function distributes the compaction process across multiple Ray workers,
    with each worker executing a subset of compaction tasks. The results are then
    committed as a single compaction operation.

    Args:
        uri: The URI of the Lance dataset to compact. Either uri OR
            (namespace_impl + table_id) must be provided.
        table_id: The table identifier as a list of strings. Must be provided
            together with namespace_impl.
        compaction_options: Options for the compaction operation.
        num_workers: Number of Ray workers to use (default: 4).
        storage_options: Storage options for the dataset.
        namespace_impl: The namespace implementation type (e.g., "rest", "dir").
            Used together with table_id for resolving the dataset location and
            credentials vending in distributed workers.
        namespace_properties: Properties for connecting to the namespace.
            Used together with namespace_impl and table_id.
        ray_remote_args: Options for Ray tasks (e.g., num_cpus, resources).
        max_memory_per_task_bytes: Maximum estimated memory for one compaction
            task. When set, tasks whose fragment files exceed this budget are
            split before they are sent to Ray. ``CompactionTask.fragments``
            contains ``FragmentMetadata`` objects; planning resolves those to
            integer ids before calling ``Dataset.get_fragment``. When omitted,
            Lance's standard plan is executed unchanged.

    Returns:
        CompactionMetrics with statistics from the compaction operation.

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If compaction fails.
    """
    validate_uri_or_namespace(uri, namespace_impl, table_id)

    # Resolve URI and get storage options from namespace if provided
    uri, merged_storage_options = resolve_namespace_table(
        uri, storage_options, namespace_impl, namespace_properties, table_id
    )

    namespace_kwargs = get_namespace_kwargs(
        namespace_impl, namespace_properties, table_id
    )

    # Load dataset
    dataset = lance.LanceDataset(
        uri,
        storage_options=merged_storage_options,
        **namespace_kwargs,
    )

    logger.info("Starting distributed compaction")

    # Compaction.plan requires a dict; CompactionOptions is a TypedDict, so
    # an empty instance stands in for "all defaults" when the caller omits it.
    # It is declared with ``total=True`` upstream even though every key is
    # optional at runtime, hence the cast instead of ``CompactionOptions()``.
    options = compaction_options or cast(CompactionOptions, {})
    if max_memory_per_task_bytes is not None:
        logger.info(
            "Using memory-aware compaction planning with max "
            f"{max_memory_per_task_bytes / (1024 * 1024):.2f} MB per task"
        )
        compaction_tasks, planning_metrics = _create_memory_aware_compaction_plan(
            dataset, options, max_memory_per_task_bytes
        )
        if planning_metrics.get("tasks_split", 0):
            logger.info(
                f"Split {planning_metrics['tasks_split']} large tasks into "
                f"{planning_metrics['final_tasks']} memory-safe tasks"
            )
    else:
        compaction_plan = Compaction.plan(dataset, options)
        compaction_tasks = list(compaction_plan.tasks)

    logger.info(f"Compaction plan created with {len(compaction_tasks)} tasks")

    if len(compaction_tasks) == 0:
        logger.info("No compaction tasks needed")
        return None

    # Adjust num_workers if needed
    if num_workers > len(compaction_tasks):
        num_workers = len(compaction_tasks)
        logger.info(f"Adjusted num_workers to {num_workers} to match task count")

    # Step 2: Execute tasks in parallel using Ray Pool
    pool = Pool(processes=num_workers, ray_remote_args=ray_remote_args)

    # Create the task handler function
    task_handler = _handle_compaction_task(
        dataset_uri=uri,
        storage_options=merged_storage_options,
        namespace_impl=namespace_impl,
        namespace_properties=namespace_properties,
        table_id=table_id,
    )

    # Submit tasks using Pool.map_async
    rst_futures = pool.map_async(
        task_handler,
        compaction_tasks,
        chunksize=1,
    )

    # Wait for results
    try:
        results = rst_futures.get()
    except Exception as e:
        raise RuntimeError(f"Failed to complete distributed compaction: {e}") from e
    finally:
        pool.close()
        pool.join()

    # Check for failures
    failed_results = [r for r in results if r["status"] == "error"]
    if failed_results:
        error_messages = [r["error"] for r in failed_results]
        raise RuntimeError(f"Compaction failed: {'; '.join(error_messages)}")

    # Step 3: Collect successful RewriteResult objects
    successful_results = [r for r in results if r["status"] == "success"]
    if not successful_results:
        raise RuntimeError("No successful compaction results found")

    rewrites = [r["result"] for r in successful_results]

    logger.info(
        f"Collected {len(rewrites)} successful compaction results, committing..."
    )

    # Step 4: Commit the compaction
    metrics = Compaction.commit(dataset, rewrites)

    logger.info(f"Compaction completed successfully. Metrics: {metrics}")

    return metrics


def compact_database(
    *,
    database: list[str],
    namespace_impl: str,
    namespace_properties: Optional[dict[str, str]] = None,
    compaction_options: Optional[CompactionOptions] = None,
    num_workers: int = 4,
    storage_options: Optional[dict[str, str]] = None,
    ray_remote_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Compact all tables under a given database (namespace) using distributed Ray workers.

    This function lists all tables under the specified database via the namespace API,
    then runs :func:`compact_files` on each table. Use this when you want to compact
    an entire database instead of a single table.

    Args:
        database: The database (namespace) identifier as a list of path segments,
            e.g. ``["my_database"]``. All tables under this namespace will be compacted.
        namespace_impl: The namespace implementation type (e.g. ``"rest"``, ``"dir"``).
            Required for resolving table locations and credentials.
        namespace_properties: Properties for connecting to the namespace.
        compaction_options: Options for the compaction operation (used for every table).
        num_workers: Number of Ray workers per table (default: 4).
        storage_options: Storage options for the datasets.
        ray_remote_args: Options for Ray tasks (e.g. num_cpus, resources).

    Returns:
        A list of dicts, one per table, with keys:
        - ``"table_id"``: ``list[str]`` – full table identifier (database + table name).
        - ``"metrics"``: :class:`~lance.lance.CompactionMetrics` or ``None`` –
          compaction result for that table, or ``None`` if no compaction was needed.

    Raises:
        ValueError: If database is empty or namespace_impl is not provided.
        RuntimeError: If listing tables fails or any table compaction fails.

    Example:
        >>> results = compact_database(
        ...     database=["my_db"],
        ...     namespace_impl="dir",
        ...     namespace_properties={"root": "/path/to/tables"},
        ...     compaction_options=CompactionOptions(target_rows_per_fragment=10000),
        ...     num_workers=2,
        ... )
        >>> for item in results:
        ...     print(item["table_id"], item["metrics"])
    """
    if not database:
        raise ValueError("'database' must be a non-empty list of path segments.")
    if not namespace_impl:
        raise ValueError("'namespace_impl' is required when using compact_database.")

    from lance_namespace import ListTablesRequest

    namespace = get_or_create_namespace(namespace_impl, namespace_properties)
    if namespace is None:
        raise RuntimeError(
            "Failed to create namespace from namespace_impl and namespace_properties."
        )

    # List all tables under the database (namespace) with pagination
    all_tables: list[str] = []
    page_token: Optional[str] = None
    limit = 500

    while True:
        request = ListTablesRequest(
            id=database,
            page_token=page_token,
            limit=limit,
        )
        response = namespace.list_tables(request)
        all_tables.extend(response.tables)
        page_token = getattr(response, "page_token", None)
        if not page_token:
            break

    if not all_tables:
        logger.info("No tables found under database %s, nothing to compact.", database)
        return []

    # table_id = database + [table_name] for each table under this namespace
    table_ids = [database + [t] for t in all_tables]
    results: list[dict[str, Any]] = []

    for table_id in table_ids:
        logger.info("Compacting table %s", table_id)
        try:
            metrics = compact_files(
                uri=None,
                table_id=table_id,
                compaction_options=compaction_options,
                num_workers=num_workers,
                storage_options=storage_options,
                namespace_impl=namespace_impl,
                namespace_properties=namespace_properties,
                ray_remote_args=ray_remote_args,
            )
            results.append({"table_id": table_id, "metrics": metrics})
        except Exception as e:
            logger.exception("Compaction failed for table %s: %s", table_id, e)
            raise RuntimeError(f"Compaction failed for table {table_id}: {e}") from e

    return results
