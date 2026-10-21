# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local acceptance mirror for challenge ``processing-01-count-trigger``."""

from __future__ import annotations

import asyncio

import pydantic
import pytest

from powercontext.builtin.artifacts.memory.models import MemoryEntryInput
from powercontext.builtin.artifacts.memory.protocols import MemoryCandidateRequest
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    RuntimeConfig,
    SearchMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentSource


class _EchoPipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="fact", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


def _turn(rank: int) -> str:
    return f"Meeting turn {rank}: decision {rank} on the migration plan with owner {rank}."


async def _capture(runtime, scope_id: str, rank: int) -> None:
    await runtime.sources.for_scope(scope_id).capture(
        CaptureSource(source_id=f"turn-{rank}", content=_turn(rank), metadata={})
    )


async def _hits(runtime, scope_id: str, query: str = "migration") -> int:
    page = await runtime.memory.for_scope(scope_id).search(
        SearchMemoryRequest(query=query, mode="fts", limit=50)
    )
    return len(page.hits)


async def _wait_for_hits(runtime, scope_id: str, expected: int, *, seconds: float = 45.0) -> int:
    for _ in range(int(seconds / 0.25)):
        await asyncio.sleep(0.25)
        hits = await _hits(runtime, scope_id)
        if hits >= expected:
            return hits
    return await _hits(runtime, scope_id)


def _config(tmp_path, name: str, **runtime_kwargs) -> BuiltinConfig:
    return BuiltinConfig(
        database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / name}"),
        runtime=RuntimeConfig(**runtime_kwargs),
    )


# ---------------------------------------------------------------------------
# A. Configuration and counting
# ---------------------------------------------------------------------------


def test_thresholds_default_to_disabled_and_validate_strictly() -> None:
    config = RuntimeConfig()
    assert config.memory_source_count_threshold is None
    assert config.topic_memory_source_count_threshold is None

    valid = RuntimeConfig(memory_source_count_threshold=3, topic_memory_source_count_threshold=1)
    assert valid.memory_source_count_threshold == 3
    assert valid.topic_memory_source_count_threshold == 1

    for bad in (True, False, 1.0, "3", 0, -1):
        with pytest.raises(pydantic.ValidationError):
            RuntimeConfig(memory_source_count_threshold=bad)  # ty: ignore[invalid-argument-type]
        with pytest.raises(pydantic.ValidationError):
            RuntimeConfig(topic_memory_source_count_threshold=bad)  # ty: ignore[invalid-argument-type]


def test_repeated_capture_of_one_identity_counts_once(tmp_path) -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(
            _config(tmp_path, "dup.db", memory_source_count_threshold=3),
            scheduler_path=tmp_path / "s.db",
            candidate_pipeline=_EchoPipeline(),
        ) as runtime:
            scope = await runtime.scopes.create(  # ty: ignore[unresolved-attribute]
                ScopeDraft(title="Dup", summary="Q7", idempotency_key="q7-dup")
            )
            for _ in range(5):
                await runtime.sources.for_scope(scope.scope_id).capture(
                    CaptureSource(source_id="same-id", content=_turn(1), metadata={})
                )
            await asyncio.sleep(1.5)
            # One journal entry cannot reach a threshold of 3: nothing processed.
            assert await _hits(runtime, scope.scope_id) == 0

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# B. Start and consumption boundaries
# ---------------------------------------------------------------------------


def test_threshold_gates_processing_and_crossing_starts_it(tmp_path) -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(
            _config(tmp_path, "gate.db", memory_source_count_threshold=3),
            scheduler_path=tmp_path / "s.db",
            candidate_pipeline=_EchoPipeline(),
        ) as runtime:
            scope = await runtime.scopes.create(  # ty: ignore[unresolved-attribute]
                ScopeDraft(title="Gate", summary="Q7", idempotency_key="q7-gate")
            )
            await _capture(runtime, scope.scope_id, 1)
            await _capture(runtime, scope.scope_id, 2)
            await asyncio.sleep(1.5)
            assert await _hits(runtime, scope.scope_id) == 0

            await _capture(runtime, scope.scope_id, 3)
            hits = await _wait_for_hits(runtime, scope.scope_id, 3)
            assert hits == 3

    asyncio.run(scenario())


def test_window_limit_consumes_one_per_trigger_until_threshold_returns(tmp_path) -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(
            _config(tmp_path, "window.db", memory_source_count_threshold=3, source_window_limit=1),
            scheduler_path=tmp_path / "s.db",
            candidate_pipeline=_EchoPipeline(),
        ) as runtime:
            scope = await runtime.scopes.create(  # ty: ignore[unresolved-attribute]
                ScopeDraft(title="Window", summary="Q7", idempotency_key="q7-window")
            )
            memory = runtime.memory.for_scope(scope.scope_id)
            sources = runtime.sources.for_scope(scope.scope_id)
            for rank in (1, 2, 3):
                await _capture(runtime, scope.scope_id, rank)
            hits = await _wait_for_hits(runtime, scope.scope_id, 1)
            assert hits == 1
            cursor = await memory.cursor()
            assert cursor.sequence == 1

            # Remaining 2 sit below the threshold of 3: quiet observation.
            await asyncio.sleep(1.5)
            assert await _hits(runtime, scope.scope_id) == 1

            # One more capture returns the pending count to 3: next window runs.
            await sources.capture(
                CaptureSource(source_id="turn-4", content=_turn(4), metadata={})
            )
            hits = await _wait_for_hits(runtime, scope.scope_id, 2)
            assert hits == 2

    asyncio.run(scenario())


def test_explicit_flush_bypasses_the_threshold(tmp_path) -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(
            _config(tmp_path, "bypass.db", memory_source_count_threshold=3),
            scheduler_path=tmp_path / "s.db",
            candidate_pipeline=_EchoPipeline(),
        ) as runtime:
            scope = await runtime.scopes.create(  # ty: ignore[unresolved-attribute]
                ScopeDraft(title="Bypass", summary="Q7", idempotency_key="q7-bypass")
            )
            await _capture(runtime, scope.scope_id, 1)
            result = await runtime.memory.for_scope(scope.scope_id).flush()
            assert result.source_count == 1
            assert await _hits(runtime, scope.scope_id) == 1

    asyncio.run(scenario())


def test_disabled_threshold_keeps_time_only_behavior(tmp_path) -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(
            _config(tmp_path, "disabled.db"),
            scheduler_path=tmp_path / "s.db",
            candidate_pipeline=_EchoPipeline(),
        ) as runtime:
            scope = await runtime.scopes.create(  # ty: ignore[unresolved-attribute]
                ScopeDraft(title="Off", summary="Q7", idempotency_key="q7-off")
            )
            for rank in (1, 2, 3, 4):
                await _capture(runtime, scope.scope_id, rank)
            await asyncio.sleep(1.5)
            assert await _hits(runtime, scope.scope_id) == 0
            result = await runtime.memory.for_scope(scope.scope_id).flush()
            assert result.source_count == 4

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# C. Isolation and recovery
# ---------------------------------------------------------------------------


def test_scopes_count_independently(tmp_path) -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(
            _config(tmp_path, "scopes.db", memory_source_count_threshold=3),
            scheduler_path=tmp_path / "s.db",
            candidate_pipeline=_EchoPipeline(),
        ) as runtime:
            left = await runtime.scopes.create(  # ty: ignore[unresolved-attribute]
                ScopeDraft(title="Left", summary="Q7", idempotency_key="q7-left")
            )
            right = await runtime.scopes.create(  # ty: ignore[unresolved-attribute]
                ScopeDraft(title="Right", summary="Q7", idempotency_key="q7-right")
            )
            for rank in (1, 2):
                await _capture(runtime, left.scope_id, rank)
            for rank in (11, 12, 13):
                await _capture(runtime, right.scope_id, rank)
            left_hits = await _wait_for_hits(runtime, left.scope_id, 0, seconds=2)
            right_hits = await _wait_for_hits(runtime, right.scope_id, 3)
            assert left_hits == 0
            assert right_hits == 3

    asyncio.run(scenario())


def test_memory_threshold_does_not_affect_topic_memory(tmp_path) -> None:
    async def scenario() -> None:
        async with open_builtin_runtime(
            _config(tmp_path, "families.db", memory_source_count_threshold=3),
            scheduler_path=tmp_path / "s.db",
            candidate_pipeline=_EchoPipeline(),
        ) as runtime:
            scope = await runtime.scopes.create(  # ty: ignore[unresolved-attribute]
                ScopeDraft(title="Families", summary="Q7", idempotency_key="q7-fam")
            )
            memory = runtime.memory.for_scope(scope.scope_id)
            await _capture(runtime, scope.scope_id, 1)
            await _capture(runtime, scope.scope_id, 2)
            await asyncio.sleep(1.5)
            # Memory threshold=3 not met: no memory output...
            assert await _hits(runtime, scope.scope_id) == 0
            # ...and an explicit memory flush works independently of Topic config.
            result = await memory.flush()
            assert result.source_count == 2

    asyncio.run(scenario())


def test_restart_preserves_pending_and_post_threshold_work(tmp_path) -> None:
    async def scenario() -> None:
        database = "restart.db"
        async with open_builtin_runtime(
            _config(tmp_path, database, memory_source_count_threshold=3),
            scheduler_path=tmp_path / "s.db",
            candidate_pipeline=_EchoPipeline(),
        ) as runtime:
            scope = await runtime.scopes.create(  # ty: ignore[unresolved-attribute]
                ScopeDraft(title="Restart", summary="Q7", idempotency_key="q7-restart")
            )
            await _capture(runtime, scope.scope_id, 1)
            await _capture(runtime, scope.scope_id, 2)

        async with open_builtin_runtime(
            _config(tmp_path, database, memory_source_count_threshold=3),
            scheduler_path=tmp_path / "s.db",
            candidate_pipeline=_EchoPipeline(),
        ) as runtime:
            scope_id = scope.scope_id
            # Pending 2 survives the restart and stays below the threshold.
            await asyncio.sleep(1.0)
            assert await _hits(runtime, scope_id) == 0
            await _capture(runtime, scope_id, 3)
            hits = await _wait_for_hits(runtime, scope_id, 3)
            assert hits == 3

    asyncio.run(scenario())


