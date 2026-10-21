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

"""Local acceptance mirror for challenge ``context-02-incremental-pack``."""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json

import pytest

from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    PrepareContextRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.context_pack import (
    CONTEXT_PACK_DELTA_SCHEMA,
    CONTEXT_PACK_SCHEMA,
    DEFAULT_CHUNK_BYTES,
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)
from powercontext.builtin.scope import ScopeDraft


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _expected_digest(items: tuple[bytes, ...]) -> str:
    payload = json.dumps(
        {"items": [_b64(i) for i in items], "schema": CONTEXT_PACK_SCHEMA},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expected_wire(items: tuple[bytes, ...]) -> bytes:
    encoded = [_b64(i) for i in items]
    digest = _expected_digest(items)
    n = 0
    for _ in range(10):
        body = json.dumps(
            {"byte_count": n, "digest": digest, "items": encoded, "schema": CONTEXT_PACK_SCHEMA},
            sort_keys=True, separators=(",", ":"),
        )
        actual = len(body.encode("utf-8"))
        if actual == n:
            return body.encode("utf-8")
        n = actual
    raise AssertionError("reference wire failed to converge")  # noqa: TRY003


# ---------------------------------------------------------------------------
# A. Wire protocol and digest
# ---------------------------------------------------------------------------


def test_wire_bytes_match_the_contract_exactly() -> None:
    pack = ContextPack.build((b"alpha", b"b"))

    wire = pack.to_bytes()

    assert wire == _expected_wire((b"alpha", b"b"))
    body = json.loads(wire)
    assert set(body) == {"byte_count", "digest", "items", "schema"}
    assert body["schema"] == CONTEXT_PACK_SCHEMA
    assert body["items"] == [_b64(b"alpha"), _b64(b"b")]
    assert body["byte_count"] == len(wire)
    assert pack.digest == _expected_digest((b"alpha", b"b"))


def test_round_trip_preserves_the_pack() -> None:
    pack = ContextPack.from_content("第一行\nsecond line 🚀\n")

    restored = ContextPack.from_bytes(pack.to_bytes())

    assert restored == pack


def test_tampered_wire_byte_is_rejected() -> None:
    pack = ContextPack.from_content("tamper target\n" * 20)
    raw = bytearray(pack.to_bytes())
    raw[-30] ^= 0x20

    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(bytes(raw))


@pytest.mark.parametrize(
    "mutation",
    [
        {"drop": "digest"},
        {"drop": "items"},
        {"drop": "schema"},
        {"drop": "byte_count"},
        {"add": {"extra": 1}},
        {"replace": {"schema": "powercontext.context-pack.v2"}},
        {"replace": {"digest": "sha256:" + "0" * 64}},
        {"replace": {"byte_count": 12345}},
        {"replace": {"byte_count": True}},
        {"replace": {"items": ["!!!not-base64!!!"]}},
        {"replace": {"items": "not-a-list"}},
    ],
)
def test_tampered_pack_fields_raise_context_pack_error(mutation: dict[str, object]) -> None:
    pack = ContextPack.from_content("tamper suite\n" * 8)
    body: dict[str, object] = json.loads(pack.to_bytes())
    if "drop" in mutation:
        drop_key = mutation["drop"]
        assert isinstance(drop_key, str)
        body.pop(drop_key)
    if "add" in mutation:
        add_fields = mutation["add"]
        assert isinstance(add_fields, dict)
        for key, value in add_fields.items():
            body[str(key)] = value
    if "replace" in mutation:
        replace_fields = mutation["replace"]
        assert isinstance(replace_fields, dict)
        for key, value in replace_fields.items():
            body[str(key)] = value

    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def test_delta_tampering_raises_context_pack_error() -> None:
    base = ContextPack.from_content("base content\n" * 6)
    target = ContextPack.from_content("base content\n" * 6 + "appended tail\n")
    delta = ContextPackDelta.between(base, target)
    body: dict[str, object] = json.loads(delta.to_bytes())

    for mutation in (
        {"base_digest": "sha256:" + "zz" * 32},
        {"target_digest": "deadbeef"},
        {"prefix_items": -1},
        {"prefix_items": 1.5},
        {"prefix_items": True},
        {"schema": CONTEXT_PACK_DELTA_SCHEMA + "x"},
        {"appended": ["!!!"]},
        {"extra": True},
    ):
        tampered = dict(body)
        tampered.update(mutation)
        with pytest.raises(ContextPackError):
            ContextPackDelta.from_bytes(json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def test_byte_count_survives_digit_count_growth() -> None:
    # Sizes chosen around the 3-to-4 digit boundary observed in the wire prototype.
    for size in (300, 620, 631, 632, 640, 6100, 6200):
        pack = ContextPack.build((b"x" * size,))
        wire = pack.to_bytes()
        assert json.loads(wire)["byte_count"] == len(wire)
        assert ContextPack.from_bytes(wire) == pack


def test_empty_pack_wire_is_well_formed() -> None:
    pack = ContextPack.build(())

    wire = pack.to_bytes()

    body = json.loads(wire)
    assert body["items"] == []
    assert body["byte_count"] == len(wire)
    assert ContextPack.from_bytes(wire) == pack


# ---------------------------------------------------------------------------
# B. Chunking
# ---------------------------------------------------------------------------


def test_empty_content_makes_an_empty_pack() -> None:
    for content in ("", None):  # type: ignore[list-item]
        pack = ContextPack.from_content(content)

        assert pack.items == ()
        assert pack.digest == _expected_digest(())
        assert ContextPack.from_bytes(pack.to_bytes()) == pack
        assert b"".join(pack.items) == b""


def test_exactly_512_bytes_of_chinese_text_round_trips() -> None:
    # '汉' is 3 bytes (170 x 3 = 510) plus one 2-byte char = 512 total.
    content = "汉" * 170 + "±"
    assert len(content.encode("utf-8")) == 512

    pack = ContextPack.from_content(content)

    assert len(pack.items) == 1
    assert pack.items[0] == content.encode("utf-8")
    assert b"".join(pack.items).decode("utf-8") == content


@pytest.mark.parametrize("size", [1, 499, 500, 501, 511, 512, 513, 1023, 1024, 1025])
def test_byte_sizes_around_the_chunk_boundary_round_trip(size: int) -> None:
    text = ""
    raw = b""
    index = 0
    while True:
        candidate = ("字" if index % 2 == 0 else "±") + str(index % 10)
        candidate_raw = candidate.encode("utf-8")
        if raw and len(raw) + len(candidate_raw) > size:
            break
        text += candidate
        raw += candidate_raw
        index += 1
        if len(raw) >= size:
            break

    pack = ContextPack.from_content(text)

    assert b"".join(pack.items).decode("utf-8") == text
    for item in pack.items:
        assert len(item) <= DEFAULT_CHUNK_BYTES
        item.decode("utf-8")


def test_multibyte_split_never_lands_inside_a_character() -> None:
    # Each '汉' is 3 bytes; force chunks that would split a character mid-sequence.
    content = "汉" * 500

    pack = ContextPack.from_content(content)

    assert b"".join(pack.items).decode("utf-8") == content
    assert all(len(item) <= DEFAULT_CHUNK_BYTES for item in pack.items)


def test_single_character_larger_than_chunk_limit_is_rejected() -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("汉", chunk_bytes=2)


def test_invalid_chunk_limit_is_rejected() -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("hello", chunk_bytes=0)


def test_line_aligned_chunking_keeps_lines_whole() -> None:
    lines = [f"line {index}: " + "字" * 40 + "\n" for index in range(20)]
    content = "".join(lines)

    pack = ContextPack.from_content(content)

    assert b"".join(pack.items).decode("utf-8") == content
    for item in pack.items[:-1]:
        assert item.endswith(b"\n")


def test_chunking_is_deterministic() -> None:
    content = "determinism check\n" * 50

    left = ContextPack.from_content(content)
    right = ContextPack.from_content(content)

    assert left.items == right.items
    assert left.digest == right.digest


# ---------------------------------------------------------------------------
# C. Delta and apply
# ---------------------------------------------------------------------------


def test_identical_packs_yield_full_prefix_and_empty_append() -> None:
    base = ContextPack.from_content("same\n" * 30)
    target = ContextPack.from_content("same\n" * 30)

    delta = ContextPackDelta.between(base, target)

    assert delta.prefix_items == len(base.items)
    assert delta.appended == ()
    assert apply_context_pack_delta(base, delta) == target


def test_tail_append_keeps_every_base_chunk() -> None:
    head = "".join(f"head line {index:04d}: {'x' * 60}\n" for index in range(20))
    base = ContextPack.from_content(head)
    target = ContextPack.from_content(head + "tail\n" * 3)

    delta = ContextPackDelta.between(base, target)

    # The final base chunk grows to absorb the tail lines, so only it is re-sent.
    assert delta.prefix_items == len(base.items) - 1
    assert len(delta.appended) > 0
    assert apply_context_pack_delta(base, delta) == target
    assert b"".join(delta.appended).decode("utf-8").startswith("head line 0018")


def test_middle_change_stops_the_prefix_at_the_first_difference() -> None:
    base = ContextPack.from_content("common\n" + "old middle\n" + "shared tail\n" * 5)
    target = ContextPack.from_content("common\n" + "new middle\n" + "shared tail\n" * 5)

    delta = ContextPackDelta.between(base, target)
    rebuilt = apply_context_pack_delta(base, delta)

    assert rebuilt == target
    assert delta.prefix_items < len(base.items)


def test_head_change_uses_a_zero_prefix() -> None:
    base = ContextPack.from_content("old head\n" + "body\n" * 5)
    target = ContextPack.from_content("new head\n" + "body\n" * 5)

    delta = ContextPackDelta.between(base, target)

    assert delta.prefix_items == 0
    assert apply_context_pack_delta(base, delta) == target


def test_apply_rejects_wrong_base_digest_prefix_and_target() -> None:
    base = ContextPack.from_content("base\n" * 10)
    target = ContextPack.from_content("base\n" * 10 + "more\n")
    delta = ContextPackDelta.between(base, target)

    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(
            base_digest="sha256:" + "8" * 64,
            target_digest=delta.target_digest,
            prefix_items=delta.prefix_items,
            appended=delta.appended,
        ))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(
            base_digest=delta.base_digest,
            target_digest=delta.target_digest,
            prefix_items=len(base.items) + 1,
            appended=delta.appended,
        ))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(
            base_digest=delta.base_digest,
            target_digest="sha256:" + "7" * 64,
            prefix_items=delta.prefix_items,
            appended=delta.appended,
        ))


def test_delta_survives_serialization_before_apply() -> None:
    base = ContextPack.from_content("wire boundary\n" * 12)
    target = ContextPack.from_content("wire boundary\n" * 12 + "new tail\n")

    delta = ContextPackDelta.between(base, target)
    transported = ContextPackDelta.from_bytes(delta.to_bytes())
    base_only = ContextPack.from_bytes(base.to_bytes())

    assert transported == delta
    assert apply_context_pack_delta(base_only, transported) == target


def test_base_pack_type_is_validated() -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(b"not json")
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(b"[]")
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(b"not json")
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(b"{}")


# ---------------------------------------------------------------------------
# D. prepare_incremental integration on a real runtime
# ---------------------------------------------------------------------------


class _EchoPipeline:
    async def extract(self, request, /):
        from powercontext.builtin.artifacts.memory.models import MemoryEntryInput

        return tuple(
            MemoryEntryInput(kind="fact", text=source.content, sources=(source,))
            for source in request.sources
        )


def _runtime(tmp_path, name: str):
    return open_builtin_runtime(
        BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / name}")),
        scheduler_path=tmp_path / "scheduler.db",
        candidate_pipeline=_EchoPipeline(),
    )


def test_prepare_incremental_full_chain(tmp_path) -> None:
    async def scenario() -> None:
        async with _runtime(tmp_path, "incremental.db") as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Incremental", summary="Q6", idempotency_key="q6-scope")
            )
            context = runtime.context.for_scope(scope.scope_id)
            memory = runtime.memory.for_scope(scope.scope_id)
            request = PrepareContextRequest(query="calibration")

            # Round 1: first prepare without a base returns a full pack.
            first_pack = await context.prepare_incremental(request)
            prepared = await context.prepare(request)
            assert isinstance(first_pack, ContextPack)
            assert b"".join(first_pack.items).decode("utf-8") == (prepared.content or "")
            if prepared.status == "ready":
                assert first_pack.items

            # Round 2: memory append changes the content; delta rebuilds it.
            for rank in range(1, 4):
                await runtime.sources.for_scope(scope.scope_id).capture(
                    CaptureSource(
                        source_id=f"turn-{rank}",
                        content=f"calibration constant {rank} is {rank * 11}",
                        metadata={},
                    )
                )
            await memory.flush()
            second_pack = await context.prepare_incremental(request)
            delta = await context.prepare_incremental(request, base_pack=first_pack)
            assert isinstance(delta, ContextPackDelta)
            rebuilt = apply_context_pack_delta(first_pack, delta)
            direct = await context.prepare(request)
            assert b"".join(rebuilt.items).decode("utf-8") == (direct.content or "")
            del second_pack

            # Round 3: query switch (content changes in the middle).
            switched_request = PrepareContextRequest(query="completely different topic")
            switched = await context.prepare_incremental(switched_request, base_pack=rebuilt)
            assert isinstance(switched, ContextPackDelta)
            target = await context.prepare(switched_request)
            rebuilt_two = apply_context_pack_delta(rebuilt, switched)
            assert b"".join(rebuilt_two.items).decode("utf-8") == (target.content or "")

    asyncio.run(scenario())


def test_prepare_incremental_empty_results(tmp_path) -> None:
    async def scenario() -> None:
        async with _runtime(tmp_path, "empty.db") as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Empty", summary="Q6 empty", idempotency_key="q6-empty")
            )
            context = runtime.context.for_scope(scope.scope_id)
            request = PrepareContextRequest(query="nothing matches here")

            pack = await context.prepare_incremental(request)
            prepared = await context.prepare(request)
            assert prepared.status == "empty"
            assert pack.items == ()
            assert ContextPack.from_bytes(pack.to_bytes()) == pack

            delta = await context.prepare_incremental(request, base_pack=pack)
            assert isinstance(delta, ContextPackDelta)
            rebuilt = apply_context_pack_delta(pack, delta)
            assert b"".join(rebuilt.items) == b""

    asyncio.run(scenario())


def test_prepare_incremental_rejects_invalid_base_pack(tmp_path) -> None:
    async def scenario() -> None:
        async with _runtime(tmp_path, "badbase.db") as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Bad", summary="Q6", idempotency_key="q6-bad")
            )
            context = runtime.context.for_scope(scope.scope_id)
            with pytest.raises(ContextPackError):
                await context.prepare_incremental(
                    PrepareContextRequest(query="q"), "not-a-pack"
                )

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# E. 40-round cost measurement
# ---------------------------------------------------------------------------


def _run_cost_sequence(database_path, *, append_every: int, switch_every: int, chunk_probe: bool = False):
    """Run one 40-round cost sequence; return (saving, distinct, raw, wire)."""

    async def scenario():
        async with _runtime(database_path, "cost.db") as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Cost", summary="Q6 cost", idempotency_key="q6-cost")
            )
            context = runtime.context.for_scope(scope.scope_id)
            memory = runtime.memory.for_scope(scope.scope_id)
            queries = [
                "migration plan",
                "migration decisions",
                "migration plan owners",
                "migration",
            ]
            for rank in range(1, 9):
                await runtime.sources.for_scope(scope.scope_id).capture(
                    CaptureSource(
                        source_id=f"turn-{rank}",
                        content=f"Meeting turn {rank}: decision {rank} on the migration plan.",
                        metadata={},
                    )
                )
            await memory.flush()
            request = PrepareContextRequest(query=queries[0])
            base_pack = await context.prepare_incremental(request)
            assert isinstance(base_pack, ContextPack)
            if base_pack.items == ():
                pytest.fail("cost sequence requires a non-empty first context")
            raw_total = len(b"".join(base_pack.items))
            wire_total = len(base_pack.to_bytes())
            distinct = {b"".join(base_pack.items)}
            for round_index in range(2, 41):
                if round_index % append_every == 0:
                    await runtime.sources.for_scope(scope.scope_id).capture(
                        CaptureSource(
                            source_id=f"turn-{round_index + 8}",
                            content=(
                                f"Meeting turn {round_index + 8}: follow-up {round_index} recorded "
                                f"for the migration plan with owner {round_index}."
                            ),
                            metadata={},
                        )
                    )
                    await memory.flush()
                query = queries[(round_index // switch_every) % len(queries)]
                target = await context.prepare(PrepareContextRequest(query=query))
                content = target.content or ""
                if not content:
                    continue
                target_pack = ContextPack.from_content(content)
                delta = ContextPackDelta.between(base_pack, target_pack)
                rebuilt = apply_context_pack_delta(base_pack, delta)
                assert b"".join(rebuilt.items).decode("utf-8") == content
                raw_total += len(content.encode("utf-8"))
                wire_total += len(delta.to_bytes())
                distinct.add(content.encode("utf-8"))
                base_pack = rebuilt
            return 1 - wire_total / raw_total, len(distinct), raw_total, wire_total

    return asyncio.run(scenario())


def test_40_round_incremental_savings(tmp_path, capsys) -> None:
    """Official gate on a moderate sequence: appends every 3rd round, query
    switches every 8th. Report harsher mixes as observations."""

    saving, distinct, raw_total, wire_total = _run_cost_sequence(
        tmp_path / "moderate.db", append_every=3, switch_every=8
    )
    print(
        f"MODERATE append/3 switch/8: raw={raw_total} wire={wire_total} "
        f"saving={saving:.4f} distinct={distinct}"
    )
    assert distinct >= 4
    assert saving >= 0.45, f"net transport saving {saving:.4f} below the official 0.45 gate"


def test_40_round_savings_under_harsher_churn(tmp_path, capsys) -> None:
    """Observation gates: rebuild correctness must hold even when the saving
    degrades under churn; the numbers are reported honestly."""

    saving_fast_append, distinct_a, raw_a, wire_a = _run_cost_sequence(
        tmp_path / "fast-append.db", append_every=2, switch_every=1
    )
    print(
        f"HARSH append/2 switch/1: raw={raw_a} wire={wire_a} "
        f"saving={saving_fast_append:.4f} distinct={distinct_a}"
    )
    saving_full_churn, distinct_b, raw_b, wire_b = _run_cost_sequence(
        tmp_path / "full-churn.db", append_every=1, switch_every=1
    )
    print(
        f"EXTREME append/1 switch/1: raw={raw_b} wire={wire_b} "
        f"saving={saving_full_churn:.4f} distinct={distinct_b}"
    )
    assert distinct_a >= 4 and distinct_b >= 4
    # Correctness is the hard requirement under churn; savings are reported.
    assert saving_full_churn < saving_fast_append


def test_gzip_reference_observation(tmp_path, capsys) -> None:
    async def scenario() -> None:
        async with _runtime(tmp_path, "gzipref.db") as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Gzip", summary="Q6 gzip", idempotency_key="q6-gzip")
            )
            context = runtime.context.for_scope(scope.scope_id)
            memory = runtime.memory.for_scope(scope.scope_id)
            for rank in range(1, 9):
                await runtime.sources.for_scope(scope.scope_id).capture(
                    CaptureSource(
                        source_id=f"turn-{rank}",
                        content=f"Meeting turn {rank}: decision {rank} recorded.",
                        metadata={},
                    )
                )
            await memory.flush()
            raw_total = 0
            gzip_total = 0
            for index, query in enumerate(["decision", "turn", "recorded", "meeting"] * 3):
                prepared = await context.prepare(PrepareContextRequest(query=query))
                content = (prepared.content or "").encode("utf-8")
                raw_total += len(content)
                gzip_total += len(gzip.compress(content, compresslevel=9, mtime=0))
                del index
            print(f"gzip observation: raw={raw_total} gzip={gzip_total} ratio={gzip_total / raw_total:.4f}")

    asyncio.run(scenario())
