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

"""Local acceptance mirror for challenge ``source-02-pi-history``."""

from __future__ import annotations

import asyncio
import copy
import json

import httpx
import pytest

from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CommitConnectorCheckpoint,
    SubmitSourceObservation,
    open_builtin_contexts,
)
from powercontext.client import PowerContextClient, RemoteConnectorWorker
from powercontext.errors import InvalidConnectorRunError
from powercontext.http import CreateScopeRequest
from powercontext.server.factory import create_server_app
from powercontext.server.settings import McpConfig, ServerSettings
from powercontext.sources import (
    ConnectorBinding,
    ConnectorRunResult,
    ConnectorRunStatus,
    SourceDefinitionRegistry,
    SourceRef,
    manifest_for_definition,
)
from powercontext.sources.connectors import ConnectorLifecycle
from powercontext.sources.observations import project_source_for_transport
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_ADAPTER,
    PI_EVENT_SOURCE_DEFINITION,
    PI_EVENT_SOURCE_NAME,
    PiEventInput,
    PiEventSource,
    PiEventTextEvidenceProjection,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
    pi_event_source_id,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

TIMESTAMP = "2026-01-01T00:00:00Z"


def _session(session_id: str = "sess-1", version: int = 3) -> dict:
    return {"type": "session", "version": version, "id": session_id, "timestamp": TIMESTAMP, "cwd": "/repo"}


def _jsonl(*records: dict) -> bytes:
    return ("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n").encode("utf-8")


def _message(event_id: str, parent_id: str | None, role: str, content: object, **extra: object) -> dict:
    return {
        "type": "message",
        "id": event_id,
        "parentId": parent_id,
        "timestamp": TIMESTAMP,
        "message": {"role": role, "content": content, "timestamp": 1, **extra},
    }


def _record(event_type: str, event_id: str, parent_id: str | None, **extra: object) -> dict:
    return {"type": event_type, "id": event_id, "parentId": parent_id, "timestamp": TIMESTAMP, **extra}


def _linear_session() -> bytes:
    return _jsonl(
        _session(),
        _message("u1", None, "user", "hello"),
        _message("a1", "u1", "assistant", [{"type": "text", "text": "hi there"}]),
        _message("u2", "a1", "user", "again"),
    )


def _branched_session() -> bytes:
    return _jsonl(
        _session(),
        _message("u1", None, "user", "hello"),
        _message("a1", "u1", "assistant", [{"type": "text", "text": "hi"}]),
        _message("b1", "a1", "user", "branch A"),
        _message("b2", "b1", "user", "branch A continued"),
        _message("c1", "a1", "user", "branch B"),
    )


# ---------------------------------------------------------------------------
# A. Parsing and event rendering
# ---------------------------------------------------------------------------


def test_parses_linear_session_in_root_to_leaf_order() -> None:
    events = parse_pi_session(PiSessionInput(content=_linear_session()))

    assert [event.event_id for event in events] == ["u1", "a1", "u2"]
    assert [event.role for event in events] == ["user", "assistant", "user"]
    assert events[1].content == "assistant: hi there"
    assert events[0].parent_id is None
    assert events[0].event_type == "message"
    assert events[0].session_id == "sess-1"
    assert events[0].record["type"] == "message"


def test_explicit_leaf_selects_only_its_ancestor_chain() -> None:
    events = parse_pi_session(PiSessionInput(content=_branched_session(), leaf_id="b1"))

    assert [event.event_id for event in events] == ["u1", "a1", "b1"]


def test_default_leaf_is_the_last_record() -> None:
    events = parse_pi_session(PiSessionInput(content=_branched_session()))

    assert [event.event_id for event in events] == ["u1", "a1", "c1"]


def test_switching_leaf_keeps_shared_ancestors_once() -> None:
    branch_a = parse_pi_session(PiSessionInput(content=_branched_session(), leaf_id="b2"))
    branch_b = parse_pi_session(PiSessionInput(content=_branched_session(), leaf_id="c1"))

    assert [event.event_id for event in branch_a] == ["u1", "a1", "b1", "b2"]
    assert [event.event_id for event in branch_b] == ["u1", "a1", "c1"]
    assert [event.event_id for event in branch_a][:2] == [event.event_id for event in branch_b][:2]


def test_control_node_between_messages_keeps_the_chain_connected() -> None:
    content = _jsonl(
        _session(),
        _message("u1", None, "user", "hello"),
        _record("model_change", "m1", "u1", provider="openai", modelId="gpt"),
        _message("a1", "m1", "assistant", [{"type": "text", "text": "hi"}]),
    )

    events = parse_pi_session(PiSessionInput(content=content))

    assert [event.event_id for event in events] == ["u1", "a1"]
    assert events[1].parent_id == "m1"


def test_message_string_and_multiple_text_blocks() -> None:
    content = _jsonl(
        _session(),
        _message("u1", None, "user", "raw string"),
        _message(
            "a1",
            "u1",
            "assistant",
            [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        ),
    )

    events = parse_pi_session(PiSessionInput(content=content))

    assert events[0].content == "user: raw string"
    assert events[1].content == "assistant: first\nsecond"


def test_tool_call_renders_sorted_compact_json_with_unicode() -> None:
    content = _jsonl(
        _session(),
        _message(
            "a1",
            None,
            "assistant",
            [
                {"type": "text", "text": "running"},
                {"type": "toolCall", "name": "bash", "id": "call-1", "arguments": {"z": 1, "a": "北京"}},
            ],
        ),
    )

    event = parse_pi_session(PiSessionInput(content=content))[0]

    assert event.content == 'assistant: running\n[tool_call bash call-1] {"a":"北京","z":1}'


def test_tool_result_rendering() -> None:
    content = _jsonl(
        _session(),
        _message(
            "r1",
            None,
            "toolResult",
            [{"type": "text", "text": "done"}],
            toolName="bash",
            toolCallId="call-1",
        ),
    )

    event = parse_pi_session(PiSessionInput(content=content))[0]

    assert event.role == "toolResult"
    assert event.content == "toolResult: [tool_result bash call-1] done"


def test_bash_execution_rendering() -> None:
    content = _jsonl(
        _session(),
        _message("b1", None, "bashExecution", "", command="ls", output="a\nb"),
    )

    event = parse_pi_session(PiSessionInput(content=content))[0]

    assert event.role == "bashExecution"
    assert event.content == "bashExecution: $ ls\na\nb"


def test_compaction_is_evidence_and_does_not_drop_history() -> None:
    content = _jsonl(
        _session(),
        _message("u1", None, "user", "before"),
        _message("a1", "u1", "assistant", [{"type": "text", "text": "reply"}]),
        _record("compaction", "c1", "a1", summary="compressed", firstKeptEntryId="a1"),
        _message("u2", "c1", "user", "after"),
    )

    events = parse_pi_session(PiSessionInput(content=content))

    assert [event.event_id for event in events] == ["u1", "a1", "c1", "u2"]
    assert events[2].event_type == "compaction"
    assert events[2].role == "compaction"
    assert events[2].content == "compaction: compressed"


def test_branch_summary_rendering() -> None:
    content = _jsonl(
        _session(),
        _message("u1", None, "user", "hello"),
        _record("branch_summary", "s1", "u1", fromId="u1", summary="branch notes"),
    )

    event = parse_pi_session(PiSessionInput(content=content))[1]

    assert event.event_type == "branch_summary"
    assert event.content == "branch_summary: branch notes"


def test_custom_message_rendering() -> None:
    content = _jsonl(
        _session(),
        _record("custom_message", "m1", None, customType="note", content=[{"type": "text", "text": "hello"}]),
    )

    event = parse_pi_session(PiSessionInput(content=content))[0]

    assert event.role == "custom"
    assert event.content == "custom: hello"


def test_system_thinking_and_image_blocks_produce_no_evidence() -> None:
    content = _jsonl(
        _session(),
        _message("s1", None, "system", "system prompt"),
        _message(
            "a1",
            "s1",
            "assistant",
            [{"type": "thinking", "thinking": "private"}, {"type": "image", "data": "..."}],
        ),
        _message("u1", "a1", "user", "visible"),
    )

    events = parse_pi_session(PiSessionInput(content=content))

    assert [event.event_id for event in events] == ["u1"]
    assert events[0].parent_id == "a1"


def test_unknown_control_types_produce_no_evidence() -> None:
    content = _jsonl(
        _session(),
        _record("usage", "x1", None, kind="turn", usage={}),
        _message("u1", "x1", "user", "visible"),
    )

    events = parse_pi_session(PiSessionInput(content=content))

    assert [event.event_id for event in events] == ["u1"]


@pytest.mark.parametrize(
    "content",
    [
        b"",
        _jsonl({"type": "message", "id": "u1", "parentId": None, "message": {"role": "user", "content": "x"}}),
        _jsonl(_session(version=2), _message("u1", None, "user", "x")),
        _jsonl({**_session(), "id": ""}, _message("u1", None, "user", "x")),
        _jsonl(_session(), _message("", None, "user", "x")),
        _jsonl(_session(), _message("u1", "missing", "user", "x")),
        _jsonl(_session(), _message("u1", None, "user", "x"), _message("u1", None, "user", "y")),
        _jsonl(_session(), {"type": "message", "id": "u1", "message": {"role": "user", "content": "x"}}),
        b'{"type":"session","version":3,"id":"s"}\nnot-json\n',
        b'{"type":"session","version":3,"id":"s"}\n[]\n',
        b'{"type":"session","version":3,"id":"s"}\n\xff\n',
    ],
)
def test_invalid_documents_raise_value_error(content: bytes) -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=content))


def test_missing_explicit_leaf_raises_value_error() -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=_linear_session(), leaf_id="missing"))


def test_header_only_document_raises_value_error() -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=_jsonl(_session())))


def test_forward_parent_reference_raises_value_error() -> None:
    content = _jsonl(
        _session(),
        _message("a1", None, "user", "root"),
        _message("a2", "a3", "user", "forward reference"),
        _message("a3", "a1", "user", "later"),
    )

    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=content))


def test_utf8_bom_is_accepted() -> None:
    content = b"\xef\xbb\xbf" + _linear_session()

    events = parse_pi_session(PiSessionInput(content=content))

    assert [event.event_id for event in events] == ["u1", "a1", "u2"]


def test_errors_do_not_return_partial_results() -> None:
    content = _jsonl(
        _session(),
        _message("u1", None, "user", "valid"),
        _message("a1", "u1", "assistant", [{"type": "text", "text": "valid"}]),
        _message("u1", "a1", "user", "duplicate id invalidates the whole document"),
    )

    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=content))


# ---------------------------------------------------------------------------
# B. Identity, projection, read
# ---------------------------------------------------------------------------


def test_identity_format_and_percent_encoding() -> None:
    assert pi_event_source_id("sess-1", "a1") == "pi:sess-1:a1"
    assert pi_event_source_id("a b/c:d", "e f") == "pi:a%20b%2Fc%3Ad:e%20f"
    assert pi_event_source_id("会话", "事件") == "pi:%E4%BC%9A%E8%AF%9D:%E4%BA%8B%E4%BB%B6"


def test_identity_is_stable_across_branches() -> None:
    branch_a = parse_pi_session(PiSessionInput(content=_branched_session(), leaf_id="b2"))
    branch_b = parse_pi_session(PiSessionInput(content=_branched_session(), leaf_id="c1"))

    shared_a = {event.event_id: pi_event_source_id(event.session_id, event.event_id) for event in branch_a[:2]}
    shared_b = {event.event_id: pi_event_source_id(event.session_id, event.event_id) for event in branch_b[:2]}
    assert shared_a == shared_b


def test_identity_rejects_values_over_the_length_limit() -> None:
    with pytest.raises(ValueError):
        pi_event_source_id("s" * 300, "e1")


def test_resolve_and_read_round_trip() -> None:
    event = parse_pi_session(PiSessionInput(content=_linear_session()))[0]

    async def scenario() -> None:
        source = await PI_EVENT_SOURCE_ADAPTER.resolve(event)
        assert type(source) is PiEventSource
        assert source.name == source.source_id == "pi:sess-1:u1"
        assert source.materialization.value == "captured"
        assert source.definition_version == "1"
        assert await PI_EVENT_SOURCE_ADAPTER.read(source) == event

    asyncio.run(scenario())


def test_projection_has_standard_shape_and_five_metadata_keys() -> None:
    event = parse_pi_session(PiSessionInput(content=_linear_session()))[0]
    source = asyncio.run(PI_EVENT_SOURCE_ADAPTER.resolve(event))

    evidence = PiEventTextEvidenceProjection().project(source)

    assert evidence.source_type == PI_EVENT_SOURCE_NAME
    assert evidence.source_id == "pi:sess-1:u1"
    assert evidence.content == "user: hello"
    assert set(evidence.metadata) == {"session_id", "event_id", "parent_id", "event_type", "role"}
    assert evidence.metadata == {
        "session_id": "sess-1",
        "event_id": "u1",
        "parent_id": None,
        "event_type": "message",
        "role": "user",
    }


def test_manifest_declares_standard_text_evidence_projection() -> None:
    manifest = manifest_for_definition(PI_EVENT_SOURCE_DEFINITION)

    assert manifest.name == "pi-event"
    assert manifest.version == "1"
    assert manifest.fingerprint.startswith("sha256:")
    projection = manifest.projections[0]
    assert projection.schema_ == TextEvidence.model_json_schema()


def test_definition_resolves_through_a_real_registry() -> None:
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    event = parse_pi_session(PiSessionInput(content=_linear_session()))[0]

    source = asyncio.run(registry.resolve(event))

    assert isinstance(source, PiEventSource)
    assert asyncio.run(registry.read(source)) == event


# ---------------------------------------------------------------------------
# C. Connector lifecycle: incremental import, branches, restart, rewrites
# ---------------------------------------------------------------------------


class _RecordingSink:
    def __init__(self) -> None:
        self.submitted: list[tuple[str, str, PiEventInput]] = []

    async def submit(self, binding: ConnectorBinding, item_id: str, definition_name: str, value: object, /) -> SourceRef:
        del binding
        assert isinstance(value, PiEventInput)
        self.submitted.append((item_id, definition_name, value))
        return SourceRef(
            source_type=definition_name,
            source_id=pi_event_source_id(value.session_id, value.event_id),
        )


class _MemoryCheckpointStore:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.saves = 0

    async def load(self, binding: ConnectorBinding, /) -> object:
        return copy.deepcopy(self.values.get(binding.binding_id))

    async def save(self, binding: ConnectorBinding, checkpoint: object, /, *, expected: object) -> None:
        if self.values.get(binding.binding_id) != expected:
            raise AssertionError("checkpoint CAS mismatch")  # noqa: TRY003
        self.values[binding.binding_id] = copy.deepcopy(checkpoint)
        self.saves += 1


class _ExplodingSink:
    async def submit(self, binding: ConnectorBinding, item_id: str, definition_name: str, value: object, /) -> SourceRef:
        del binding, item_id, definition_name, value
        raise RuntimeError("sink unavailable")  # noqa: TRY003


def _binding() -> ConnectorBinding:
    return ConnectorBinding(
        scope_id="scope-a",
        binding_id="pi-1",
        connector_name="pi-session-history",
        connector_version="1",
    )


def _run(
    connector: PiSessionConnector,
    sink: object,
    store: _MemoryCheckpointStore,
    binding: ConnectorBinding | None = None,
) -> ConnectorRunResult:
    return asyncio.run(
        ConnectorLifecycle(sink=sink, checkpoints=store).run(connector, binding or _binding())  # ty: ignore[invalid-argument-type]
    )


def test_connector_imports_every_visible_event_once(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(_linear_session())
    sink, store = _RecordingSink(), _MemoryCheckpointStore()

    result = _run(PiSessionConnector(path), sink, store)

    assert result.status == "complete"
    assert [item[0] for item in sink.submitted] == ["u1", "a1", "u2"]
    assert all(item[1] == PI_EVENT_SOURCE_NAME for item in sink.submitted)
    assert result.committed_checkpoint == result.proposed_checkpoint
    assert result.committed_checkpoint["leaf_id"] == "u2"
    assert store.saves == 1


def test_rerun_without_changes_submits_nothing(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(_linear_session())
    sink, store = _RecordingSink(), _MemoryCheckpointStore()
    _run(PiSessionConnector(path), sink, store)

    result = _run(PiSessionConnector(path), sink, store)

    assert len(sink.submitted) == 3
    assert result.items == ()
    assert result.committed_checkpoint == result.previous_checkpoint
    assert store.saves == 1


def test_appended_events_are_the_only_new_submissions(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(_linear_session())
    sink, store = _RecordingSink(), _MemoryCheckpointStore()
    _run(PiSessionConnector(path), sink, store)
    path.write_bytes(
        _jsonl(
            _session(),
            _message("u1", None, "user", "hello"),
            _message("a1", "u1", "assistant", [{"type": "text", "text": "hi there"}]),
            _message("u2", "a1", "user", "again"),
            _message("a2", "u2", "assistant", [{"type": "text", "text": "appended"}]),
        )
    )

    result = _run(PiSessionConnector(path), sink, store)

    assert [item[0] for item in sink.submitted[3:]] == ["a2"]
    assert result.committed_checkpoint["leaf_id"] == "a2"


def test_switching_branch_skips_shared_ancestors_and_keeps_old_events(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(_branched_session())
    sink, store = _RecordingSink(), _MemoryCheckpointStore()
    first = _run(PiSessionConnector(path, leaf_id="b2"), sink, store)

    second = _run(PiSessionConnector(path, leaf_id="c1"), sink, store)

    assert [item[0] for item in sink.submitted] == ["u1", "a1", "b1", "b2", "c1"]
    checkpoint = second.committed_checkpoint
    assert checkpoint["leaf_id"] == "c1"
    assert {"b1", "b2", "c1"} <= set(checkpoint["events"])
    assert first.committed_checkpoint["events"]["b1"] == checkpoint["events"]["b1"]


def test_restart_resumes_from_the_persisted_checkpoint(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(_linear_session())
    store = _MemoryCheckpointStore()
    _run(PiSessionConnector(path), _RecordingSink(), store)

    restarted_sink = _RecordingSink()
    result = _run(PiSessionConnector(path), restarted_sink, store)

    assert restarted_sink.submitted == []
    assert result.committed_checkpoint == result.previous_checkpoint


def test_rewritten_imported_event_raises_and_keeps_checkpoint(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(_linear_session())
    sink, store = _RecordingSink(), _MemoryCheckpointStore()
    _run(PiSessionConnector(path), sink, store)
    before = copy.deepcopy(store.values["pi-1"])
    path.write_bytes(
        _jsonl(
            _session(),
            _message("u1", None, "user", "changed"),
            _message("a1", "u1", "assistant", [{"type": "text", "text": "hi there"}]),
            _message("u2", "a1", "user", "again"),
        )
    )

    with pytest.raises(ValueError):
        _run(PiSessionConnector(path), sink, store)

    assert store.values["pi-1"] == before
    assert len(sink.submitted) == 3


def test_corrupt_input_raises_without_advancing_checkpoint(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(_linear_session())
    sink, store = _RecordingSink(), _MemoryCheckpointStore()
    _run(PiSessionConnector(path), sink, store)
    before = copy.deepcopy(store.values["pi-1"])
    path.write_bytes(b"not json\n")

    with pytest.raises(ValueError):
        _run(PiSessionConnector(path), sink, store)

    assert store.values["pi-1"] == before


def test_corrupt_checkpoint_raises_before_submitting(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(_linear_session())
    sink, store = _RecordingSink(), _MemoryCheckpointStore()
    store.values["pi-1"] = {"schema_version": "1"}

    with pytest.raises(InvalidConnectorRunError):
        _run(PiSessionConnector(path), sink, store)

    assert sink.submitted == []


def test_sink_failure_aborts_without_committing_checkpoint(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(_linear_session())
    store = _MemoryCheckpointStore()

    with pytest.raises(RuntimeError):
        _run(PiSessionConnector(path), _ExplodingSink(), store)

    assert store.values == {}
    assert store.saves == 0


# ---------------------------------------------------------------------------
# D. Real relational contexts: observations, checkpoint CAS, restart
# ---------------------------------------------------------------------------


class _ContextSink:
    def __init__(self, contexts: object, registry: SourceDefinitionRegistry) -> None:
        self._contexts = contexts
        self._registry = registry

    async def submit(self, binding: ConnectorBinding, item_id: str, definition_name: str, value: object, /) -> SourceRef:
        del item_id
        assert definition_name == PI_EVENT_SOURCE_NAME
        source = await self._registry.resolve(value)
        observation = project_source_for_transport(self._registry, source)
        receipt = await self._contexts.submit_source_observation(  # ty: ignore[possibly-missing-attribute]
            SubmitSourceObservation(scope_id=binding.scope_id, observation=observation)
        )
        return receipt.source_ref


class _ContextCheckpointStore:
    def __init__(self, contexts: object) -> None:
        self._contexts = contexts

    async def load(self, binding: ConnectorBinding, /) -> object:
        return (await self._contexts.connector_checkpoint(binding)).checkpoint  # ty: ignore[possibly-missing-attribute]

    async def save(self, binding: ConnectorBinding, checkpoint: object, /, *, expected: object) -> None:
        await self._contexts.commit_connector_checkpoint(  # ty: ignore[possibly-missing-attribute]
            CommitConnectorCheckpoint(binding=binding, expected=expected, checkpoint=checkpoint)
        )


def _sqlite(tmp_path, name: str = "pi.db") -> SQLiteConfig:
    return SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / name}")


def test_contexts_ingestion_persists_events_and_checkpoint(tmp_path) -> None:
    async def scenario() -> None:
        registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
        binding = _binding()
        path = tmp_path / "session.jsonl"
        path.write_bytes(_linear_session())
        async with open_builtin_contexts(BuiltinConfig(database=_sqlite(tmp_path))) as contexts:
            await contexts.register_source_definition(manifest_for_definition(PI_EVENT_SOURCE_DEFINITION))
            result = await ConnectorLifecycle(
                sink=_ContextSink(contexts, registry),
                checkpoints=_ContextCheckpointStore(contexts),
            ).run(PiSessionConnector(path), binding)
            assert result.status == "complete"

            context = await contexts.get(binding.scope_id)
            stored = await context.sources.list()
            assert [source.name for source in stored] == ["pi:sess-1:u1", "pi:sess-1:a1", "pi:sess-1:u2"]
            checkpoint = (await contexts.connector_checkpoint(binding)).checkpoint
            assert checkpoint["leaf_id"] == "u2"
            assert set(checkpoint["events"]) == {"u1", "a1", "u2"}

    asyncio.run(scenario())


def test_contexts_restart_keeps_checkpoint_and_does_not_duplicate(tmp_path) -> None:
    async def scenario() -> None:
        registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
        binding = _binding()
        path = tmp_path / "session.jsonl"
        path.write_bytes(_linear_session())
        database = _sqlite(tmp_path)
        async with open_builtin_contexts(BuiltinConfig(database=database)) as contexts:
            await contexts.register_source_definition(manifest_for_definition(PI_EVENT_SOURCE_DEFINITION))
            await ConnectorLifecycle(
                sink=_ContextSink(contexts, registry),
                checkpoints=_ContextCheckpointStore(contexts),
            ).run(PiSessionConnector(path), binding)

        async with open_builtin_contexts(BuiltinConfig(database=database)) as contexts:
            result = await ConnectorLifecycle(
                sink=_ContextSink(contexts, registry),
                checkpoints=_ContextCheckpointStore(contexts),
            ).run(PiSessionConnector(path), binding)
            assert result.items == ()
            assert result.committed_checkpoint == result.previous_checkpoint
            context = await contexts.get(binding.scope_id)
            assert len(await context.sources.list()) == 3

    asyncio.run(scenario())


def test_contexts_branch_switch_imports_only_the_new_branch(tmp_path) -> None:
    async def scenario() -> None:
        registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
        binding = _binding()
        path = tmp_path / "session.jsonl"
        path.write_bytes(_branched_session())
        async with open_builtin_contexts(BuiltinConfig(database=_sqlite(tmp_path))) as contexts:
            await contexts.register_source_definition(manifest_for_definition(PI_EVENT_SOURCE_DEFINITION))
            lifecycle = ConnectorLifecycle(
                sink=_ContextSink(contexts, registry),
                checkpoints=_ContextCheckpointStore(contexts),
            )
            await lifecycle.run(PiSessionConnector(path, leaf_id="b2"), binding)
            switched = await lifecycle.run(PiSessionConnector(path, leaf_id="c1"), binding)

            assert [outcome.status for outcome in switched.items] == ["accepted"]
            context = await contexts.get(binding.scope_id)
            names = {source.name for source in await context.sources.list()}
            assert names == {"pi:sess-1:u1", "pi:sess-1:a1", "pi:sess-1:b1", "pi:sess-1:b2", "pi:sess-1:c1"}

    asyncio.run(scenario())


def test_contexts_read_back_full_event_payload_and_projection(tmp_path) -> None:
    async def scenario() -> None:
        registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
        binding = _binding()
        path = tmp_path / "session.jsonl"
        path.write_bytes(_linear_session())
        async with open_builtin_contexts(BuiltinConfig(database=_sqlite(tmp_path))) as contexts:
            await contexts.register_source_definition(manifest_for_definition(PI_EVENT_SOURCE_DEFINITION))
            await ConnectorLifecycle(
                sink=_ContextSink(contexts, registry),
                checkpoints=_ContextCheckpointStore(contexts),
            ).run(PiSessionConnector(path), binding)

            context = await contexts.get(binding.scope_id)
            target = next(source for source in await context.sources.list() if source.name == "pi:sess-1:u1")
            payload = await context.sources.read(target)
            assert payload["content"] == "user: hello"
            assert payload["record"]["message"]["content"] == "hello"
            evidence = TextEvidence.model_validate(target.projection(TEXT_EVIDENCE_PROJECTION_KEY))  # ty: ignore[possibly-missing-attribute]
            assert evidence.source_id == "pi:sess-1:u1"
            assert evidence.content == "user: hello"

    asyncio.run(scenario())


def test_large_tool_result_is_stored_without_truncation(tmp_path) -> None:
    async def scenario() -> None:
        large = "x" * (1024 * 1024)
        registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
        binding = _binding()
        path = tmp_path / "session.jsonl"
        path.write_bytes(
            _jsonl(
                _session(),
                _message(
                    "r1",
                    None,
                    "toolResult",
                    [{"type": "text", "text": large}],
                    toolName="bash",
                    toolCallId="call-1",
                ),
            )
        )
        async with open_builtin_contexts(BuiltinConfig(database=_sqlite(tmp_path))) as contexts:
            await contexts.register_source_definition(manifest_for_definition(PI_EVENT_SOURCE_DEFINITION))
            await ConnectorLifecycle(
                sink=_ContextSink(contexts, registry),
                checkpoints=_ContextCheckpointStore(contexts),
            ).run(PiSessionConnector(path), binding)

            context = await contexts.get(binding.scope_id)
            target = next(iter(await context.sources.list()))
            payload = await context.sources.read(target)
            assert payload["content"] == f"toolResult: [tool_result bash call-1] {large}"
            assert len(payload["record"]["message"]["content"][0]["text"]) == len(large)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# E. Real HTTP protocol through the ASGI server and the remote worker
# ---------------------------------------------------------------------------


def test_asgi_remote_worker_registers_manifest_and_resumes(tmp_path) -> None:
    async def scenario() -> None:
        app = create_server_app(
            settings=ServerSettings(
                database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'server.db'}"),
                mcp=McpConfig(enabled=False),
            )
        )
        path = tmp_path / "session.jsonl"
        path.write_bytes(_linear_session())
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as transport,
        ):
            client = PowerContextClient(
                "http://testserver",
                http_client=transport,
                trust_transport_security=True,
            )
            scope = await client.create_scope(
                CreateScopeRequest(title="Pi history", summary="Pi branch evidence", idempotency_key="pi-scope")
            )
            worker = RemoteConnectorWorker(
                client=client,
                registry=SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,)),
            )
            binding = ConnectorBinding(
                scope_id=scope.scope_id,
                binding_id="pi-1",
                connector_name="pi-session-history",
                connector_version="1",
            )

            first = await worker.run(PiSessionConnector(path), binding)
            assert first.status == ConnectorRunStatus.COMPLETE
            assert len(first.items) == 3

            second = await worker.run(PiSessionConnector(path), binding)
            assert second.items == ()
            assert second.committed_checkpoint == second.previous_checkpoint

    asyncio.run(scenario())
