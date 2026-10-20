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

"""Pi v3 session history as per-event Sources.

The parser walks the raw ``parentId`` ancestry of one selected leaf and keeps
the complete visible evidence, including history that Pi itself would drop at a
compaction boundary. The connector only decides what is new: persistence,
checkpoint CAS, and durable receipts stay with ``ConnectorLifecycle``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError, model_validator

from powercontext.errors import InvalidConnectorRunError
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"


class PiSessionInput(BaseModel):
    """One offline Pi v3 JSONL file and the leaf whose branch becomes evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, before it becomes a Source."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_identity(self) -> PiEventInput:
        if not self.session_id or not self.event_id:
            raise ValueError("Pi event identity must not be blank")  # noqa: TRY003
        if not self.content:
            raise ValueError("Pi event content must not be blank")  # noqa: TRY003
        return self


class PiEventSource(Source):
    """One captured Pi session event with a stable session-plus-event identity."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_identity(self) -> PiEventSource:
        if not self.source_id or self.source_id != self.source_id.strip():
            raise ValueError("source_id must be a non-empty trimmed string")  # noqa: TRY003
        if self.name != self.source_id:
            raise ValueError("Pi event Source name must equal its source_id")  # noqa: TRY003
        return self


@dataclass(frozen=True, slots=True)
class _ParsedSession:
    session_id: str
    leaf_id: str
    events: tuple[PiEventInput, ...]


class PiSessionCheckpoint(BaseModel):
    """Durable incremental state for one Pi session binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    session_id: str
    leaf_id: str
    events: dict[str, str]


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Return every visible event on the selected leaf's raw ancestor chain."""

    return _parse_session(value).events


def pi_event_source_id(session_id: str, event_id: str, /) -> str:
    """Build the stable Pi event identity from the session and event ids."""

    if not session_id or session_id != session_id.strip():
        raise ValueError("session_id must be a non-empty trimmed string")  # noqa: TRY003
    if not event_id or event_id != event_id.strip():
        raise ValueError("event_id must be a non-empty trimmed string")  # noqa: TRY003
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"Pi event source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def _parse_session(value: PiSessionInput, /) -> _ParsedSession:
    records = _parse_records(_decode_content(value.content))
    if not records:
        raise ValueError("Pi session document must contain a session header")  # noqa: TRY003

    session_id = _session_id(records[0])
    nodes = _build_nodes(records[1:])
    if not nodes:
        raise ValueError("Pi session document must contain at least one event")  # noqa: TRY003
    leaf_id = value.leaf_id if value.leaf_id is not None else next(reversed(nodes))
    if leaf_id not in nodes:
        raise ValueError(f"Pi leaf {leaf_id!r} does not exist")  # noqa: TRY003
    chain = _ancestor_chain(nodes, leaf_id)
    return _ParsedSession(session_id=session_id, leaf_id=leaf_id, events=_events(session_id, chain))


def _build_nodes(records: list[dict[str, JsonValue]]) -> dict[str, dict[str, JsonValue]]:
    nodes: dict[str, dict[str, JsonValue]] = {}
    for record in records:
        event_id = _event_id(record)
        if event_id in nodes:
            raise ValueError(f"duplicate Pi event id {event_id!r}")  # noqa: TRY003
        if "parentId" not in record:
            raise ValueError(f"Pi event {event_id!r} is missing parentId")  # noqa: TRY003
        parent_id = record["parentId"]
        if parent_id is not None and (not isinstance(parent_id, str) or not parent_id or parent_id not in nodes):
            raise ValueError(f"Pi event {event_id!r} references an unknown parent")  # noqa: TRY003
        nodes[event_id] = record
    return nodes


def _ancestor_chain(nodes: dict[str, dict[str, JsonValue]], leaf_id: str) -> list[dict[str, JsonValue]]:
    chain: list[dict[str, JsonValue]] = []
    current: str | None = leaf_id
    while current is not None:
        record = nodes[current]
        chain.append(record)
        parent = record["parentId"]
        current = parent if isinstance(parent, str) else None
    chain.reverse()
    return chain


def _events(session_id: str, chain: list[dict[str, JsonValue]]) -> tuple[PiEventInput, ...]:
    events: list[PiEventInput] = []
    for record in chain:
        rendered = _render_record(record)
        if rendered is None:
            continue
        role, body = rendered
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=_event_id(record),
                parent_id=record["parentId"] if isinstance(record["parentId"], str) else None,
                event_type=_event_type(record),
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


def _decode_content(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be valid UTF-8") from error  # noqa: TRY003


def _parse_records(text: str) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pi session line {line_number} is not valid JSON") from error  # noqa: TRY003
        if not isinstance(record, dict):
            raise ValueError(f"Pi session line {line_number} must be a JSON object")  # noqa: TRY003, TRY004
        records.append(record)
    return records


def _session_id(header: dict[str, JsonValue]) -> str:
    if header.get("type") != "session":
        raise ValueError("Pi session document must start with a session record")  # noqa: TRY003
    if header.get("version") != 3:
        raise ValueError("Pi session document must use version 3")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header id must be a non-empty string")  # noqa: TRY003
    return session_id


def _event_id(record: dict[str, JsonValue]) -> str:
    event_id = record.get("id")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("Pi event id must be a non-empty string")  # noqa: TRY003
    return event_id


def _event_type(record: dict[str, JsonValue]) -> str:
    event_type = record.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("Pi event type must be a non-empty string")  # noqa: TRY003
    return event_type


def _render_record(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    event_type = _event_type(record)
    if event_type == "message":
        return _render_message(record)
    if event_type == "bashExecution":
        return _render_bash_execution(record)
    if event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return event_type, summary
    if event_type == "custom_message":
        body = _render_blocks(record.get("content"))
        if not body:
            return None
        return "custom", body
    return None


def _render_message(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    message = record.get("message")
    if not isinstance(message, dict):
        raise ValueError("Pi message record must contain a message object")  # noqa: TRY003, TRY004
    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError("Pi message role must be a non-empty string")  # noqa: TRY003
    if role == "system":
        return None
    if role == "toolResult":
        name = _string_or(message.get("toolName"), "?")
        call_id = _string_or(message.get("toolCallId"), "?")
        return role, f"[tool_result {name} {call_id}] {_render_blocks(message.get('content'))}"
    if role == "bashExecution":
        return _render_bash_execution(message)
    body = _render_blocks(message.get("content"))
    if not body:
        return None
    return role, body


def _render_bash_execution(value: dict[str, JsonValue]) -> tuple[str, str] | None:
    command = _first_string(value, ("command", "cmd"))
    output = _first_string(value, ("output", "stdout"))
    if output is None:
        output = _render_blocks(value.get("content"))
    if command is None and not output:
        return None
    return "bashExecution", f"$ {command or ''}\n{output}"


def _render_blocks(content: JsonValue | None) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    rendered: list[str] = []
    for block in content:
        if isinstance(block, str):
            if block:
                rendered.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                rendered.append(text)
        elif block_type == "toolCall":
            rendered.append(_render_tool_call(block))
    return "\n".join(rendered)


def _render_tool_call(block: dict[str, JsonValue]) -> str:
    name = _string_or(block.get("name") or block.get("toolName"), "?")
    call_id = _string_or(block.get("id") or block.get("toolCallId"), "?")
    arguments = block.get("arguments", block.get("args", {}))
    if isinstance(arguments, str):
        rendered_arguments = arguments
    else:
        try:
            rendered_arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValueError("Pi tool call arguments must be JSON serializable") from error  # noqa: TRY003
    return f"[tool_call {name} {call_id}] {rendered_arguments}"


def _string_or(value: JsonValue | None, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def _first_string(value: dict[str, JsonValue], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str):
            return candidate
    return None


def _record_digest(record: dict[str, JsonValue]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_checkpoint(raw: JsonValue | None, /) -> PiSessionCheckpoint | None:
    if raw is None:
        return None
    try:
        return PiSessionCheckpoint.model_validate(raw)
    except ValidationError as error:
        raise InvalidConnectorRunError("checkpoint", str(error)) from error


class PiEventSourceAdapter:
    """Resolve and read the runtime's Pi event Source definition."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one Pi event as standard textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class PiSessionConnector:
    """Import the selected branch of one Pi session incrementally."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        parsed = _parse_session(PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id))
        checkpoint = _load_checkpoint(session.checkpoint)
        if checkpoint is not None and checkpoint.session_id != parsed.session_id:
            raise ValueError("Pi session identity changed for the bound file")  # noqa: TRY003
        imported = dict(checkpoint.events) if checkpoint is not None else {}
        for event in parsed.events:
            digest = _record_digest(event.record)
            previous = imported.get(event.event_id)
            if previous is not None:
                if previous != digest:
                    raise ValueError(f"already imported Pi event {event.event_id!r} was rewritten")  # noqa: TRY003
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            imported[event.event_id] = digest
        next_checkpoint = PiSessionCheckpoint(
            session_id=parsed.session_id,
            leaf_id=parsed.leaf_id,
            events=imported,
        )
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint=next_checkpoint.model_dump(mode="json"),
        )


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    projections=(PiEventTextEvidenceProjection(),),
)


__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionCheckpoint",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
