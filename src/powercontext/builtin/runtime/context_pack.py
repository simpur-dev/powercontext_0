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

"""Incremental Context Packs for prepared context transport.

A ContextPack frames prepared context bytes as deterministic, self-verifying
chunks; a delta carries only the suffix a receiver must append to one pack to
obtain another. Wire forms are pure JSON with Base64 items so any client can
rebuild without process-local state.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass

from powercontext.errors import PowerContextError

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512
_DIGEST_PREFIX = "sha256:"
_HEX_LENGTH = 64
_MAX_FIXPOINT_ROUNDS = 10


class ContextPackError(PowerContextError, ValueError):
    """Raised when a pack or delta violates the wire or chunking contract."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


def _canonical_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_digest(digest: bytes) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(digest).hexdigest()


def _require_digest(value: object, *, code: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(_DIGEST_PREFIX)
        or len(value) != len(_DIGEST_PREFIX) + _HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value[len(_DIGEST_PREFIX) :])
    ):
        raise ContextPackError(code, f"{field} must be a sha256 hex digest")
    return value


def _require_base64_items(value: object, *, code: str, field: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(code, f"{field} must be a list of base64 strings")
    items: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError(code, f"{field} entries must be base64 strings")
        try:
            items.append(base64.b64decode(entry, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError(code, f"{field} entries must be valid base64") from error
    return tuple(items)


def _chunk_content(content: bytes, *, chunk_bytes: int) -> tuple[bytes, ...]:
    """Split content into deterministic chunks: line-aligned first, hard-cut fallback.

    A chunk boundary is preferred right after a newline so appended tails keep
    base chunks stable; oversized lines are hard-cut on UTF-8 character bounds.
    """

    if chunk_bytes < 1:
        raise ContextPackError("chunk-size", "chunk_bytes must be a positive integer")
    if not content:
        return ()
    chunks: list[bytes] = []
    pending = b""
    for line in content.splitlines(keepends=True):
        candidate = pending + line
        if len(candidate) > chunk_bytes and pending:
            chunks.append(pending)
            candidate = line
        if len(candidate) > chunk_bytes:
            pieces = _hard_cut(candidate, chunk_bytes=chunk_bytes)
            chunks.extend(pieces[:-1])
            pending = pieces[-1]
            continue
        pending = candidate
    if pending:
        chunks.append(pending)
    return tuple(chunks)


def _hard_cut(segment: bytes, *, chunk_bytes: int) -> list[bytes]:
    """Cut one oversized segment on complete UTF-8 character boundaries."""

    pieces: list[bytes] = []
    start = 0
    while start < len(segment):
        take = min(chunk_bytes, len(segment) - start)
        # Walk the cut point back until the piece decodes as complete UTF-8.
        # A piece may legitimately end on a continuation byte (the tail of a
        # 3/4-byte char), so only decoding decides validity; at most 3 steps.
        while take > 0:
            try:
                segment[start : start + take].decode("utf-8")
                break
            except UnicodeDecodeError:
                take -= 1
        if take <= 0:
            raise ContextPackError("chunk-utf8", "cannot fit one UTF-8 character inside chunk_bytes")
        pieces.append(segment[start : start + take])
        start += take
    return pieces


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One deterministic, self-verifying framing of prepared context bytes."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...], /) -> ContextPack:
        for entry in items:
            if not isinstance(entry, bytes):
                raise ContextPackError("item-type", "pack items must be bytes")
        encoded = [base64.b64encode(entry).decode("ascii") for entry in items]
        digest = _encode_digest(_canonical_json({"items": encoded, "schema": CONTEXT_PACK_SCHEMA}))
        byte_count = cls._resolve_byte_count(encoded, digest)
        return cls(items=tuple(items), digest=digest, byte_count=byte_count)

    @classmethod
    def from_content(cls, content: str | None, /, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        raw = (content or "").encode("utf-8")
        return cls.build(_chunk_content(raw, chunk_bytes=chunk_bytes))

    def to_bytes(self, /) -> bytes:
        encoded = [base64.b64encode(entry).decode("ascii") for entry in self.items]
        return _canonical_json(
            {"byte_count": self.byte_count, "digest": self.digest, "items": encoded, "schema": CONTEXT_PACK_SCHEMA}
        )

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPack:
        if not isinstance(data, (bytes, bytearray)):
            raise ContextPackError("wire-type", "pack bytes must be bytes")
        try:
            body = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ContextPackError("wire-json", "pack bytes are not valid JSON") from error
        if not isinstance(body, dict) or set(body) != {"byte_count", "digest", "items", "schema"}:
            raise ContextPackError("wire-keys", "pack must contain exactly the wire fields")
        if body["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("wire-schema", "pack schema mismatch")
        byte_count = body["byte_count"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count != len(data):
            raise ContextPackError("wire-byte-count", "pack byte_count does not match its bytes")
        digest = _require_digest(body["digest"], code="wire-digest", field="pack digest")
        items = _require_base64_items(body["items"], code="wire-items", field="pack items")
        rebuilt = cls.build(items)
        if rebuilt.digest != digest:
            raise ContextPackError("wire-digest", "pack digest does not match its items")
        return rebuilt

    @staticmethod
    def _resolve_byte_count(encoded: list[str], digest: str) -> int:
        guess = len(
            _canonical_json({"digest": digest, "items": encoded, "schema": CONTEXT_PACK_SCHEMA})
        ) + len("byte_count") + 4
        for _ in range(_MAX_FIXPOINT_ROUNDS):
            frame = _canonical_json(
                {"byte_count": guess, "digest": digest, "items": encoded, "schema": CONTEXT_PACK_SCHEMA}
            )
            actual = len(frame)
            if actual == guess:
                return actual
            guess = actual
        raise ContextPackError("wire-byte-count", "byte_count fixpoint did not converge")


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """The suffix one pack must append to become another pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        shared = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            shared += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=shared,
            appended=target.items[shared:],
        )

    @property
    def byte_count(self, /) -> int:
        return len(self.to_bytes())

    def to_bytes(self, /) -> bytes:
        encoded = [base64.b64encode(entry).decode("ascii") for entry in self.appended]
        return _canonical_json(
            {
                "appended": encoded,
                "base_digest": self.base_digest,
                "prefix_items": self.prefix_items,
                "schema": CONTEXT_PACK_DELTA_SCHEMA,
                "target_digest": self.target_digest,
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPackDelta:
        if not isinstance(data, (bytes, bytearray)):
            raise ContextPackError("wire-type", "delta bytes must be bytes")
        try:
            body = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ContextPackError("wire-json", "delta bytes are not valid JSON") from error
        if not isinstance(body, dict) or set(body) != {"appended", "base_digest", "prefix_items", "schema", "target_digest"}:
            raise ContextPackError("wire-keys", "delta must contain exactly the wire fields")
        if body["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("wire-schema", "delta schema mismatch")
        base_digest = _require_digest(body["base_digest"], code="wire-digest", field="base_digest")
        target_digest = _require_digest(body["target_digest"], code="wire-digest", field="target_digest")
        prefix_items = body["prefix_items"]
        if isinstance(prefix_items, bool) or not isinstance(prefix_items, int) or prefix_items < 0:
            raise ContextPackError("wire-prefix", "prefix_items must be a non-negative integer")
        appended = _require_base64_items(body["appended"], code="wire-items", field="delta appended")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Rebuild the target pack from a base pack and one delta."""

    if delta.base_digest != base.digest:
        raise ContextPackError("base-digest", "delta base digest does not match the base pack")
    if isinstance(delta.prefix_items, bool) or not isinstance(delta.prefix_items, int):
        raise ContextPackError("prefix-type", "prefix_items must be an integer")
    if delta.prefix_items < 0 or delta.prefix_items > len(base.items):
        raise ContextPackError("prefix-range", "prefix_items is outside the base pack")
    rebuilt = ContextPack.build(base.items[: delta.prefix_items] + tuple(delta.appended))
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("target-digest", "rebuilt pack digest does not match the delta target")
    return rebuilt


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
