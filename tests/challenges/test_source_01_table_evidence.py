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

"""Local acceptance mirror for challenge ``source-01-table-evidence``."""

from __future__ import annotations

import asyncio
import json

import pytest

from powercontext.artifacts import MemoryCitation
from powercontext.builtin.artifacts.memory.models import MemoryEntryInput
from powercontext.builtin.artifacts.memory.protocols import MemoryCandidateRequest
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    SearchMemoryRequest,
    SubmitSourceObservation,
    open_builtin_contexts,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import BUILTIN_SOURCE_REGISTRY
from powercontext.sources import (
    SourceDefinitionRegistry,
    SourceRef,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_ADAPTER,
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowInput,
    CSVRowSource,
    CSVRowTextEvidenceProjection,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.observations import SourceObservation
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

AIRPORTS_CSV = (
    "id,ident,name,type,iso_country,municipality\n"
    "1,00A,Total Rf Heliport,heliport,US,Bensalem\n"
    "2,00AA,Aero B Ranch Airport,small_airport,US,Leoti\n"
    "3,00AK,Lowell Field,small_airport,US,Anchor Point\n"
    "4,00AL,Epps Airpark,small_airport,US,Harvest\n"
    "5,00AR,Hospital Heliport,heliport,US,Osceola\n"
    "6,00AS,Johnnie's Airport,small_airport,US,Fort Worth\n"
    "7,00AZ,Polly Ranch Airport,small_airport,US,Camp Verde\n"
    "8,00CA,Goldstone Airport,small_airport,US,Goldstone\n"
    "9,00CO,Tejon Ag Airport,small_airport,US,Center\n"
    "10,00FA,Gator Creek Airport,small_airport,US,Sarasota\n"
    "11,00FD,Florida Flying Service Airport,small_airport,US,Hilliard\n"
    "12,00FL,East Lake Airport,small_airport,US,Paisley\n"
)


def _document(content: bytes | str, *, document_id: str = "airports.csv", primary_key: str = "id") -> CSVDocumentInput:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return CSVDocumentInput(document_id=document_id, primary_key=primary_key, content=content)


# ---------------------------------------------------------------------------
# Group 1: parsing and atomic validation
# ---------------------------------------------------------------------------


def test_parses_all_rows_and_preserves_order() -> None:
    rows = parse_csv_document(_document(AIRPORTS_CSV))

    assert len(rows) == 12
    assert [row.primary_key_value for row in rows] == [str(index) for index in range(1, 13)]
    assert rows[0].columns == ("id", "ident", "name", "type", "iso_country", "municipality")
    assert rows[0].values == ("1", "00A", "Total Rf Heliport", "heliport", "US", "Bensalem")
    assert rows[-1].values == ("12", "00FL", "East Lake Airport", "small_airport", "US", "Paisley")


def test_accepts_utf8_bom() -> None:
    document = _document("\ufeffid,name\n1,Alpha\n")

    rows = parse_csv_document(document)

    assert rows[0].columns == ("id", "name")
    assert rows[0].values == ("1", "Alpha")


def test_handles_quoted_commas_escaped_quotes_and_newlines() -> None:
    document = _document('id,note,country\n1,"a,b",US\n2,"He said ""hi""",US\n3,"line one\nline two",US\n')

    rows = parse_csv_document(document)

    assert rows[0].values[1] == "a,b"
    assert rows[1].values[1] == 'He said "hi"'
    assert rows[2].values[1] == "line one\nline two"


def test_ignores_completely_blank_physical_records() -> None:
    document = _document("id,name\n1,Alpha\n\n2,Beta\n\r\n3,Gamma\n")

    rows = parse_csv_document(document)

    assert [row.primary_key_value for row in rows] == ["1", "2", "3"]


def test_header_only_document_returns_empty_tuple() -> None:
    assert parse_csv_document(_document("id,name\n")) == ()


def test_rejects_empty_document() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document(b""))


def test_rejects_blank_header() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document("\n1,Alpha\n"))


def test_rejects_blank_header_field() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document("id, ,name\n1,x,Alpha\n"))


def test_rejects_pure_whitespace_header_field() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document("id,   ,name\n1,x,Alpha\n"))


def test_rejects_duplicate_header_fields() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document("id,name,id\n1,Alpha,2\n"))


def test_rejects_unknown_primary_key() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document("id,name\n1,Alpha\n", primary_key="missing"))


def test_rejects_blank_primary_key_value() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document("id,name\n,Alpha\n"))


def test_rejects_pure_whitespace_primary_key_value() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document("id,name\n   ,Alpha\n"))


def test_rejects_duplicate_primary_key_values() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document("id,name\n1,Alpha\n1,Beta\n"))


def test_rejects_row_width_mismatch() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document("id,name,country\n1,Alpha\n"))


def test_rejects_invalid_utf8() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document(b"id,name\n1,\xff\n"))


def test_rejects_malformed_quoting() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document('id,name\n1,"unterminated\n'))


def test_preserves_whitespace_and_string_values() -> None:
    document = _document('id,name,code,note\n 1 , Alpha ,001,\n')

    rows = parse_csv_document(document)

    assert rows[0].primary_key_value == " 1 "
    assert rows[0].values == (" 1 ", " Alpha ", "001", "")


def test_preserves_column_names_with_spaces_and_exact_primary_key() -> None:
    document = _document(" id , name \n 1 , Alpha \n", primary_key=" id ")

    rows = parse_csv_document(document)

    assert rows[0].columns == (" id ", " name ")
    assert rows[0].primary_key == " id "
    assert rows[0].primary_key_value == " 1 "


def test_error_in_later_row_never_returns_partial_results() -> None:
    document = _document("id,name\n1,Alpha\n2,Beta\n2,Gamma\n")

    with pytest.raises(ValueError):
        parse_csv_document(document)


# ---------------------------------------------------------------------------
# Group 2: identity and TextEvidence projection
# ---------------------------------------------------------------------------


def test_identity_uses_document_and_primary_key_value() -> None:
    assert csv_row_source_id("airports.csv", "1") == "csv:airports.csv:1"


def test_identity_percent_encodes_each_component() -> None:
    assert csv_row_source_id("a b/c:d", " 1 ") == "csv:a%20b%2Fc%3Ad:%201%20"
    assert csv_row_source_id("北京", "1") == "csv:%E5%8C%97%E4%BA%AC:1"


def test_identity_distinguishes_whitespace_variants() -> None:
    assert csv_row_source_id("doc", "1") != csv_row_source_id("doc", " 1 ")


def test_identity_rejects_values_over_the_length_limit() -> None:
    with pytest.raises(ValueError):
        csv_row_source_id("d" * 300, "1")


def test_resolve_and_read_round_trip() -> None:
    row = CSVRowInput(
        document_id="airports.csv",
        primary_key="id",
        primary_key_value="1",
        columns=("id", "name"),
        values=("1", "Alpha"),
    )

    async def scenario() -> None:
        source = await CSV_ROW_SOURCE_ADAPTER.resolve(row)
        assert type(source) is CSVRowSource
        assert source.name == source.source_id == "csv:airports.csv:1"
        assert source.materialization.value == "captured"
        assert source.definition_version == "1"
        assert await CSV_ROW_SOURCE_ADAPTER.read(source) == row

    asyncio.run(scenario())


def test_projection_serializes_compact_json_in_header_order() -> None:
    source = CSVRowSource(
        name="csv:airports.csv:1",
        source_id="csv:airports.csv:1",
        materialization="captured",
        document_id="airports.csv",
        primary_key="id",
        primary_key_value="1",
        columns=("id", "ident", "name"),
        values=("1", "00A", "Total Rf Heliport"),
    )

    evidence = CSVRowTextEvidenceProjection().project(source)

    assert evidence.source_type == "csv-row"
    assert evidence.source_id == "csv:airports.csv:1"
    assert evidence.content == '{"id":"1","ident":"00A","name":"Total Rf Heliport"}'
    assert json.loads(evidence.content) == {"id": "1", "ident": "00A", "name": "Total Rf Heliport"}


def test_projection_keeps_non_ascii_verbatim() -> None:
    source = CSVRowSource(
        name="csv:doc:1",
        source_id="csv:doc:1",
        materialization="captured",
        document_id="doc",
        primary_key="id",
        primary_key_value="1",
        columns=("id", "name"),
        values=("1", "北京"),
    )

    evidence = CSVRowTextEvidenceProjection().project(source)

    assert evidence.content == '{"id":"1","name":"北京"}'


def test_projection_metadata_has_exactly_five_keys_and_full_row() -> None:
    source = CSVRowSource(
        name="csv:doc:1",
        source_id="csv:doc:1",
        materialization="captured",
        document_id="doc",
        primary_key="id",
        primary_key_value="1",
        columns=("id", "name", "note"),
        values=("1", "Alpha", "Alpha"),
    )

    evidence = CSVRowTextEvidenceProjection().project(source)

    assert set(evidence.metadata) == {"document_id", "primary_key", "primary_key_value", "columns", "row"}
    assert evidence.metadata["document_id"] == "doc"
    assert evidence.metadata["primary_key"] == "id"
    assert evidence.metadata["primary_key_value"] == "1"
    assert evidence.metadata["columns"] == ["id", "name", "note"]
    assert evidence.metadata["row"] == {"id": "1", "name": "Alpha", "note": "Alpha"}
    assert list(evidence.metadata["row"]) == ["id", "name", "note"]


# ---------------------------------------------------------------------------
# Group 3: manifest, observation, and real runtime persistence
# ---------------------------------------------------------------------------


class _CSVRowPipeline:
    """Deterministic stand-in for the fixed Memory pipeline used by the acceptance."""

    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        entries: list[MemoryEntryInput] = []
        for source in request.sources:
            if isinstance(source, SourceObservation):
                evidence = TextEvidence.model_validate(source.projection(TEXT_EVIDENCE_PROJECTION_KEY))
                text = evidence.content
            elif isinstance(source, CSVRowSource):
                row = dict(zip(source.columns, source.values, strict=True))
                text = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            else:
                continue
            entries.append(MemoryEntryInput(kind="fact", text=text, sources=(source,)))
        return tuple(entries)


def _registry() -> SourceDefinitionRegistry:
    return SourceDefinitionRegistry((*BUILTIN_SOURCE_REGISTRY.definitions, CSV_ROW_SOURCE_DEFINITION))


def _observations(rows: tuple[CSVRowInput, ...], registry: SourceDefinitionRegistry) -> list[SourceObservation]:
    async def build() -> list[SourceObservation]:
        result: list[SourceObservation] = []
        for row in rows:
            source = await registry.resolve(row)
            result.append(project_source_for_transport(registry, source))
        return result

    return asyncio.run(build())


def test_manifest_declares_standard_text_evidence_projection() -> None:
    manifest = manifest_for_definition(CSV_ROW_SOURCE_DEFINITION)

    assert manifest.name == "csv-row"
    assert manifest.version == "1"
    assert manifest.fingerprint.startswith("sha256:")
    assert len(manifest.projections) == 1
    projection = manifest.projections[0]
    assert projection.key == TEXT_EVIDENCE_PROJECTION_KEY
    assert projection.schema_ == TextEvidence.model_json_schema()
    assert manifest_for_definition(CSV_ROW_SOURCE_DEFINITION) == manifest


def test_observation_envelope_matches_manifest_and_row() -> None:
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    row = parse_csv_document(_document(AIRPORTS_CSV))[0]
    source = asyncio.run(registry.resolve(row))
    observation = project_source_for_transport(registry, source)
    manifest = manifest_for_definition(CSV_ROW_SOURCE_DEFINITION)

    assert observation.source_type == "csv-row"
    assert observation.name == "csv:airports.csv:1"
    assert observation.definition_version == "1"
    assert observation.definition_fingerprint == manifest.fingerprint
    assert observation.payload["values"] == ["1", "00A", "Total Rf Heliport", "heliport", "US", "Bensalem"]
    evidence = TextEvidence.model_validate(observation.projection(TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.source_id == observation.name
    assert json.loads(evidence.content)["name"] == "Total Rf Heliport"


def test_contexts_ingestion_persists_rows_and_exposes_citation_sources(tmp_path) -> None:
    async def scenario() -> None:
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'rows.db'}"))
        registry = _registry()
        rows = parse_csv_document(_document(AIRPORTS_CSV))
        manifest = manifest_for_definition(CSV_ROW_SOURCE_DEFINITION)
        async with open_builtin_contexts(
            config,
            candidate_pipeline=_CSVRowPipeline(),
        ) as contexts:
            scope = await contexts.scopes.create(
                ScopeDraft(title="Rows", summary="CSV rows", idempotency_key="csv-contexts")
            )
            await contexts.register_source_definition(manifest)
            for row in rows:
                source = await registry.resolve(row)
                observation = project_source_for_transport(registry, source)
                receipt = await contexts.submit_source_observation(
                    SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                )
                assert receipt.source_ref == SourceRef(source_type="csv-row", source_id=source.source_id)

            context = await contexts.get(scope.scope_id)
            flushed = await context.triggers.flush(limit=100)
            assert flushed.source_count == 12
            assert flushed.memory_ref is not None

            memory = await context.artifacts.memory.head("memory")
            page = await context.artifacts.memory.search("Total Rf Heliport", memories=(memory,), mode="fts")
            assert page.hits
            hit = page.hits[0]
            entries = await context.artifacts.memory.entries(memory)
            entry = next(
                item
                for item in entries
                if item.entry_id == hit.entry_id and item.entry_version_id == hit.entry_version_id
            )
            expected_ref = SourceRef(source_type="csv-row", source_id=csv_row_source_id("airports.csv", "1"))
            assert expected_ref in entry.sources

            stored = await context.sources.list()
            target = next(item for item in stored if item.name == expected_ref.source_id)
            payload = await context.sources.read(target)
            assert payload["values"] == ["1", "00A", "Total Rf Heliport", "heliport", "US", "Bensalem"]

    asyncio.run(scenario())


def test_runtime_ingestion_registers_manifest_and_exposes_citation_sources(tmp_path) -> None:
    async def scenario() -> None:
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}"))
        registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
        rows = parse_csv_document(_document(AIRPORTS_CSV))
        manifest = manifest_for_definition(CSV_ROW_SOURCE_DEFINITION)
        async with open_builtin_runtime(
            config,
            scheduler_path=tmp_path / "scheduler.db",
            candidate_pipeline=_CSVRowPipeline(),
        ) as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Rows", summary="CSV runtime", idempotency_key="csv-runtime")
            )
            await runtime.ingestion.register(manifest)
            for row in rows:
                source = await registry.resolve(row)
                observation = project_source_for_transport(registry, source)
                receipt = await runtime.ingestion.submit(
                    SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                )
                assert receipt.source_ref == SourceRef(source_type="csv-row", source_id=source.source_id)

            memory = runtime.memory.for_scope(scope.scope_id)
            flushed = await memory.flush()
            assert flushed.source_count == 12
            page = await memory.search(SearchMemoryRequest(query="Total Rf Heliport", mode="fts"))
            assert page.hits
            hit = page.hits[0]
            record = await memory.get(
                GetMemoryEntryRequest(
                    citation=MemoryCitation(
                        memory_ref=hit.memory_ref,
                        entry_id=hit.entry_id,
                        entry_version_id=hit.entry_version_id,
                    )
                )
            )
            expected_ref = SourceRef(source_type="csv-row", source_id=csv_row_source_id("airports.csv", "1"))
            assert expected_ref in record.entry.sources
            assert record.entry.text == (
                '{"id":"1","ident":"00A","name":"Total Rf Heliport","type":"heliport",'
                '"iso_country":"US","municipality":"Bensalem"}'
            )

    asyncio.run(scenario())


def test_projections_survive_restart(tmp_path) -> None:
    async def scenario() -> None:
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'restart.db'}"))
        registry = _registry()
        rows = parse_csv_document(_document(AIRPORTS_CSV))
        manifest = manifest_for_definition(CSV_ROW_SOURCE_DEFINITION)
        expected: dict[str, tuple[dict, object]] = {}

        async with open_builtin_contexts(config) as contexts:
            scope = await contexts.scopes.create(
                ScopeDraft(title="Rows", summary="CSV restart", idempotency_key="csv-restart")
            )
            await contexts.register_source_definition(manifest)
            for row in rows:
                source = await registry.resolve(row)
                observation = project_source_for_transport(registry, source)
                await contexts.submit_source_observation(
                    SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                )
                expected[observation.name] = (
                    observation.payload,
                    observation.projection(TEXT_EVIDENCE_PROJECTION_KEY),
                )

        async with open_builtin_contexts(config) as contexts:
            context = await contexts.get(scope.scope_id)
            stored = await context.sources.list()
            actual = {
                source.name: (source.payload, source.projection(TEXT_EVIDENCE_PROJECTION_KEY))
                for source in stored
                if isinstance(source, SourceObservation)
            }
            assert actual == expected

    asyncio.run(scenario())


def test_typed_sources_resolve_add_read_and_list(tmp_path) -> None:
    async def scenario() -> None:
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'typed.db'}"))
        registry = _registry()
        row = parse_csv_document(_document(AIRPORTS_CSV))[0]
        async with open_builtin_contexts(config, source_registry=registry) as contexts:
            context = await contexts.get("project")
            source = await context.sources.resolve(row)
            assert type(source) is CSVRowSource
            stored = await context.sources.add(source)
            assert type(stored) is CSVRowSource
            assert await context.sources.read(stored) == row
            assert stored in await context.sources.list()

    asyncio.run(scenario())

