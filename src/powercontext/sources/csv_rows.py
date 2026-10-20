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

"""Row-level Sources for fully offline CSV documents.

Each data row becomes one captured Source whose identity is derived from the
document identity and the row's primary-key value. Values and column order are
preserved verbatim so a citation can read the complete original record back.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"


class CSVDocumentInput(BaseModel):
    """Complete offline bytes for one CSV document plus its row-identity choice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    primary_key: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("document_id must not be blank")  # noqa: TRY003
        if value != value.strip():
            raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
        return value

    @field_validator("primary_key")
    @classmethod
    def validate_primary_key(cls, value: str) -> str:
        if not value:
            raise ValueError("primary_key must not be empty")  # noqa: TRY003
        return value


class CSVRowInput(BaseModel):
    """One parsed row before it becomes a Source."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_width(self) -> CSVRowInput:
        if len(self.columns) != len(self.values):
            raise ValueError("CSV row columns and values must have the same width")  # noqa: TRY003
        return self


class CSVRowSource(Source):
    """One captured CSV row with a stable document-plus-primary-key identity."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_identity(self) -> CSVRowSource:
        if not self.source_id or self.source_id != self.source_id.strip():
            raise ValueError("source_id must be a non-empty trimmed string")  # noqa: TRY003
        if self.name != self.source_id:
            raise ValueError("CSV row Source name must equal its source_id")  # noqa: TRY003
        if len(self.columns) != len(self.values):
            raise ValueError("CSV row columns and values must have the same width")  # noqa: TRY003
        return self


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse and validate a whole CSV document before returning any row."""

    text = _decode_content(value.content)
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = _read_header(reader)
    try:
        key_index = header.index(value.primary_key)
    except ValueError:
        raise ValueError(f"primary_key {value.primary_key!r} is not a CSV column") from None  # noqa: TRY003
    return tuple(_row_inputs(reader, header, key_index, value))


def _decode_content(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error  # noqa: TRY003


def _read_header(reader: Iterator[list[str]]) -> list[str]:
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV document must contain a header row") from None  # noqa: TRY003
    except csv.Error as error:
        raise ValueError("CSV document contains malformed quoting") from error  # noqa: TRY003
    _validate_header(header)
    return header


def _row_inputs(
    reader: Iterator[list[str]],
    header: list[str],
    key_index: int,
    value: CSVDocumentInput,
) -> list[CSVRowInput]:
    rows: list[CSVRowInput] = []
    seen_primary_keys: set[str] = set()
    try:
        for record in reader:
            if not record:
                continue
            if len(record) != len(header):
                raise ValueError("CSV data row width does not match the header")  # noqa: TRY003
            primary_key_value = record[key_index]
            if not primary_key_value or not primary_key_value.strip():
                raise ValueError("CSV primary-key values must not be blank")  # noqa: TRY003
            if primary_key_value in seen_primary_keys:
                raise ValueError("CSV primary-key values must be unique within one document")  # noqa: TRY003
            seen_primary_keys.add(primary_key_value)
            rows.append(
                CSVRowInput(
                    document_id=value.document_id,
                    primary_key=value.primary_key,
                    primary_key_value=primary_key_value,
                    columns=tuple(header),
                    values=tuple(record),
                )
            )
    except csv.Error as error:
        raise ValueError("CSV document contains malformed quoting") from error  # noqa: TRY003
    return rows


def csv_row_source_id(document_id: str, primary_key_value: str, /) -> str:
    """Build the stable, collision-free identity for one CSV row."""

    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty trimmed string")  # noqa: TRY003
    if not primary_key_value or not primary_key_value.strip():
        raise ValueError("primary_key_value must not be blank")  # noqa: TRY003
    identity = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def _validate_header(header: list[str]) -> None:
    if not header:
        raise ValueError("CSV document must contain a non-empty header row")  # noqa: TRY003
    for name in header:
        if not name or not name.strip():
            raise ValueError("CSV header fields must not be blank")  # noqa: TRY003
    if len(set(header)) != len(header):
        raise ValueError("CSV header fields must be unique")  # noqa: TRY003


class CSVRowSourceAdapter:
    """Resolve and read the runtime's CSV row Source definition."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one captured row as standard textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    projections=(CSVRowTextEvidenceProjection(),),
)


__all__ = [
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
