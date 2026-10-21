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

"""HTTP cross-encoder reranking for coarse Memory search candidates.

One rerank call fans the candidate pool out to an offline scoring service in
bounded batches, keeps the original global ranks intact, validates every
response strictly, and selects the final ranks from one global sort. Identity
stays with the caller: only ranks are returned, never rewritten hits.
"""

from __future__ import annotations

import math
from typing import cast

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import (
    MemoryRerankDecision,
    _normalize_ranks,
    _validate_bounds,
)
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANKER_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Rank candidates with one offline HTTP scoring service."""

    policy_id: str

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANKER_POLICY_ID,
    ) -> None:
        if not endpoint or not endpoint.strip():
            raise ValueError("rerank endpoint must be a non-empty string")  # noqa: TRY003
        if batch_size < 1:
            raise ValueError("rerank batch size must be at least one")  # noqa: TRY003
        if not policy_id or not policy_id.strip():
            raise ValueError("rerank policy id must be a non-empty string")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Batch-score the pool and return globally sorted original ranks."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = range(start + 1, start + 1 + len(batch))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text} for rank, candidate in zip(ranks, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            for rank, score in _parse_batch_scores(response, expected_ranks=set(ranks)):
                scores[rank] = score
            requests += 1
        ordered = tuple(sorted(scores, key=lambda rank: (-scores[rank], rank)))
        selected_ranks, discarded_rank_count = _normalize_ranks(
            ordered,
            candidate_count=len(candidates),
            limit=limit,
        )
        usage = InferenceUsage(requests=requests)
        if selected_ranks:
            return MemoryRerankDecision(
                selected_ranks=selected_ranks,
                usage=usage,
                discarded_rank_count=discarded_rank_count,
            )
        return MemoryRerankDecision(
            selected_ranks=tuple(range(1, limit + 1)),
            usage=usage,
            discarded_rank_count=discarded_rank_count,
            used_fallback=True,
        )


def _parse_batch_scores(response: httpx.Response, *, expected_ranks: set[int]) -> list[tuple[int, float]]:
    """Validate one strict batch response and return its rank/score pairs."""

    body = _parse_json_body(response)
    records = _validate_scores_envelope(body, batch_count=len(expected_ranks))
    seen: set[int] = set()
    parsed: list[tuple[int, float]] = []
    for record in records:
        rank, score = _validate_score_record(record, expected_ranks=expected_ranks, seen=seen)
        parsed.append((rank, score))
    return parsed


def _parse_json_body(response: httpx.Response) -> dict[str, object]:
    try:
        body: object = response.json()
    except ValueError as error:
        raise ValueError("rerank response is not valid JSON") from error  # noqa: TRY003
    if not isinstance(body, dict) or set(body) != {"scores"}:
        raise ValueError("rerank response must contain exactly the key 'scores'")  # noqa: TRY003
    return cast("dict[str, object]", body)


def _validate_scores_envelope(body: dict[str, object], *, batch_count: int) -> list[dict[str, object]]:
    records = body["scores"]
    if not isinstance(records, list):
        raise ValueError("rerank scores must be a list")  # noqa: TRY003, TRY004
    if len(records) != batch_count:
        raise ValueError("rerank scores must cover exactly the batch candidates")  # noqa: TRY003
    typed: list[dict[str, object]] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise ValueError("rerank score records must be JSON objects")  # noqa: TRY003, TRY004
        typed.append(cast("dict[str, object]", raw_record))
    return typed


def _validate_score_record(record: object, *, expected_ranks: set[int], seen: set[int]) -> tuple[int, float]:
    if not isinstance(record, dict) or set(record) != {"rank", "score"}:
        raise ValueError("rerank score records must contain exactly 'rank' and 'score'")  # noqa: TRY003
    fields = cast("dict[str, object]", record)
    rank = fields["rank"]
    score = fields["score"]
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError("rerank score rank must be an integer")  # noqa: TRY003, TRY004
    if rank in seen:
        raise ValueError(f"rerank scores contain duplicate rank {rank}")  # noqa: TRY003
    if rank not in expected_ranks:
        raise ValueError(f"rerank scores contain rank {rank} outside the batch")  # noqa: TRY003
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("rerank score must be a finite number")  # noqa: TRY003
    seen.add(rank)
    return rank, float(score)


__all__ = [
    "DEFAULT_HTTP_RERANKER_POLICY_ID",
    "HTTPMemoryReranker",
]
