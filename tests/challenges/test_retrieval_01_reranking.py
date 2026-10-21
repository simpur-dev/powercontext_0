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

"""Local acceptance mirror for challenge ``retrieval-01-reranking``."""

from __future__ import annotations

import asyncio
import json
import math
import statistics
from pathlib import Path
from typing import cast

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.http_reranker import (
    HTTPMemoryReranker,
)
from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.inference import InferenceUsage
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    SearchMemoryRequest,
    open_builtin_runtime,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scifact_approx"


def _hit(rank: int, text: str | None = None) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="memory", revision=1),
        entry_id=f"entry-{rank}",
        entry_version_id=f"version-{rank}",
        text=text if text is not None else f"candidate {rank}",
        score=1.0 / rank,
        matched_by=("fts",),
    )


def _candidates(count: int) -> tuple[MemoryHit, ...]:
    return tuple(_hit(rank) for rank in range(1, count + 1))


def _service(scores_by_rank: dict[int, float], requests: list[dict[str, object]], *, shuffle: bool = False):
    """Build a MockTransport scoring service returning one score per request rank."""

    import random

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        records = [{"rank": candidate["rank"], "score": scores_by_rank[candidate["rank"]]} for candidate in body["candidates"]]
        if shuffle:
            random.shuffle(records)
        requests.append({"path": request.url.path, "candidates": body["candidates"]})
        return httpx.Response(200, json={"scores": records})

    return httpx.MockTransport(handler)


def _ranks(records: list[dict[str, object]]) -> list[int]:
    return [int(record["rank"]) for record in records]  # ty: ignore[invalid-argument-type]


def _make_reranker(scores_by_rank: dict[int, float], *, batch_size: int = 7, shuffle: bool = False):
    requests: list[dict[str, object]] = []
    transport = _service(scores_by_rank, requests, shuffle=shuffle)
    reranker = HTTPMemoryReranker(
        "http://scoring.local/rerank",
        client=httpx.AsyncClient(transport=transport),
        batch_size=batch_size,
    )
    return reranker, requests


def _rerank(reranker: HTTPMemoryReranker, candidates: tuple[MemoryHit, ...], limit: int):
    async def scenario():
        return await reranker.rerank("sensor calibration", candidates, limit)

    return asyncio.run(scenario())


# ---------------------------------------------------------------------------
# A. Protocol and batching
# ---------------------------------------------------------------------------


def test_single_batch_request_body_and_decision() -> None:
    requests: list[dict[str, object]] = []
    transport = _service({1: 0.75, 2: 0.10}, requests)
    reranker = HTTPMemoryReranker(
        "http://scoring.local/rerank", client=httpx.AsyncClient(transport=transport), batch_size=32
    )

    decision = _rerank(reranker, _candidates(2), 2)

    assert len(requests) == 1
    body_candidates_raw: object = requests[0]["candidates"]
    assert isinstance(body_candidates_raw, list)
    body_candidates: list[dict[str, object]] = []
    for raw_item in body_candidates_raw:
        if not isinstance(raw_item, dict):
            continue
        item = cast("dict[str, object]", raw_item)
        entry: dict[str, object] = {"rank": item["rank"], "text": item["text"]}
        body_candidates.append(entry)
    assert _ranks(body_candidates) == [1, 2]
    assert body_candidates[0]["text"] == "candidate 1"
    assert body_candidates[1]["text"] == "candidate 2"
    assert decision.selected_ranks == (1, 2)
    assert decision.usage == InferenceUsage(requests=1)
    assert decision.used_fallback is False
    assert decision.discarded_rank_count == 0
    asyncio.run(reranker._client.aclose())


def test_173_candidates_in_batches_of_seven_make_25_requests() -> None:
    scores = {rank: 1.0 / rank for rank in range(1, 174)}
    reranker, requests = _make_reranker(scores, batch_size=7)

    decision = _rerank(reranker, _candidates(173), 10)

    assert len(requests) == 25
    assert decision.usage.requests == 25
    sizes = [len(record["candidates"]) for record in requests]
    assert sizes == [7] * 24 + [5]
    asyncio.run(reranker._client.aclose())


def test_batch_ranks_are_global_not_renumbered() -> None:
    scores = {rank: 1.0 / rank for rank in range(1, 24)}
    reranker, requests = _make_reranker(scores, batch_size=7)

    _rerank(reranker, _candidates(23), 5)

    third_batch = requests[2]["candidates"]
    assert _ranks(third_batch)[0] == 15
    assert _ranks(third_batch)[-1] == 21
    asyncio.run(reranker._client.aclose())


def test_every_global_rank_is_sent_exactly_once() -> None:
    scores = {rank: float(rank % 11) / 11 for rank in range(1, 174)}
    reranker, requests = _make_reranker(scores, batch_size=7)

    _rerank(reranker, _candidates(173), 10)

    sent = [candidate["rank"] for record in requests for candidate in record["candidates"]]
    assert sorted(sent) == list(range(1, 174))
    asyncio.run(reranker._client.aclose())


def test_shuffled_responses_do_not_change_the_result() -> None:
    scores = {rank: (rank * 37) % 101 / 101 for rank in range(1, 51)}
    ordered, ordered_requests = _make_reranker(scores, batch_size=7)
    shuffled, shuffled_requests = _make_reranker(scores, batch_size=7, shuffle=True)

    left = _rerank(ordered, _candidates(50), 10)
    right = _rerank(shuffled, _candidates(50), 10)

    assert left.selected_ranks == right.selected_ranks
    assert len(ordered_requests) == len(shuffled_requests) == 8
    asyncio.run(ordered._client.aclose())
    asyncio.run(shuffled._client.aclose())


# ---------------------------------------------------------------------------
# B. Response validation
# ---------------------------------------------------------------------------


def _rerank_with_handler(candidates_count: int, handler) -> None:
    async def scenario() -> None:
        reranker = HTTPMemoryReranker(
            "http://scoring.local/rerank",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            batch_size=7,
        )
        await reranker.rerank("q", _candidates(candidates_count), 1)

    asyncio.run(scenario())


def test_invalid_json_body_is_a_value_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json", headers={"content-type": "application/json"})

    with pytest.raises(ValueError):
        _rerank_with_handler(2, handler)


def test_extra_top_level_key_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 1, "score": 0.5}], "model": "x"})

    with pytest.raises(ValueError):
        _rerank_with_handler(2, handler)


def test_missing_scores_key_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    with pytest.raises(ValueError):
        _rerank_with_handler(2, handler)


def test_scores_not_a_list_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": {"rank": 1, "score": 0.5}})

    with pytest.raises(ValueError):
        _rerank_with_handler(2, handler)


def test_wrong_scores_length_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 1, "score": 0.5}]})

    with pytest.raises(ValueError):
        _rerank_with_handler(3, handler)


def test_extra_record_field_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 1, "score": 0.5, "model": "x"}, {"rank": 2, "score": 0.4}]})

    with pytest.raises(ValueError):
        _rerank_with_handler(2, handler)


def test_missing_record_field_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 1}, {"rank": 2, "score": 0.4}]})

    with pytest.raises(ValueError):
        _rerank_with_handler(2, handler)


def test_duplicate_rank_in_one_batch_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 1, "score": 0.5}, {"rank": 1, "score": 0.4}]})

    with pytest.raises(ValueError):
        _rerank_with_handler(2, handler)


def test_missing_rank_in_batch_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 1, "score": 0.5}, {"rank": 3, "score": 0.4}]})

    with pytest.raises(ValueError):
        _rerank_with_handler(2, handler)


def test_cross_batch_rank_is_rejected() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if len(captured) == 1:
            return httpx.Response(200, json={"scores": [{"rank": 1, "score": 0.9}, {"rank": 8, "score": 0.8}]})
        return httpx.Response(200, json={"scores": [{"rank": r, "score": 0.1} for r in _ranks(body["candidates"])]})

    with pytest.raises(ValueError):
        _rerank_with_handler(10, handler)


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scores_are_rejected(raw: float) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 1, "score": raw}, {"rank": 2, "score": 0.4}]})

    with pytest.raises(ValueError):
        _rerank_with_handler(2, handler)


@pytest.mark.parametrize("raw", [True, "0.5", None])
def test_non_numeric_scores_are_rejected(raw: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 1, "score": raw}, {"rank": 2, "score": 0.4}]})

    with pytest.raises(ValueError):
        _rerank_with_handler(2, handler)


def test_boolean_rank_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": True, "score": 0.5}, {"rank": 2, "score": 0.4}]})

    with pytest.raises(ValueError):
        _rerank_with_handler(2, handler)


def test_http_500_keeps_the_httpx_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async def scenario() -> None:
        reranker = HTTPMemoryReranker(
            "http://scoring.local/rerank",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(httpx.HTTPError):
            await reranker.rerank("q", _candidates(2), 1)

    asyncio.run(scenario())


def test_connect_errors_propagate_untouched() -> None:
    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_service({}, []) and (lambda request: httpx.Response(200))))
        del client
        reranker = HTTPMemoryReranker(
            "http://127.0.0.1:1/rerank",
            client=httpx.AsyncClient(timeout=0.2),
        )
        with pytest.raises(httpx.HTTPError):
            await reranker.rerank("q", _candidates(2), 1)
        await reranker._client.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# C. Argument bounds
# ---------------------------------------------------------------------------


def test_empty_candidates_raise_value_error() -> None:
    reranker, _requests = _make_reranker({}, batch_size=7)
    with pytest.raises(ValueError):
        _rerank(reranker, (), 1)
    asyncio.run(reranker._client.aclose())


@pytest.mark.parametrize("limit", [0, -1, 4])
def test_limit_out_of_bounds_raises_value_error(limit: int) -> None:
    reranker, _requests = _make_reranker({1: 0.1, 2: 0.2, 3: 0.3}, batch_size=7)
    with pytest.raises(ValueError):
        _rerank(reranker, _candidates(3), limit)
    asyncio.run(reranker._client.aclose())


@pytest.mark.parametrize("batch_size", [0, -3])
def test_non_positive_batch_size_is_rejected_at_construction(batch_size: int) -> None:
    with pytest.raises(ValueError):
        HTTPMemoryReranker(
            "http://scoring.local/rerank",
            client=httpx.AsyncClient(),
            batch_size=batch_size,
        )


@pytest.mark.parametrize("endpoint", ["", "   "])
def test_blank_endpoint_is_rejected_at_construction(endpoint: str) -> None:
    with pytest.raises(ValueError):
        HTTPMemoryReranker(endpoint, client=httpx.AsyncClient())


def test_blank_policy_id_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        HTTPMemoryReranker(
            "http://scoring.local/rerank",
            client=httpx.AsyncClient(),
            policy_id="  ",
        )


# ---------------------------------------------------------------------------
# D. Global ordering semantics
# ---------------------------------------------------------------------------


def test_scores_sort_descending_across_batches() -> None:
    scores = dict.fromkeys(range(1, 21), 0.0)
    scores[3] = 0.95
    scores[17] = 0.90
    scores[8] = 0.85
    reranker, _requests = _make_reranker(scores, batch_size=7)

    decision = _rerank(reranker, _candidates(20), 3)

    assert decision.selected_ranks == (3, 17, 8)
    asyncio.run(reranker._client.aclose())


def test_ties_break_toward_the_original_global_rank() -> None:
    scores = dict.fromkeys(range(1, 21), 0.5)
    reranker, _requests = _make_reranker(scores, batch_size=7)

    decision = _rerank(reranker, _candidates(20), 5)

    assert decision.selected_ranks == (1, 2, 3, 4, 5)
    asyncio.run(reranker._client.aclose())


def test_cross_batch_ties_break_by_global_rank() -> None:
    scores = dict.fromkeys(range(1, 21), 0.0)
    for rank in (19, 4, 11, 6):
        scores[rank] = 0.99
    reranker, _requests = _make_reranker(scores, batch_size=7)

    decision = _rerank(reranker, _candidates(20), 4)

    assert decision.selected_ranks == (4, 6, 11, 19)
    asyncio.run(reranker._client.aclose())


def test_limit_truncates_the_global_sort() -> None:
    scores = {rank: 1.0 / rank for rank in range(1, 31)}
    reranker, _requests = _make_reranker(scores, batch_size=7)

    decision = _rerank(reranker, _candidates(30), 10)

    assert decision.selected_ranks == tuple(range(1, 11))
    assert decision.usage.requests == 5
    asyncio.run(reranker._client.aclose())


# ---------------------------------------------------------------------------
# E. Real runtime integration
# ---------------------------------------------------------------------------


class _TextCandidatePipeline:
    async def extract(self, request, /):
        from powercontext.builtin.artifacts.memory.models import MemoryEntryInput

        return tuple(
            MemoryEntryInput(kind="fact", text=source.text if hasattr(source, "text") else source.content, sources=(source,))
            for source in request.sources
        )


def _runtime_scores(count: int) -> dict[int, float]:
    return {rank: ((rank * 73) % 17) / 17 for rank in range(1, count + 1)}


def test_runtime_search_uses_http_reranker_end_to_end(tmp_path) -> None:
    from powercontext.builtin.runtime import CaptureSource
    from powercontext.builtin.scope import ScopeDraft

    async def scenario() -> None:
        scores = _runtime_scores(9)
        requests: list[dict[str, object]] = []
        transport = _service(scores, requests)
        reranker = HTTPMemoryReranker(
            "http://scoring.local/rerank",
            client=httpx.AsyncClient(transport=transport),
            batch_size=4,
            policy_id="test.http.rerank.v1",
        )
        database = SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'rerank.db'}")
        async with open_builtin_runtime(
            BuiltinConfig(database=database),
            scheduler_path=tmp_path / "scheduler.db",
            candidate_pipeline=_TextCandidatePipeline(),
            memory_reranker=reranker,
        ) as runtime:
            scope = await runtime.scopes.create(  # ty: ignore[unresolved-attribute]
                ScopeDraft(title="Rerank", summary="HTTP rerank", idempotency_key="rerank-scope")
            )
            memory = runtime.memory.for_scope(scope.scope_id)
            for rank in range(1, 10):
                await runtime.sources.for_scope(scope.scope_id).capture(
                    CaptureSource(
                        source_id=f"turn-{rank}",
                        content=f"fact number {rank}: the calibration constant is {rank * 11}",
                        metadata={},
                    )
                )
            flushed = await memory.flush()
            assert flushed.source_count == 9

            page = await memory.search(SearchMemoryRequest(query="calibration constant", mode="fts", limit=3))

            assert page.rerank is not None
            assert page.rerank.policy_id == "test.http.rerank.v1"
            assert page.rerank.usage.requests == math.ceil(9 / 4)
            coarse = page.rerank.candidate_hits
            assert len(coarse) == 9  # the full coarse pool, reranker scores all of it
            assert page.rerank.selected_ranks == decision_ranks(scores, 9, 3)
            assert page.rerank.latency_ms >= 0.0
            assert [hit.entry_id for hit in page.hits] == [coarse[rank - 1].entry_id for rank in page.rerank.selected_ranks]
            assert all(
                page.hits[i].entry_version_id == coarse[rank - 1].entry_version_id
                for i, rank in enumerate(page.rerank.selected_ranks)
            )

    asyncio.run(scenario())


def decision_ranks(scores: dict[int, float], count: int, limit: int) -> tuple[int, ...]:
    ordered = sorted(range(1, count + 1), key=lambda rank: (-scores[rank], rank))
    return tuple(ordered[:limit])


# ---------------------------------------------------------------------------
# F. Quality comparison on the local SciFact approximations
# ---------------------------------------------------------------------------


def _load_variant(name: str) -> tuple[list[dict[str, object]], dict[int, dict[str, str]], dict[str, list[int]], list[int]]:
    directory = FIXTURES / name
    claims_raw = [json.loads(line) for line in (directory / "claims.jsonl").read_text(encoding="utf-8").splitlines() if line]
    claims: list[dict[str, object]] = claims_raw
    corpus: dict[int, dict[str, str]] = {}
    for line in (directory / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        doc = json.loads(line)
        corpus[int(doc["doc_id"])] = {"text": doc["text"]}
    qrels: dict[str, list[int]] = json.loads((directory / "qrels.json").read_text(encoding="utf-8"))
    pool: list[int] = json.loads((directory / "pool.json").read_text(encoding="utf-8"))
    return claims, corpus, qrels, pool


class _PoolBM25:
    def __init__(self, documents: dict[int, dict[str, str]]) -> None:
        import re
        from collections import Counter

        self._token = re.compile(r"[a-z0-9]+")
        self._tokens = {doc_id: self._token.findall(doc["text"].lower()) for doc_id, doc in documents.items()}
        self._frequencies = {doc_id: Counter(tokens) for doc_id, tokens in self._tokens.items()}
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens.values():
            document_frequency.update(set(tokens))
        self._document_frequency = document_frequency
        self._count = len(documents)
        self._average_length = sum(len(tokens) for tokens in self._tokens.values()) / max(1, self._count)

    def score(self, query: str, doc_id: int) -> float:
        tokens = self._tokens[doc_id]
        frequencies = self._frequencies[doc_id]
        total = 0.0
        for term in self._token.findall(query.lower()):
            frequency_in_corpus = self._document_frequency.get(term, 0)
            if frequency_in_corpus == 0:
                continue
            idf = math.log((self._count - frequency_in_corpus + 0.5) / (frequency_in_corpus + 0.5) + 1.0)
            frequency = frequencies.get(term, 0)
            total += idf * frequency * 2.2 / (frequency + 1.2 * (1 - 0.75 + 0.75 * len(tokens) / self._average_length))
        return total


class _PoolMiniLM:
    def __init__(self, directory: Path, claims: list[dict[str, object]]) -> None:
        import numpy as np

        data = np.load(directory / "embeddings.npz")
        self._np = np
        self._documents = {int(doc_id): vector for doc_id, vector in zip(data["doc_ids"], data["doc_vectors"], strict=True)}
        query_vectors = {
            int(claim_id): vector for claim_id, vector in zip(data["claim_ids"], data["query_vectors"], strict=True)
        }
        self._queries = {
            str(claim["claim"]): query_vectors[int(claim["id"])]  # ty: ignore[invalid-argument-type]
            for claim in claims
        }

    def score(self, query: str, doc_id: int) -> float:
        return float(self._np.dot(self._queries[query], self._documents[doc_id]))


@pytest.mark.parametrize("variant", ["k6", "k12", "k20"])
@pytest.mark.parametrize("scorer_name", ["bm25", "minilm"])
def test_quality_trend_on_scifact_approximation(variant: str, scorer_name: str) -> None:
    directory = FIXTURES / variant
    claims, corpus, qrels, pool = _load_variant(variant)
    scorer: object = _PoolBM25(corpus) if scorer_name == "bm25" else _PoolMiniLM(directory, claims)

    def score(query: str, doc_id: int, _scorer=scorer) -> float:
        return _scorer.score(query, doc_id)

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for claim in claims:
        scores = {rank: score(str(claim["claim"]), doc_id) for rank, doc_id in enumerate(pool, start=1)}
        reranker, _requests = _make_reranker(scores, batch_size=7)
        decision = _rerank(reranker, _candidates(len(pool)), 10)
        asyncio.run(reranker._client.aclose())
        top = [pool[rank - 1] for rank in decision.selected_ranks]
        relevant = set(qrels[str(claim["id"])])
        recalls.append(len(relevant & set(top)) / len(relevant))
        first = next((index + 1 for index, doc_id in enumerate(top) if doc_id in relevant), 0)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        assert decision.usage.requests == math.ceil(len(pool) / 7)

    recall = statistics.mean(recalls)
    mrr = statistics.mean(reciprocal_ranks)
    baseline_name = "bm25" if scorer_name == "bm25" else "minilm"
    baseline_file = "baseline_metrics.json" if scorer_name == "bm25" else "minilm_metrics.json"
    baseline = json.loads((directory / baseline_file).read_text(encoding="utf-8"))
    expected_recall = baseline["bm25_recall_at_10"] if scorer_name == "bm25" else baseline["recall_at_10"]
    expected_mrr = baseline["bm25_mrr_at_10"] if scorer_name == "bm25" else baseline["mrr_at_10"]
    assert math.isclose(recall, expected_recall, abs_tol=1e-9), (variant, scorer_name, recall, expected_recall)
    assert math.isclose(mrr, expected_mrr, abs_tol=1e-9), (variant, scorer_name, mrr, expected_mrr)
    assert recall >= 0.95
    print(f"quality {variant}/{scorer_name}: recall@10={recall:.4f} mrr@10={mrr:.4f} baseline_mrr={baseline_name}")
