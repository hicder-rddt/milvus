#!/usr/bin/env python3
# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a filtered ANN search with group-by on the subreddit collection."""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seed_subreddit_collection import (
    DEFAULT_COLLECTION,
    DEFAULT_ENDPOINT,
    DEFAULT_TOKEN,
    SUBREDDIT_IDS,
    VECTOR_DIM,
    MilvusRestClient,
    random_embedding,
)

DEFAULT_TOPK = 10
DEFAULT_FILTER_COUNT = 9
DEFAULT_GROUP_SIZE = 2


def build_in_filter(field: str, values: list[str]) -> str:
    quoted = ", ".join(f'"{value}"' for value in values)
    return f"{field} in [{quoted}]"


def pick_filter_subreddits(rng: random.Random, count: int) -> list[str]:
    return sorted(rng.sample(SUBREDDIT_IDS, min(count, len(SUBREDDIT_IDS))))


def search_groupby(
    client: MilvusRestClient,
    collection_name: str,
    query_vector: list[float],
    filter_expr: str,
    topk: int,
    group_size: int,
    strict_group_size: bool,
) -> dict[str, Any]:
    payload = {
        "collectionName": collection_name,
        "data": [query_vector],
        "annsField": "embeddings",
        "filter": filter_expr,
        "groupingField": "subreddit_id",
        "groupSize": group_size,
        "strictGroupSize": strict_group_size,
        "limit": topk,
        "outputFields": ["id", "subreddit_id"],
        "searchParams": {
            "metricType": "COSINE",
            "params": {"ef": 64},
            "group_by_refill": "false"
        },
    }
    return client.post("/v2/vectordb/entities/search", payload)


def print_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No results returned.")
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for hit in results:
        grouped.setdefault(hit["subreddit_id"], []).append(hit)

    print(f"Returned {len(results)} hits across {len(grouped)} subreddits:")
    for subreddit_id in sorted(grouped):
        hits = grouped[subreddit_id]
        print(f"\n  {subreddit_id} ({len(hits)} hit(s))")
        for hit in hits:
            distance = hit.get("distance", hit.get("score"))
            print(f"    id={hit['id']}  distance={distance}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run filtered ANN search with group-by on subreddit_id.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Milvus REST endpoint (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--token",
        default=DEFAULT_TOKEN,
        help="Milvus auth token",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOPK,
        help=f"Number of groups to return (default: {DEFAULT_TOPK})",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"Minimum hits per group (default: {DEFAULT_GROUP_SIZE})",
    )
    parser.add_argument(
        "--strict-group-size",
        action="store_true",
        help="Require exactly group-size hits per group",
    )
    parser.add_argument(
        "--filter-count",
        type=int,
        default=DEFAULT_FILTER_COUNT,
        help=f"Number of subreddit_ids in the IN filter (default: {DEFAULT_FILTER_COUNT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=19530,
        help="Random seed for query vector and subreddit filter pick",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON response",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = MilvusRestClient(args.endpoint, args.token)

    if not client.collection_has(args.collection):
        print(f"Collection {args.collection} does not exist.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    filter_subreddits = pick_filter_subreddits(rng, args.filter_count)
    filter_expr = build_in_filter("subreddit_id", filter_subreddits)
    query_vector = random_embedding(rng)

    print(f"Collection: {args.collection}")
    print(f"Filter: {filter_expr}")
    print(f"Query vector (dim={VECTOR_DIM}): {query_vector[:4]}... (truncated)")
    print(
        f"Search: topk={args.topk}, group_by=subreddit_id, "
        f"group_size={args.group_size}, strict={args.strict_group_size}"
    )

    response = search_groupby(
        client,
        args.collection,
        query_vector,
        filter_expr,
        args.topk,
        args.group_size,
        args.strict_group_size,
    )

    if args.json:
        print(json.dumps(response, indent=2))
        return 0

    print(f"\nSearch completed (code={response.get('code')}, cost={response.get('cost')})")
    print_results(response.get("data", []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
