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

"""Create a subreddit collection and ingest sample documents into Milvus."""

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
import uuid
from typing import Any

DEFAULT_ENDPOINT = "http://127.0.0.1:19530"
DEFAULT_TOKEN = "root:Milvus"
DEFAULT_COLLECTION = "subreddit_posts"
DEFAULT_NUM_DOCS = 10_000
DEFAULT_BATCH_SIZE = 1_000
VECTOR_DIM = 256

SUBREDDIT_IDS = [f"r/subreddit_{i:03d}" for i in range(100)]


class MilvusRestClient:
    def __init__(self, endpoint: str, token: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=json.dumps(payload).encode(),
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            body = json.loads(exc.read().decode())
        if body.get("code", 0) != 0:
            raise RuntimeError(f"{path} failed: {body}")
        return body

    def collection_has(self, collection_name: str) -> bool:
        body = self.post(
            "/v2/vectordb/collections/has",
            {"collectionName": collection_name},
        )
        return bool(body.get("data", {}).get("has", False))

    def collection_drop(self, collection_name: str) -> None:
        self.post(
            "/v2/vectordb/collections/drop",
            {"collectionName": collection_name},
        )

    def collection_create(self, collection_name: str) -> None:
        payload = {
            "collectionName": collection_name,
            "schema": {
                "autoId": False,
                "enableDynamicField": False,
                "fields": [
                    {
                        "fieldName": "id",
                        "dataType": "VarChar",
                        "isPrimary": True,
                        "elementTypeParams": {"max_length": "36"},
                    },
                    {
                        "fieldName": "embeddings",
                        "dataType": "FloatVector",
                        "elementTypeParams": {"dim": str(VECTOR_DIM)},
                    },
                    {
                        "fieldName": "subreddit_id",
                        "dataType": "VarChar",
                        "elementTypeParams": {"max_length": "64"},
                    },
                ],
            },
            "indexParams": [
                {
                    "fieldName": "embeddings",
                    "indexName": "embeddings_hnsw",
                    "metricType": "COSINE",
                    "indexType": "HNSW",
                    "params": {"M": 16, "efConstruction": 200},
                },
                {
                    "fieldName": "subreddit_id",
                    "indexName": "subreddit_id_inverted",
                    "indexType": "INVERTED",
                    "params": {"index_type": "INVERTED"},
                },
            ],
        }
        self.post("/v2/vectordb/collections/create", payload)

    def insert(self, collection_name: str, rows: list[dict[str, Any]]) -> int:
        body = self.post(
            "/v2/vectordb/entities/insert",
            {"collectionName": collection_name, "data": rows},
        )
        return int(body.get("data", {}).get("insertCount", 0))

    def collection_flush(self, collection_name: str) -> None:
        self.post(
            "/v2/vectordb/collections/flush",
            {"collectionName": collection_name},
        )

    def get_stats(self, collection_name: str) -> dict[str, Any]:
        return self.post(
            "/v2/vectordb/collections/get_stats",
            {"collectionName": collection_name},
        )


def random_embedding(rng: random.Random) -> list[float]:
    return [rng.random() for _ in range(VECTOR_DIM)]


def build_document(rng: random.Random) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "embeddings": random_embedding(rng),
        "subreddit_id": rng.choice(SUBREDDIT_IDS),
    }


def ingest_documents(
    client: MilvusRestClient,
    collection_name: str,
    num_docs: int,
    batch_size: int,
    seed: int,
) -> int:
    rng = random.Random(seed)
    inserted = 0
    for start in range(0, num_docs, batch_size):
        batch = [build_document(rng) for _ in range(min(batch_size, num_docs - start))]
        inserted += client.insert(collection_name, batch)
        print(f"Inserted {inserted}/{num_docs} documents")
    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a subreddit collection and ingest sample documents.",
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
        "--num-docs",
        type=int,
        default=DEFAULT_NUM_DOCS,
        help=f"Number of documents to insert (default: {DEFAULT_NUM_DOCS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Insert batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=19530,
        help="Random seed for reproducible embeddings and subreddit picks",
    )
    parser.add_argument(
        "--drop-if-exists",
        action="store_true",
        help="Drop the collection before creating it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = MilvusRestClient(args.endpoint, args.token)

    if client.collection_has(args.collection):
        if args.drop_if_exists:
            print(f"Dropping existing collection: {args.collection}")
            client.collection_drop(args.collection)
        else:
            print(
                f"Collection {args.collection} already exists. "
                "Use --drop-if-exists to recreate it.",
                file=sys.stderr,
            )
            return 1

    print(f"Creating collection: {args.collection}")
    client.collection_create(args.collection)

    print(f"Ingesting {args.num_docs} documents...")
    inserted = ingest_documents(
        client,
        args.collection,
        args.num_docs,
        args.batch_size,
        args.seed,
    )

    print(f"Flushing collection: {args.collection}")
    client.collection_flush(args.collection)

    stats = client.get_stats(args.collection)
    row_count = stats.get("data", {}).get("rowCount", "unknown")
    print(f"Done. Inserted {inserted} documents. Collection row count: {row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
