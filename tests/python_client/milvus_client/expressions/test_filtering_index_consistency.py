import pytest
from base.client_v2_base import TestMilvusClientV2Base
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel, CheckTasks
from milvus_client.expressions.expression_test_utils import (
    query_ids,
    register_collection_cleanup,
    wait_for_materialized_index,
    wait_for_segment_mode,
)
from milvus_client.expressions.filtering_case_matrix import (
    INDEX_CONSISTENCY_CASES,
    INDEX_NEGATIVE_ERROR_CASES,
    REAL_INDEX_ROW_COUNT,
)
from pymilvus import DataType

default_pk = "id"
default_vec = "vector"
default_dim = 8

INDEX_NAMES = {
    default_vec: "idx_vector_hnsw",
    "i64_indexed": "idx_i64_inverted",
    "i64_bitmap_indexed": "idx_i64_bitmap",
    "name_indexed": "idx_name_ngram",
    "name_trie_indexed": "idx_name_trie",
    "meta_rank_indexed": "idx_meta_rank",
    "meta_group_indexed": "idx_meta_group",
    "meta_active_indexed": "idx_meta_active",
    "meta_arr_indexed": "idx_meta_arr_scores",
}


def vector_for_id(row_id):
    return [float(((row_id + 1) * (dimension + 5)) % 23 + 1) / 25.0 for dimension in range(default_dim)]


def hit_id(hit):
    if default_pk in hit:
        return hit[default_pk]
    entity = hit.get("entity", {})
    if default_pk in entity:
        return entity[default_pk]
    return hit["id"]


def index_case_params(field_types):
    params = []
    for case in INDEX_CONSISTENCY_CASES:
        if case["field_type"] not in field_types:
            continue
        level = CaseLabel.L2 if case["index_type"] == "NGRAM" else CaseLabel.L1
        params.append(pytest.param(case, marks=pytest.mark.tags(level), id=case["case_name"]))
    return params


L2_INDEX_NEGATIVE_ERROR_CASE_NAMES = {
    "json_path_index_missing_cast_type",
    "json_path_index_unsupported_array_int64_cast_type",
    "ngram_json_path_missing_cast_type",
    "ngram_min_greater_than_max",
}


def index_negative_error_params():
    params = []
    for case in INDEX_NEGATIVE_ERROR_CASES:
        case_name = case[0]
        level = CaseLabel.L2 if case_name in L2_INDEX_NEGATIVE_ERROR_CASE_NAMES else CaseLabel.L1
        params.append(pytest.param(*case, marks=pytest.mark.tags(level), id=case_name))
    return params


def build_index_consistency_schema(testcase, client):
    schema = testcase.create_schema(client, auto_id=False, enable_dynamic_field=False)[0]
    schema.add_field(default_pk, DataType.INT64, is_primary=True)
    schema.add_field(default_vec, DataType.FLOAT_VECTOR, dim=default_dim, nullable=True)
    schema.add_field("i64_plain", DataType.INT64)
    schema.add_field("i64_indexed", DataType.INT64)
    schema.add_field("i64_bitmap_plain", DataType.INT64)
    schema.add_field("i64_bitmap_indexed", DataType.INT64)
    schema.add_field("name_plain", DataType.VARCHAR, max_length=64)
    schema.add_field("name_indexed", DataType.VARCHAR, max_length=64)
    schema.add_field("name_trie_plain", DataType.VARCHAR, max_length=64)
    schema.add_field("name_trie_indexed", DataType.VARCHAR, max_length=64)
    schema.add_field("meta_plain", DataType.JSON)
    schema.add_field("meta_rank_indexed", DataType.JSON)
    schema.add_field("meta_group_indexed", DataType.JSON)
    schema.add_field("meta_active_indexed", DataType.JSON)
    schema.add_field("meta_arr_plain", DataType.JSON)
    schema.add_field("meta_arr_indexed", DataType.JSON)
    return schema


def make_index_consistency_row(i):
    if i <= 10:
        group = "qa" if i in {1, 4, 7, 10} else "dev" if i in {2, 5, 8} else "ops"
        active = i in {1, 2, 4}
        name = f"svc_{i}" if i % 2 == 0 else f"account_{i}"
        meta = {"rank": i, "group": group, "active": active}
        meta_arr = {"scores": [float(i), float(i) + 10.0]}
    else:
        name = f"account_{i}"
        meta = {"rank": 0, "group": "filler", "active": False}
        meta_arr = {"scores": [0.0]}
    name_trie = "svcX_11" if i == 11 else (f"svc_trie_{i}" if i in {2, 4, 6, 8, 10} else f"account_trie_{i}")
    i64_value = i * 10 + 1
    bitmap_value = 10000 + i
    return {
        default_pk: i,
        default_vec: vector_for_id(i),
        "i64_plain": i64_value,
        "i64_indexed": i64_value,
        "i64_bitmap_plain": bitmap_value,
        "i64_bitmap_indexed": bitmap_value,
        "name_plain": name,
        "name_indexed": name,
        "name_trie_plain": name_trie,
        "name_trie_indexed": name_trie,
        "meta_plain": meta,
        "meta_rank_indexed": {"rank": meta["rank"], "group": "rank_decoy", "active": False},
        "meta_group_indexed": {"rank": -1000, "group": meta["group"], "active": False},
        "meta_active_indexed": {"rank": -2000, "group": "active_decoy", "active": meta["active"]},
        "meta_arr_plain": meta_arr,
        "meta_arr_indexed": {"scores": meta_arr["scores"], "rank": -3000},
    }


@pytest.mark.xdist_group("TestFilteringIndexConsistency")
class TestFilteringIndexConsistency(TestMilvusClientV2Base):
    shared_alias = "TestFilteringIndexConsistency"

    @pytest.fixture(scope="class", autouse=True)
    def prepare_index_consistency_collection(self, request):
        client = self._client(alias=self.shared_alias)
        collection_name = "filter_index_consistency" + cf.gen_unique_str("_")
        self.create_collection(
            client,
            collection_name,
            schema=build_index_consistency_schema(self, client),
            force_teardown=False,
            consistency_level="Strong",
        )
        register_collection_cleanup(self, request, self.shared_alias, collection_name)
        self.insert(
            client,
            collection_name,
            data=[make_index_consistency_row(i) for i in range(1, REAL_INDEX_ROW_COUNT + 1)],
        )
        self.flush(client, collection_name)

        index_params = self.prepare_index_params(client)[0]
        index_params.add_index(
            default_vec,
            index_type="HNSW",
            index_name=INDEX_NAMES[default_vec],
            metric_type="COSINE",
            params={"M": 8, "efConstruction": 64},
        )
        index_params.add_index(
            "i64_indexed",
            index_type="INVERTED",
            index_name=INDEX_NAMES["i64_indexed"],
        )
        index_params.add_index(
            "i64_bitmap_indexed",
            index_type="BITMAP",
            index_name=INDEX_NAMES["i64_bitmap_indexed"],
        )
        index_params.add_index(
            "name_indexed",
            index_type="NGRAM",
            index_name=INDEX_NAMES["name_indexed"],
            params={"min_gram": 2, "max_gram": 4},
        )
        index_params.add_index(
            "name_trie_indexed",
            index_type="TRIE",
            index_name=INDEX_NAMES["name_trie_indexed"],
        )
        index_params.add_index(
            "meta_rank_indexed",
            index_type="INVERTED",
            index_name=INDEX_NAMES["meta_rank_indexed"],
            params={
                "json_path": "meta_rank_indexed['rank']",
                "json_cast_type": "DOUBLE",
            },
        )
        index_params.add_index(
            "meta_group_indexed",
            index_type="INVERTED",
            index_name=INDEX_NAMES["meta_group_indexed"],
            params={
                "json_path": "meta_group_indexed['group']",
                "json_cast_type": "VARCHAR",
            },
        )
        index_params.add_index(
            "meta_active_indexed",
            index_type="INVERTED",
            index_name=INDEX_NAMES["meta_active_indexed"],
            params={
                "json_path": "meta_active_indexed['active']",
                "json_cast_type": "BOOL",
            },
        )
        index_params.add_index(
            "meta_arr_indexed",
            index_type="INVERTED",
            index_name=INDEX_NAMES["meta_arr_indexed"],
            params={
                "json_path": "meta_arr_indexed['scores']",
                "json_cast_type": "ARRAY_DOUBLE",
            },
        )
        self.create_index(client, collection_name, index_params=index_params)
        self.__class__.index_infos = {
            field_name: wait_for_materialized_index(
                self,
                client,
                collection_name,
                index_name,
                expected_indexed_rows=REAL_INDEX_ROW_COUNT,
            )
            for field_name, index_name in INDEX_NAMES.items()
        }
        self.load_collection(client, collection_name)
        self.__class__.collection_name = collection_name
        yield

    def assert_index_consistency_case(self, case):
        client = self._client(alias=self.shared_alias)
        plain_ids = query_ids(self, client, self.collection_name, case["plain_expr"], pk_field=default_pk)
        indexed_ids = query_ids(self, client, self.collection_name, case["indexed_expr"], pk_field=default_pk)
        assert plain_ids == case["expected_ids"], (
            f"{case['case_name']} plain oracle mismatch: expected={case['expected_ids']}, actual={plain_ids}"
        )
        assert indexed_ids == case["expected_ids"], (
            f"{case['case_name']} indexed oracle mismatch: expected={case['expected_ids']}, actual={indexed_ids}"
        )
        assert plain_ids == indexed_ids, (
            f"{case['case_name']} index consistency mismatch: plain={plain_ids}, indexed={indexed_ids}"
        )

    @pytest.mark.tags(CaseLabel.L1)
    def test_scalar_indexes_are_materialized(self):
        assert set(self.index_infos) == set(INDEX_NAMES)
        for field_name, index_info in self.index_infos.items():
            assert index_info["field_name"] == field_name
            assert index_info["index_name"] == INDEX_NAMES[field_name]
            assert index_info["total_rows"] == REAL_INDEX_ROW_COUNT
            assert index_info["indexed_rows"] >= REAL_INDEX_ROW_COUNT
            assert index_info["pending_index_rows"] == 0

    @pytest.mark.parametrize("case", index_case_params({"INT64", "VARCHAR"}))
    def test_scalar_index_consistency(self, case):
        self.assert_index_consistency_case(case)

    @pytest.mark.parametrize("case", index_case_params({"JSON"}))
    def test_json_path_index_consistency(self, case):
        self.assert_index_consistency_case(case)

    @pytest.mark.parametrize("case", index_case_params({"JSON_ARRAY"}))
    def test_json_path_array_index_consistency(self, case):
        self.assert_index_consistency_case(case)

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize(
        "expr, expected_ids",
        [
            pytest.param("i64_indexed in [31, 51, 71]", [3, 5, 7], id="int64_indexed_prefilter"),
            pytest.param('meta_group_indexed["group"] == "qa"', [1, 4, 7, 10], id="json_indexed_prefilter"),
        ],
    )
    def test_materialized_index_search_prefilter(self, expr, expected_ids):
        client = self._client(alias=self.shared_alias)
        result = client.search(
            self.collection_name,
            data=[vector_for_id(4)],
            anns_field=default_vec,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            filter=expr,
            output_fields=[default_pk],
            limit=10,
        )
        assert len(result) == 1
        hits = result[0]
        assert sorted(hit_id(hit) for hit in hits) == expected_ids
        distances = [hit["distance"] for hit in hits]
        assert distances == sorted(distances, reverse=True)


@pytest.mark.xdist_group("TestFilteringIndexedMixedSegments")
class TestFilteringIndexedMixedSegments(TestMilvusClientV2Base):
    shared_alias = "TestFilteringIndexedMixedSegments"
    vector_index_name = "idx_mixed_vector_hnsw"
    scalar_index_name = "idx_mixed_i64_inverted"

    @pytest.fixture(scope="class")
    def indexed_mixed_collection(self, request):
        client = self._client(alias=self.shared_alias)
        collection_name = "filter_indexed_mixed" + cf.gen_unique_str("_")
        schema = self.create_schema(client, auto_id=False, enable_dynamic_field=False)[0]
        schema.add_field(default_pk, DataType.INT64, is_primary=True)
        schema.add_field(default_vec, DataType.FLOAT_VECTOR, dim=default_dim)
        schema.add_field("i64_plain", DataType.INT64)
        schema.add_field("i64_indexed", DataType.INT64)
        self.create_collection(
            client,
            collection_name,
            schema=schema,
            force_teardown=False,
            consistency_level="Strong",
        )
        register_collection_cleanup(self, request, self.shared_alias, collection_name)
        sealed_rows = [
            {
                default_pk: row_id,
                default_vec: vector_for_id(row_id),
                "i64_plain": row_id * 10 + 1,
                "i64_indexed": row_id * 10 + 1,
            }
            for row_id in range(1, REAL_INDEX_ROW_COUNT + 1)
        ]
        self.insert(client, collection_name, data=sealed_rows)
        self.flush(client, collection_name)
        index_params = self.prepare_index_params(client)[0]
        index_params.add_index(
            default_vec,
            index_name=self.vector_index_name,
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 8, "efConstruction": 64},
        )
        index_params.add_index(
            "i64_indexed",
            index_name=self.scalar_index_name,
            index_type="INVERTED",
        )
        self.create_index(client, collection_name, index_params=index_params)
        self.__class__.mixed_index_infos = {
            index_name: wait_for_materialized_index(
                self,
                client,
                collection_name,
                index_name,
                expected_indexed_rows=REAL_INDEX_ROW_COUNT,
            )
            for index_name in (self.vector_index_name, self.scalar_index_name)
        }
        self.load_collection(client, collection_name)
        growing_rows = [
            {default_pk: 3001, default_vec: vector_for_id(3001), "i64_plain": 31, "i64_indexed": 31},
            {default_pk: 3002, default_vec: vector_for_id(3002), "i64_plain": 999999, "i64_indexed": 999999},
            {default_pk: 3003, default_vec: vector_for_id(3003), "i64_plain": 51, "i64_indexed": 51},
        ]
        self.insert(client, collection_name, data=growing_rows)
        wait_for_segment_mode(
            client,
            collection_name,
            "mixed",
            expected_row_count=REAL_INDEX_ROW_COUNT + len(growing_rows),
            expected_sealed_rows=REAL_INDEX_ROW_COUNT,
        )
        yield collection_name

    @pytest.mark.tags(CaseLabel.L2)
    def test_materialized_index_and_growing_scan_merge(self, indexed_mixed_collection):
        client = self._client(alias=self.shared_alias)
        expected_fields = {
            self.vector_index_name: default_vec,
            self.scalar_index_name: "i64_indexed",
        }
        assert set(self.mixed_index_infos) == set(expected_fields)
        for index_name, index_info in self.mixed_index_infos.items():
            assert index_info["index_name"] == index_name
            assert index_info["field_name"] == expected_fields[index_name]
            assert index_info["total_rows"] == REAL_INDEX_ROW_COUNT
            assert index_info["indexed_rows"] >= REAL_INDEX_ROW_COUNT
            assert index_info["pending_index_rows"] == 0

        plain_ids = query_ids(self, client, indexed_mixed_collection, "i64_plain in [31, 51]")
        indexed_ids = query_ids(self, client, indexed_mixed_collection, "i64_indexed in [31, 51]")
        assert plain_ids == [3, 5, 3001, 3003]
        assert indexed_ids == plain_ids

        result = client.search(
            indexed_mixed_collection,
            data=[vector_for_id(3001)],
            anns_field=default_vec,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            filter="i64_indexed in [31, 51]",
            output_fields=[default_pk],
            limit=10,
        )
        assert len(result) == 1
        hits = result[0]
        assert sorted(hit_id(hit) for hit in hits) == plain_ids
        distances = [hit["distance"] for hit in hits]
        assert distances == sorted(distances, reverse=True)


@pytest.mark.xdist_group("TestFilteringIndexNegative")
class TestFilteringIndexNegative(TestMilvusClientV2Base):
    shared_alias = "TestFilteringIndexNegative"

    @pytest.fixture(scope="class")
    def negative_index_collection(self, request):
        client = self._client(alias=self.shared_alias)
        collection_name = "filter_index_negative" + cf.gen_unique_str("_")
        schema = self.create_schema(client, auto_id=False, enable_dynamic_field=False)[0]
        schema.add_field(default_pk, DataType.INT64, is_primary=True)
        schema.add_field(default_vec, DataType.FLOAT_VECTOR, dim=default_dim, nullable=True)
        schema.add_field("name_indexed", DataType.VARCHAR, max_length=64)
        schema.add_field("meta_indexed", DataType.JSON)
        self.create_collection(
            client,
            collection_name,
            schema=schema,
            force_teardown=False,
            consistency_level="Strong",
        )
        register_collection_cleanup(self, request, self.shared_alias, collection_name)
        yield collection_name

    @pytest.mark.parametrize(
        "case_name, index_spec, expected_message_substring",
        index_negative_error_params(),
    )
    def test_index_negative_meaningful_error_cases(
        self,
        negative_index_collection,
        case_name,
        index_spec,
        expected_message_substring,
    ):
        client = self._client(alias=self.shared_alias)
        index_params = self.prepare_index_params(client)[0]
        index_params.add_index(**index_spec)
        self.create_index(
            client,
            negative_index_collection,
            index_params=index_params,
            check_task=CheckTasks.err_res,
            check_items={ct.err_msg: expected_message_substring},
        )
