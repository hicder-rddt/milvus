import pytest
from base.client_v2_base import TestMilvusClientV2Base
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel, CheckTasks
from milvus_client.expressions.expression_test_utils import (
    query_ids,
    register_collection_cleanup,
    wait_for_materialized_index,
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
    "i64_indexed": "idx_i64_inverted",
    "i64_bitmap_indexed": "idx_i64_bitmap",
    "name_indexed": "idx_name_ngram",
    "name_trie_indexed": "idx_name_trie",
    "meta_rank_indexed": "idx_meta_rank",
    "meta_group_indexed": "idx_meta_group",
    "meta_active_indexed": "idx_meta_active",
    "meta_arr_indexed": "idx_meta_arr_scores",
}


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
    name_trie = "svcX_11" if i == 11 else name
    return {
        default_pk: i,
        "i64_plain": i,
        "i64_indexed": i,
        "i64_bitmap_plain": i,
        "i64_bitmap_indexed": i,
        "name_plain": name,
        "name_indexed": name,
        "name_trie_plain": name_trie,
        "name_trie_indexed": name_trie,
        "meta_plain": meta,
        "meta_rank_indexed": meta,
        "meta_group_indexed": meta,
        "meta_active_indexed": meta,
        "meta_arr_plain": meta_arr,
        "meta_arr_indexed": meta_arr,
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
        index_params.add_index(default_vec, index_type="FLAT", metric_type="COSINE")
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
        for index_info in self.index_infos.values():
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
