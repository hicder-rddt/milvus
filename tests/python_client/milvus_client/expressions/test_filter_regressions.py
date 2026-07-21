from collections import Counter

import pytest
from base.client_v2_base import TestMilvusClientV2Base
from common import common_func as cf
from common.common_type import CaseLabel
from milvus_client.expressions.expression_test_utils import (
    add_minimal_query_vector_field,
    assert_query_ids,
    assert_same_query_result,
    create_minimal_vector_index,
    insert_by_segment_mode,
    prepare_loaded_empty_collection_for_segment,
    register_collection_cleanup,
    wait_for_segment_mode,
)
from milvus_client.expressions.filtering_case_matrix import (
    BOOLEAN_FANOUT_EXPRESSIONS_L1,
    BOOLEAN_FANOUT_EXPRESSIONS_L2,
    EQUIVALENT_EXPRESSION_CASES,
    JSON_BOOL_MIXED_51567_CONTROL_CASES,
    JSON_BOOL_MIXED_IN_51567_CASES,
    JSON_BOOL_MIXED_OR_51567_CASES,
    JSON_MIXED_TYPE_IN_51489_CASES,
    JSON_MIXED_TYPE_OR_51568_CASES,
    ORDER_SENSITIVE_EXPRESSIONS,
    SAME_FIELD_OR_FANOUT_EXPRESSIONS_L1,
    SEGMENT_MODES_ACTIVE,
)
from pymilvus import DataType, MilvusException

default_pk = "id"
default_vec = "vector"
default_dim = 8

KNOWN_ISSUE_51568_LOSS_IDS = {
    "float_int_or_5": [1, 2, 5],
    "float_middle_int_or_5": [1, 3, 5],
}
KNOWN_ISSUE_51568_ERROR_VALUE_CASES = {
    "float_int_or_4": "kInt64Val",
    "float_int_or_alternating_6": "kInt64Val",
    "float_int_or_20": "kInt64Val",
    "str_int_or_3": "kStringVal",
}
KNOWN_ISSUE_51489_IN_ERROR_VALUE_CASES = {
    "int_string_in": "kStringVal",
    "string_int_in": "kInt64Val",
    "int_unrelated_string_in": "kStringVal",
    "json_array_subscript_mixed_in": "kStringVal",
}
KNOWN_ISSUE_51567_IN_ERROR_VALUE_CASES = {
    "bool_int_in_true_one": "kInt64Val",
    "bool_string_in": "kStringVal",
    "bool_int_string_in": "kInt64Val",
}
KNOWN_ISSUE_51567_IN_SUCCESS_IDS = {
    "bool_int_in_false_one": [3, 4],
}
KNOWN_ISSUE_51567_OR_LOSS_IDS = {
    "bool_int_or_bool_first_three": [5, 6],
    "bool_int_or_bool_last_three": [5, 6],
    "bool_int_or_false_first_three": [5, 6],
    "bool_int_or_false_last_three": [5, 6],
}
KNOWN_ISSUE_51567_OR_ERROR_VALUE_CASES = {
    "bool_string_or_three": "kBoolVal",
}
SEGMENT_51568_CASES = [case for case in JSON_MIXED_TYPE_OR_51568_CASES if case[0] == "float_int_or_4"]
SEARCH_51568_CASES = [
    case for case in JSON_MIXED_TYPE_OR_51568_CASES if case[0] in {"float_int_or_4", "float_last_int_or_5"}
]
SEGMENT_51568_MODES = ["growing", "mixed"]
L0_51568_CASE_NAMES = {"pure_int_in_5_control", "float_int_or_4"}
L0_51489_IN_CASE_NAMES = {"int_string_in"}
L0_51567_BOOL_IN_CASE_NAMES = {"bool_int_in_true_one"}
L0_51567_BOOL_OR_CASE_NAMES = {"bool_int_or_int_first_three", "bool_int_or_bool_first_three"}


def fanout_params(cases):
    return [pytest.param(*case, id=case[0]) for case in cases]


def order_params(cases):
    params = []
    for i, case in enumerate(cases, start=1):
        level = CaseLabel.L0 if i <= 2 else CaseLabel.L1
        params.append(pytest.param(*case, marks=pytest.mark.tags(level), id=f"order_{i}"))
    return params


def equivalent_expression_params(cases):
    return [pytest.param(*case, id=case[0]) for case in cases]


def hit_id(hit):
    if default_pk in hit:
        return hit[default_pk]
    entity = hit.get("entity", {})
    if default_pk in entity:
        return entity[default_pk]
    return hit["id"]


def sorted_hit_ids(hits):
    return sorted(hit_id(hit) for hit in hits)


def id_delta(actual_ids, expected_ids):
    actual_counter = Counter(actual_ids)
    expected_counter = Counter(expected_ids)
    missing = sorted((expected_counter - actual_counter).elements())
    extra = sorted((actual_counter - expected_counter).elements())
    return missing, extra


def is_known_executor_type_assertion(exc, expected_value_case):
    message = str(exc)
    return exc.code == 2000 and all(
        token in message
        for token in (
            "Operator:PhyFilterBitsNode",
            "value_proto.val_case()",
            f"GenericValue::{expected_value_case}",
            "internal/core/src/exec/expression/Utils.h:",
            "segcoreCode=2001",
        )
    )


def assert_ids_or_xfail_known_loss(actual_ids, expected_ids, case_name, known_actual_ids, issue, context):
    actual_ids = sorted(actual_ids)
    expected_ids = sorted(expected_ids)
    if actual_ids == expected_ids:
        return

    missing, extra = id_delta(actual_ids, expected_ids)
    if case_name in known_actual_ids and actual_ids == sorted(known_actual_ids[case_name]):
        pytest.xfail(
            f"Known issue {issue}: {context} reproduced exact result {actual_ids}, missing={missing}, extra={extra}"
        )

    assert actual_ids == expected_ids, (
        f"{context} got {actual_ids}, expected {expected_ids}, missing={missing}, extra={extra}"
    )


def query_ids_or_xfail_known_loss(
    client,
    collection_name,
    expr,
    expected_ids,
    case_name,
    known_actual_ids,
    known_error_value_cases,
    issue,
    context,
):
    try:
        rows = client.query(collection_name, filter=expr, output_fields=[default_pk])
    except MilvusException as exc:
        expected_value_case = known_error_value_cases.get(case_name)
        if expected_value_case and is_known_executor_type_assertion(exc, expected_value_case):
            pytest.xfail(f"Known issue {issue}: {context} failed with {exc}")
        raise

    actual_ids = [row[default_pk] for row in rows]
    assert_ids_or_xfail_known_loss(actual_ids, expected_ids, case_name, known_actual_ids, issue, context)


def search_ids_or_xfail_known_loss(
    client,
    collection_name,
    expr,
    expected_ids,
    case_name,
    known_actual_ids,
    known_error_value_cases,
    issue,
    context,
):
    try:
        hits = client.search(
            collection_name,
            data=[[0.1] * default_dim],
            anns_field=default_vec,
            search_params={"metric_type": "COSINE", "params": {}},
            filter=expr,
            output_fields=[default_pk],
            limit=30,
        )[0]
    except MilvusException as exc:
        expected_value_case = known_error_value_cases.get(case_name)
        if expected_value_case and is_known_executor_type_assertion(exc, expected_value_case):
            pytest.xfail(f"Known issue {issue}: {context} failed with {exc}")
        raise

    assert_ids_or_xfail_known_loss(
        sorted_hit_ids(hits),
        expected_ids,
        case_name,
        known_actual_ids,
        issue,
        context,
    )


def result_ids(result):
    rows = result[0] if result and isinstance(result[0], list) else result
    return sorted(hit_id(row) for row in rows)


def assert_meaningful_type_mismatch_rejection(
    call,
    known_issue=None,
    known_executor_value_case=None,
    known_success_ids=None,
):
    try:
        result = call()
    except MilvusException as exc:
        message = str(exc).lower()
        has_meaningful_message = any(
            substring in message
            for substring in (
                "value type mismatch",
                "type mismatch",
                "cannot be casted",
                "mixed types",
            )
        )
        reached_executor = "phyfilterbitsnode" in message or "segcore" in message
        if has_meaningful_message and not reached_executor:
            return
        if (
            known_issue
            and known_executor_value_case
            and is_known_executor_type_assertion(exc, known_executor_value_case)
        ):
            pytest.xfail(f"Known issue {known_issue}: expected planner rejection, got {exc}")
        assert has_meaningful_message, f"expected a meaningful type-mismatch error, got: {exc}"
        assert not reached_executor, f"mixed types reached segcore instead of planner rejection: {exc}"
        return

    actual_ids = result_ids(result)
    if known_issue and known_success_ids is not None and actual_ids == sorted(known_success_ids):
        pytest.xfail(f"Known issue {known_issue}: mixed-type JSON IN returned exact known IDs {actual_ids}")
    pytest.fail(f"mixed-type JSON IN unexpectedly succeeded with IDs {actual_ids}; planner rejection is required")


def mark_51568_cases(cases):
    params = []
    for case in cases:
        case_name = case[0]
        level = CaseLabel.L0 if case_name in L0_51568_CASE_NAMES else CaseLabel.L1
        marks = [pytest.mark.tags(level)]
        params.append(pytest.param(*case, marks=marks, id=case_name))
    return params


def mark_51568_segment_cases(cases, segment_modes):
    params = []
    for case in cases:
        case_name = case[0]
        for segment_mode in segment_modes:
            marks = [pytest.mark.tags(CaseLabel.L1)]
            params.append(
                pytest.param(
                    *case,
                    segment_mode,
                    marks=marks,
                    id=f"{case_name}-{segment_mode}",
                )
            )
    return params


def mark_json_mixed_type_in_cases(cases):
    params = []
    for case in cases:
        case_name = case[0]
        level = CaseLabel.L0 if case_name in L0_51489_IN_CASE_NAMES else CaseLabel.L1
        if case_name == "json_array_subscript_mixed_in":
            level = CaseLabel.L2
        params.append(pytest.param(*case, marks=pytest.mark.tags(level), id=case_name))
    return params


def mark_json_bool_mixed_in_cases(cases):
    params = []
    for case in cases:
        case_name = case[0]
        level = CaseLabel.L0 if case_name in L0_51567_BOOL_IN_CASE_NAMES else CaseLabel.L1
        if case_name == "bool_int_string_in":
            level = CaseLabel.L2
        params.append(pytest.param(*case, marks=pytest.mark.tags(level), id=case_name))
    return params


def mark_51567_or_cases(cases):
    params = []
    for case in cases:
        case_name = case[0]
        level = CaseLabel.L0 if case_name in L0_51567_BOOL_OR_CASE_NAMES else CaseLabel.L1
        marks = [pytest.mark.tags(level)]
        params.append(pytest.param(*case, marks=marks, id=case_name))
    return params


def build_order_rows():
    return [
        {
            default_pk: 1,
            "age": 8,
            "score": 95.0,
            "active": True,
            "tag": "qa",
            "meta": {"group": "qa", "rank": 1, "p": 1},
        },
        {
            default_pk: 2,
            "age": 12,
            "score": 88.0,
            "active": True,
            "tag": "qa",
            "meta": {"group": "qa", "rank": 1, "p": 2},
        },
        {
            default_pk: 3,
            "age": 13,
            "score": 89.0,
            "active": False,
            "tag": "dev",
            "meta": {"group": "dev", "rank": 3, "p": 3},
        },
        {
            default_pk: 4,
            "age": 14,
            "score": 80.0,
            "active": True,
            "tag": "qa",
            "meta": {"group": "qa", "rank": 2, "p": 4},
        },
        {
            default_pk: 5,
            "age": 15,
            "score": 91.0,
            "active": False,
            "tag": "ops",
            "meta": {"group": "ops", "rank": 5, "p": 5},
        },
        {
            default_pk: 6,
            "age": 16,
            "score": 70.0,
            "active": False,
            "tag": "ops",
            "meta": {"group": "ops", "rank": 6, "p": 6},
        },
        {
            default_pk: 7,
            "age": 17,
            "score": 75.0,
            "active": False,
            "tag": "dev",
            "meta": {"group": "dev", "rank": 7, "p": 7},
        },
        {
            default_pk: 8,
            "age": 18,
            "score": 76.0,
            "active": False,
            "tag": "dev",
            "meta": {"group": "dev", "rank": 8, "p": 8},
        },
        {
            default_pk: 9,
            "age": 19,
            "score": 77.0,
            "active": False,
            "tag": "dev",
            "meta": {"group": "dev", "rank": 9, "p": 9},
        },
        {
            default_pk: 10,
            "age": 20,
            "score": 78.0,
            "active": False,
            "tag": "dev",
            "meta": {"group": "dev", "rank": 10, "p": 10},
        },
    ]


@pytest.mark.xdist_group("TestFilterRegressions")
class TestFilterRegressions(TestMilvusClientV2Base):
    shared_alias = "TestFilterRegressions"

    def build_order_schema(self, client):
        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field(default_pk, DataType.INT64, is_primary=True, auto_id=False)
        add_minimal_query_vector_field(schema, vector_field=default_vec, dim=default_dim)
        schema.add_field("age", DataType.INT64)
        schema.add_field("score", DataType.DOUBLE)
        schema.add_field("active", DataType.BOOL)
        schema.add_field("tag", DataType.VARCHAR, max_length=64)
        schema.add_field("meta", DataType.JSON)
        return schema

    def build_51568_schema(self, client):
        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field(default_pk, DataType.INT64, is_primary=True, auto_id=False)
        add_minimal_query_vector_field(schema, vector_field=default_vec, dim=default_dim)
        schema.add_field("meta", DataType.JSON)
        return schema

    @pytest.fixture(scope="class")
    def regression_51568_collection(self, request):
        client = self._client(alias=self.shared_alias)
        collection_name = "filter_regression_51568" + cf.gen_unique_str("_")
        self.create_collection(
            client,
            collection_name,
            schema=self.build_51568_schema(client),
            force_teardown=False,
            consistency_level="Strong",
        )
        register_collection_cleanup(self, request, self.shared_alias, collection_name)
        bool_mixed_values = {
            1: True,
            2: True,
            3: False,
            4: False,
            5: 0,
            6: 1,
            7: "yes",
            8: "no",
        }
        rows = []
        for i in range(1, 21):
            meta = {"p": i, "arr": [i, i + 10]}
            if i in bool_mixed_values:
                meta["b"] = bool_mixed_values[i]
            rows.append(
                {
                    default_pk: i,
                    default_vec: [float(i) / 20.0] * default_dim,
                    "meta": meta,
                }
            )
        self.insert(client, collection_name, data=rows)
        self.flush(client, collection_name)
        create_minimal_vector_index(self, client, collection_name, vector_field=default_vec)
        self.load_collection(client, collection_name)
        yield collection_name

    @pytest.fixture(scope="class")
    def order_fanout_collection(self, request):
        client = self._client(alias=self.shared_alias)
        collection_name = "filter_order" + cf.gen_unique_str("_")
        self.create_collection(
            client,
            collection_name,
            schema=self.build_order_schema(client),
            force_teardown=False,
            consistency_level="Strong",
        )
        register_collection_cleanup(self, request, self.shared_alias, collection_name)
        self.insert(client, collection_name, data=build_order_rows())
        self.flush(client, collection_name)
        create_minimal_vector_index(self, client, collection_name, vector_field=default_vec)
        self.load_collection(client, collection_name)
        yield collection_name

    @pytest.mark.parametrize(
        "case_name, expr, fanout_count, expected_ids",
        mark_51568_cases(JSON_MIXED_TYPE_OR_51568_CASES),
    )
    def test_json_same_path_mixed_type_or_regression_51568(
        self,
        regression_51568_collection,
        case_name,
        expr,
        fanout_count,
        expected_ids,
    ):
        client = self._client(alias=self.shared_alias)
        query_ids_or_xfail_known_loss(
            client,
            regression_51568_collection,
            expr,
            expected_ids,
            case_name,
            KNOWN_ISSUE_51568_LOSS_IDS,
            KNOWN_ISSUE_51568_ERROR_VALUE_CASES,
            "https://github.com/milvus-io/milvus/issues/51568",
            f"{case_name} query",
        )

    @pytest.mark.parametrize(
        "case_name, expr",
        mark_json_mixed_type_in_cases(JSON_MIXED_TYPE_IN_51489_CASES),
    )
    def test_json_mixed_type_in_rejected_at_planner_51489(
        self,
        regression_51568_collection,
        case_name,
        expr,
    ):
        client = self._client(alias=self.shared_alias)
        assert_meaningful_type_mismatch_rejection(
            lambda: client.query(
                regression_51568_collection,
                filter=expr,
                output_fields=[default_pk],
            ),
            known_issue="https://github.com/milvus-io/milvus/issues/51489",
            known_executor_value_case=KNOWN_ISSUE_51489_IN_ERROR_VALUE_CASES[case_name],
        )

    @pytest.mark.tags(CaseLabel.L1)
    def test_json_mixed_type_in_search_rejected_at_planner_51489(self, regression_51568_collection):
        client = self._client(alias=self.shared_alias)
        assert_meaningful_type_mismatch_rejection(
            lambda: client.search(
                regression_51568_collection,
                data=[[0.1] * default_dim],
                anns_field=default_vec,
                search_params={"metric_type": "COSINE", "params": {}},
                filter='meta["p"] in [1, "2"]',
                output_fields=[default_pk],
                limit=20,
            ),
            known_issue="https://github.com/milvus-io/milvus/issues/51489",
            known_executor_value_case=KNOWN_ISSUE_51489_IN_ERROR_VALUE_CASES["int_string_in"],
        )

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize(
        "case_name, expr, expected_ids",
        [pytest.param(*case, id=case[0]) for case in JSON_BOOL_MIXED_51567_CONTROL_CASES],
    )
    def test_json_bool_mixed_type_controls_51567(
        self,
        regression_51568_collection,
        case_name,
        expr,
        expected_ids,
    ):
        client = self._client(alias=self.shared_alias)
        assert_query_ids(self, client, regression_51568_collection, expr, expected_ids, pk_field=default_pk)

    @pytest.mark.parametrize(
        "case_name, expr",
        mark_json_bool_mixed_in_cases(JSON_BOOL_MIXED_IN_51567_CASES),
    )
    def test_json_bool_mixed_type_in_rejected_at_planner_51567(
        self,
        regression_51568_collection,
        case_name,
        expr,
    ):
        client = self._client(alias=self.shared_alias)
        assert_meaningful_type_mismatch_rejection(
            lambda: client.query(
                regression_51568_collection,
                filter=expr,
                output_fields=[default_pk],
            ),
            known_issue="https://github.com/milvus-io/milvus/issues/51567",
            known_executor_value_case=KNOWN_ISSUE_51567_IN_ERROR_VALUE_CASES.get(case_name),
            known_success_ids=KNOWN_ISSUE_51567_IN_SUCCESS_IDS.get(case_name),
        )

    @pytest.mark.tags(CaseLabel.L1)
    def test_json_bool_mixed_type_in_search_rejected_at_planner_51567(self, regression_51568_collection):
        client = self._client(alias=self.shared_alias)
        case_name = "bool_int_in_false_one"
        assert_meaningful_type_mismatch_rejection(
            lambda: client.search(
                regression_51568_collection,
                data=[[0.1] * default_dim],
                anns_field=default_vec,
                search_params={"metric_type": "COSINE", "params": {}},
                filter='meta["b"] in [false, 1]',
                output_fields=[default_pk],
                limit=20,
            ),
            known_issue="https://github.com/milvus-io/milvus/issues/51567",
            known_success_ids=KNOWN_ISSUE_51567_IN_SUCCESS_IDS[case_name],
        )

    @pytest.mark.parametrize(
        "case_name, expr, expected_ids",
        mark_51567_or_cases(JSON_BOOL_MIXED_OR_51567_CASES),
    )
    def test_json_bool_mixed_type_or_typed_union_51567(
        self,
        regression_51568_collection,
        case_name,
        expr,
        expected_ids,
    ):
        client = self._client(alias=self.shared_alias)
        query_ids_or_xfail_known_loss(
            client,
            regression_51568_collection,
            expr,
            expected_ids,
            case_name,
            KNOWN_ISSUE_51567_OR_LOSS_IDS,
            KNOWN_ISSUE_51567_OR_ERROR_VALUE_CASES,
            "https://github.com/milvus-io/milvus/issues/51567",
            f"{case_name} query",
        )

    @pytest.mark.tags(CaseLabel.L1)
    def test_json_bool_mixed_type_or_search_51567(self, regression_51568_collection):
        client = self._client(alias=self.shared_alias)
        case_name = "bool_int_or_bool_first_three"
        search_ids_or_xfail_known_loss(
            client,
            regression_51568_collection,
            '(meta["b"] == true) or (meta["b"] == 1) or (meta["b"] == 0)',
            [1, 2, 5, 6],
            case_name,
            KNOWN_ISSUE_51567_OR_LOSS_IDS,
            KNOWN_ISSUE_51567_OR_ERROR_VALUE_CASES,
            "https://github.com/milvus-io/milvus/issues/51567",
            f"{case_name} search",
        )

    @pytest.mark.tags(CaseLabel.L1)
    def test_json_same_path_mixed_type_or_51568_empty_collection_control(self):
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        self.create_collection(
            client,
            collection_name,
            schema=self.build_51568_schema(client),
            consistency_level="Strong",
        )
        create_minimal_vector_index(self, client, collection_name, vector_field=default_vec)
        self.load_collection(client, collection_name)
        expr = '(meta["p"] == 1.0) or (meta["p"] == 2) or (meta["p"] == 3) or (meta["p"] == 4)'
        assert_query_ids(self, client, collection_name, expr, [], pk_field=default_pk)

    @pytest.mark.tags(CaseLabel.L1)
    def test_json_same_path_mixed_type_or_51568_one_doc_collection(self):
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        try:
            self.create_collection(
                client,
                collection_name,
                schema=self.build_51568_schema(client),
                consistency_level="Strong",
            )
            self.insert(client, collection_name, data=[{default_pk: 1, "meta": {"p": 1}}])
            self.flush(client, collection_name)
            create_minimal_vector_index(self, client, collection_name, vector_field=default_vec)
            self.load_collection(client, collection_name)
            expr = '(meta["p"] == 1.0) or (meta["p"] == 2) or (meta["p"] == 3) or (meta["p"] == 4)'
            query_ids_or_xfail_known_loss(
                client,
                collection_name,
                expr,
                [1],
                "float_int_or_4",
                KNOWN_ISSUE_51568_LOSS_IDS,
                KNOWN_ISSUE_51568_ERROR_VALUE_CASES,
                "https://github.com/milvus-io/milvus/issues/51568",
                "one-doc 51568 query",
            )
        finally:
            if client.has_collection(collection_name):
                client.drop_collection(collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize(
        "case_name, expr, fanout_count, expected_ids",
        fanout_params(BOOLEAN_FANOUT_EXPRESSIONS_L1 + SAME_FIELD_OR_FANOUT_EXPRESSIONS_L1),
    )
    def test_boolean_fanout_count_query_result_l1(
        self,
        order_fanout_collection,
        case_name,
        expr,
        fanout_count,
        expected_ids,
    ):
        client = self._client(alias=self.shared_alias)
        assert_query_ids(self, client, order_fanout_collection, expr, expected_ids, pk_field=default_pk)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize(
        "case_name, expr, fanout_count, expected_ids",
        fanout_params(BOOLEAN_FANOUT_EXPRESSIONS_L2),
    )
    def test_boolean_fanout_count_query_result_l2(
        self,
        order_fanout_collection,
        case_name,
        expr,
        fanout_count,
        expected_ids,
    ):
        client = self._client(alias=self.shared_alias)
        assert_query_ids(self, client, order_fanout_collection, expr, expected_ids, pk_field=default_pk)

    @pytest.mark.parametrize(
        "left_expr, right_expr, expected_ids",
        order_params(ORDER_SENSITIVE_EXPRESSIONS),
    )
    def test_expression_order_permutation_same_result(
        self,
        order_fanout_collection,
        left_expr,
        right_expr,
        expected_ids,
    ):
        client = self._client(alias=self.shared_alias)
        assert_same_query_result(
            self,
            client,
            order_fanout_collection,
            left_expr,
            right_expr,
            expected_ids,
            pk_field=default_pk,
        )

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize(
        "case_name, left_expr, right_expr, expected_ids",
        equivalent_expression_params(EQUIVALENT_EXPRESSION_CASES),
    )
    def test_equivalent_expression_same_result(
        self,
        order_fanout_collection,
        case_name,
        left_expr,
        right_expr,
        expected_ids,
    ):
        client = self._client(alias=self.shared_alias)
        assert_same_query_result(
            self,
            client,
            order_fanout_collection,
            left_expr,
            right_expr,
            expected_ids,
            pk_field=default_pk,
        )

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize(
        "case_name, expr, fanout_count, expected_ids",
        [pytest.param(*case, id=case[0]) for case in SEARCH_51568_CASES],
    )
    def test_json_same_path_mixed_type_or_search_51568(
        self,
        regression_51568_collection,
        case_name,
        expr,
        fanout_count,
        expected_ids,
    ):
        client = self._client(alias=self.shared_alias)
        search_ids_or_xfail_known_loss(
            client,
            regression_51568_collection,
            expr,
            expected_ids,
            case_name,
            KNOWN_ISSUE_51568_LOSS_IDS,
            KNOWN_ISSUE_51568_ERROR_VALUE_CASES,
            "https://github.com/milvus-io/milvus/issues/51568",
            f"{case_name} search",
        )

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("segment_mode", SEGMENT_MODES_ACTIVE, ids=SEGMENT_MODES_ACTIVE)
    def test_segment_mode_order_permutation(self, segment_mode):
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        prepare_loaded_empty_collection_for_segment(
            self,
            client,
            collection_name,
            self.build_order_schema(client),
            vector_field=default_vec,
        )
        rows = build_order_rows()
        insert_by_segment_mode(self, client, collection_name, rows[:5], rows[5:], segment_mode)
        wait_for_segment_mode(
            client,
            collection_name,
            segment_mode,
            expected_row_count=len(rows),
            expected_sealed_rows=len(rows[:5]) if segment_mode == "mixed" else None,
        )
        left_expr, right_expr, expected_ids = ORDER_SENSITIVE_EXPRESSIONS[0]
        assert_same_query_result(
            self,
            client,
            collection_name,
            left_expr,
            right_expr,
            expected_ids,
            pk_field=default_pk,
        )

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize(
        "case_name, expr, fanout_count, expected_ids, segment_mode",
        mark_51568_segment_cases(SEGMENT_51568_CASES, SEGMENT_51568_MODES),
    )
    def test_json_same_path_mixed_type_or_regression_51568_by_segment(
        self,
        case_name,
        expr,
        fanout_count,
        expected_ids,
        segment_mode,
    ):
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        prepare_loaded_empty_collection_for_segment(
            self,
            client,
            collection_name,
            self.build_51568_schema(client),
            vector_field=default_vec,
        )
        rows = [{default_pk: i, "meta": {"p": i}} for i in range(1, 21)]
        insert_by_segment_mode(self, client, collection_name, rows[:10], rows[10:], segment_mode)
        wait_for_segment_mode(
            client,
            collection_name,
            segment_mode,
            expected_row_count=len(rows),
            expected_sealed_rows=len(rows[:10]) if segment_mode == "mixed" else None,
        )
        query_ids_or_xfail_known_loss(
            client,
            collection_name,
            expr,
            expected_ids,
            case_name,
            KNOWN_ISSUE_51568_LOSS_IDS,
            KNOWN_ISSUE_51568_ERROR_VALUE_CASES,
            "https://github.com/milvus-io/milvus/issues/51568",
            f"{case_name} {segment_mode} query",
        )
