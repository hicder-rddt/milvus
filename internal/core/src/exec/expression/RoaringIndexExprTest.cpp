// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/BitmapVector.h"
#include "common/BitsetView.h"
#include "common/OpContext.h"
#include "common/Schema.h"
#include "exec/expression/Expr.h"
#include "exec/expression/ExprCache.h"
#include "index/InvertedIndexTantivy.h"
#include "index/ScalarIndexSort.h"
#include "segcore/SegmentSealed.h"
#include "test_utils/DataGen.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "test_utils/storage_test_utils.h"

namespace milvus::exec {
namespace {

class TestIndexExpr : public SegmentExpr {
 public:
    TestIndexExpr(OpContext* op_ctx,
                  const segcore::SegmentInternalInterface* segment,
                  FieldId field_id,
                  int64_t active_count,
                  int64_t batch_size,
                  std::string signature)
        : SegmentExpr({},
                      "TestIndexExpr",
                      op_ctx,
                      segment,
                      field_id,
                      {},
                      DataType::INT64,
                      active_count,
                      batch_size,
                      0),
          signature_(std::move(signature)) {
        EnsurePinnedIndex();
    }

    VectorPtr
    In(const std::vector<int64_t>& values, int* calls = nullptr) {
        auto query = [calls](index::ScalarIndex<int64_t>* index_ptr,
                             const std::vector<int64_t>& query_values) {
            if (calls != nullptr) {
                ++*calls;
            }
            return index_ptr->InBitmap(query_values.size(),
                                       query_values.data());
        };
        return ProcessIndexChunks<int64_t>(query, values);
    }

    VectorPtr
    Range(int64_t lower, int64_t upper, int* calls = nullptr) {
        auto query = [calls](index::ScalarIndex<int64_t>* index_ptr,
                             int64_t lower_bound,
                             int64_t upper_bound) {
            if (calls != nullptr) {
                ++*calls;
            }
            return index_ptr->RangeBitmap(lower_bound, true, upper_bound, true);
        };
        return ProcessIndexChunks<int64_t>(query, lower, upper);
    }

    void
    Eval(EvalCtx&, VectorPtr&) override {
    }

    std::string
    ToString() const override {
        return signature_;
    }

    std::optional<expr::ColumnInfo>
    GetColumnInfo() const override {
        return std::nullopt;
    }

 private:
    std::string signature_;
};

struct IndexedSegment {
    SchemaPtr schema;
    segcore::SegmentSealedSPtr segment;
    FieldId field_id;
    FixedVector<int64_t> values;
};

IndexedSegment
BuildIndexedSegment(int64_t row_count) {
    auto schema = std::make_shared<Schema>();
    auto primary_key = schema->AddDebugField("pk", DataType::INT64);
    auto field_id = schema->AddDebugField("value", DataType::INT64);
    schema->set_primary_field_id(primary_key);

    auto raw_data = segcore::DataGen(schema, row_count);
    auto values = raw_data.get_col<int64_t>(field_id);
    auto segment = CreateSealedWithFieldDataLoaded(schema, raw_data);

    auto inverted_index =
        std::make_unique<index::InvertedIndexTantivy<int64_t>>();
    inverted_index->BuildWithRawDataForUT(
        values.size(), values.data(), Config{});

    segcore::LoadIndexInfo load_info;
    load_info.field_id = field_id.get();
    load_info.field_type = DataType::INT64;
    load_info.index_params = GenIndexParams(inverted_index.get());
    load_info.cache_index =
        CreateTestCacheIndex("roaring_scalar_index", std::move(inverted_index));
    segment->LoadIndex(load_info);

    return {schema,
            segcore::SegmentSealedSPtr(segment.release()),
            field_id,
            std::move(values)};
}

struct NestedIndexedSegment {
    SchemaPtr schema;
    segcore::SegmentSealedSPtr segment;
    FieldId field_id;
    std::vector<int64_t> elements;
    std::vector<size_t> row_starts;
};

NestedIndexedSegment
BuildNestedIndexedSegment(int64_t row_count) {
    auto schema = std::make_shared<Schema>();
    auto primary_key = schema->AddDebugField("pk", DataType::INT64);
    auto field_id =
        schema->AddDebugArrayField("structA[values]", DataType::INT64, false);
    schema->set_primary_field_id(primary_key);

    auto raw_data = segcore::DataGen(schema, row_count, 42, 0, 1, 3);
    std::vector<int64_t> elements;
    std::vector<size_t> row_starts;
    row_starts.reserve(row_count + 1);
    row_starts.push_back(0);

    for (int i = 0; i < raw_data.raw_->fields_data_size(); ++i) {
        auto* field_data = raw_data.raw_->mutable_fields_data(i);
        if (field_data->field_id() != field_id.get()) {
            continue;
        }
        auto* arrays =
            field_data->mutable_scalars()->mutable_array_data()->mutable_data();
        arrays->Clear();
        for (int64_t row = 0; row < row_count; ++row) {
            auto* array = arrays->Add();
            const auto length = static_cast<int>(row % 5);
            for (int elem = 0; elem < length; ++elem) {
                const auto value = row * 10 + elem;
                array->mutable_long_data()->mutable_data()->Add(value);
                elements.push_back(value);
            }
            row_starts.push_back(elements.size());
        }
        break;
    }

    auto segment = CreateSealedWithFieldDataLoaded(schema, raw_data);
    auto nested_index = std::make_unique<index::ScalarIndexSort<int64_t>>(
        storage::FileManagerContext(), true);
    nested_index->Build(elements.size(), elements.data(), nullptr);

    segcore::LoadIndexInfo load_info;
    load_info.field_id = field_id.get();
    load_info.field_type = DataType::ARRAY;
    load_info.element_type = DataType::INT64;
    load_info.index_params = GenIndexParams(nested_index.get());
    load_info.cache_index =
        CreateTestCacheIndex("roaring_nested_index", std::move(nested_index));
    segment->LoadIndex(load_info);

    return {schema,
            segcore::SegmentSealedSPtr(segment.release()),
            field_id,
            std::move(elements),
            std::move(row_starts)};
}

void
AssertResultMatches(const BitmapVector& result,
                    const FixedVector<int64_t>& values,
                    size_t offset,
                    const std::function<bool(int64_t)>& predicate) {
    ASSERT_LE(offset + result.size(), values.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result.result().test(i), predicate(values[offset + i]));
        EXPECT_TRUE(result.validity().test(i));
    }
}

class RoaringIndexExprTest : public ::testing::Test {
 protected:
    void
    TearDown() override {
        ExprResCacheManager::Instance().Clear();
        ExprResCacheManager::SetEnabled(false);
    }
};

TEST_F(RoaringIndexExprTest, DisabledCachePreservesRoaringAcrossBatches) {
    constexpr int64_t row_count = 257;
    constexpr int64_t batch_size = 31;
    auto indexed = BuildIndexedSegment(row_count);
    OpContext op_context;
    const auto selected = indexed.values[17];
    TestIndexExpr expression(&op_context,
                             indexed.segment.get(),
                             indexed.field_id,
                             row_count,
                             batch_size,
                             "roaring-disabled-cache");

    ExprResCacheManager::SetEnabled(false);
    size_t offset = 0;
    int calls = 0;
    while (offset < indexed.values.size()) {
        auto output = GetBitmapVector(expression.In({selected}, &calls));
        ASSERT_NE(output, nullptr);
        EXPECT_TRUE(output->result().is_roaring());
        EXPECT_TRUE(output->validity().is_roaring());
        FrozenRoaringBitsetView frozen_view(*output);
        EXPECT_TRUE(frozen_view.view().is_roaring());
        AssertResultMatches(
            *output, indexed.values, offset, [selected](int64_t value) {
                return value == selected;
            });
        offset += output->size();
    }
    EXPECT_EQ(calls, 1);
}

TEST_F(RoaringIndexExprTest,
       NestedIndexSlicesStayRoaringAcrossVariableLengthRowBatches) {
    constexpr int64_t row_count = 67;
    constexpr int64_t batch_size = 7;
    auto indexed = BuildNestedIndexedSegment(row_count);
    OpContext op_context;
    TestIndexExpr expression(&op_context,
                             indexed.segment.get(),
                             indexed.field_id,
                             row_count,
                             batch_size,
                             "roaring-nested-index-slices");

    const std::vector<int64_t> selected = {11, 42, 133, 314, 522, 641};
    ExprResCacheManager::SetEnabled(false);
    size_t row_offset = 0;
    size_t element_offset = 0;
    int calls = 0;
    while (row_offset < row_count) {
        auto output = GetBitmapVector(expression.In(selected, &calls));
        ASSERT_NE(output, nullptr);
        EXPECT_TRUE(output->result().is_roaring());
        EXPECT_TRUE(output->validity().is_roaring());

        const auto row_end =
            std::min<size_t>(row_count, row_offset + batch_size);
        const auto expected_size =
            indexed.row_starts[row_end] - indexed.row_starts[row_offset];
        ASSERT_EQ(output->size(), expected_size);
        for (size_t i = 0; i < expected_size; ++i) {
            const auto value = indexed.elements[element_offset + i];
            EXPECT_EQ(output->result().test(i),
                      std::find(selected.begin(), selected.end(), value) !=
                          selected.end());
            EXPECT_TRUE(output->validity().test(i));
        }

        row_offset = row_end;
        element_offset += expected_size;
    }
    EXPECT_EQ(element_offset, indexed.elements.size());
    EXPECT_EQ(calls, 1);
}

TEST_F(RoaringIndexExprTest, CacheHitAndAllAtOnceDoNotConsumeSharedValue) {
    constexpr int64_t row_count = 257;
    auto indexed = BuildIndexedSegment(row_count);
    OpContext op_context;

    auto& manager = ExprResCacheManager::Instance();
    ExprResCacheManager::SetEnabled(false);
    CacheConfig config;
    config.mode = CacheMode::Memory;
    config.mem_max_bytes = 1 << 20;
    config.compression_enabled = false;
    config.admission_threshold = 1;
    config.mem_min_eval_duration_us = 0;
    ASSERT_TRUE(manager.SetConfig(config));
    manager.Clear();
    ExprResCacheManager::SetEnabled(true);

    const auto lower = indexed.values[40];
    const auto upper = indexed.values[180];
    const auto min_value = std::min(lower, upper);
    const auto max_value = std::max(lower, upper);
    constexpr auto signature = "roaring-range-cache-ownership";

    int miss_calls = 0;
    TestIndexExpr first(&op_context,
                        indexed.segment.get(),
                        indexed.field_id,
                        row_count,
                        row_count,
                        signature);
    first.SetExecuteAllAtOnce();
    auto first_output =
        GetBitmapVector(first.Range(min_value, max_value, &miss_calls));
    ASSERT_NE(first_output, nullptr);
    ASSERT_TRUE(first_output->result().is_roaring());
    ASSERT_TRUE(first_output->validity().is_roaring());
    EXPECT_EQ(miss_calls, 1);
    AssertResultMatches(*first_output,
                        indexed.values,
                        0,
                        [min_value, max_value](int64_t value) {
                            return min_value <= value && value <= max_value;
                        });

    const bool cached_bit = first_output->result().test(0);
    first_output->result().set(0, !cached_bit);

    int hit_calls = 0;
    TestIndexExpr second(&op_context,
                         indexed.segment.get(),
                         indexed.field_id,
                         row_count,
                         row_count,
                         signature);
    second.SetExecuteAllAtOnce();
    auto second_output =
        GetBitmapVector(second.Range(min_value, max_value, &hit_calls));
    ASSERT_NE(second_output, nullptr);
    EXPECT_EQ(hit_calls, 0);
    EXPECT_TRUE(second_output->result().is_roaring());
    EXPECT_TRUE(second_output->validity().is_roaring());
    EXPECT_EQ(second_output->result().test(0), cached_bit);
    AssertResultMatches(*second_output,
                        indexed.values,
                        0,
                        [min_value, max_value](int64_t value) {
                            return min_value <= value && value <= max_value;
                        });
}

}  // namespace
}  // namespace milvus::exec
