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

#include "FilterBitsNode.h"

#include <algorithm>
#include <chrono>
#include <ratio>
#include <utility>
#include <vector>

#include "common/EasyAssert.h"
#include "common/BitmapVector.h"
#include "common/Tracer.h"
#include "common/Types.h"
#include "exec/QueryContext.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/ExprCache.h"
#include "expr/ITypeExpr.h"
#include "fmt/core.h"
#include "monitor/Monitor.h"
#include "plan/PlanNode.h"
#include "prometheus/histogram.h"

namespace milvus {
namespace exec {

namespace {

std::string
BuildExprCacheKey(const plan::FilterBitsNode& filter,
                  QueryContext* query_context) {
    auto key = filter.ToString();
    auto* segment =
        query_context != nullptr ? query_context->get_segment() : nullptr;
    if (segment != nullptr &&
        segment->get_schema_snapshot()->get_ttl_field_id().has_value()) {
        key += fmt::format("|entity_ttl_physical_time_us:{}",
                           query_context->get_entity_ttl_physical_time_us());
    }
    return key;
}

}  // namespace

bool
ConvertPredicateToFilteredBitset(TargetBitmapView data,
                                 TargetBitmapView valid,
                                 const size_t size) {
    // FilterBitsNode outputs a filtered-row bitset: 1 means excluded. A SQL-style
    // predicate passes only when it is definitely TRUE, so UNKNOWN/NULL must be
    // excluded together with FALSE.
    if (valid.all()) {
        data.flip();
        return true;
    }

    data.flip();
    TargetBitmap invalid(valid);
    invalid.flip();
    data.inplace_or(invalid, size);
    valid.set();
    return false;
}

bool
ConvertPredicateToFilteredBitset(BitmapVector& bitmap) {
    const bool all_valid = bitmap.valid_values_all_valid();
    // FilterBits is the semantic boundary where result=TRUE changes from
    // "predicate matched" to "row is excluded":
    // excluded = ~result | ~validity.
    bitmap.result().flip();
    auto invalid = bitmap.validity().clone();
    invalid.flip();
    bitmap.result().or_with(invalid);
    bitmap.validity().set_all();
    return all_valid;
}

PhyFilterBitsNode::PhyFilterBitsNode(
    int32_t operator_id,
    DriverContext* driverctx,
    const std::shared_ptr<const plan::FilterBitsNode>& filter)
    : Operator(driverctx,
               filter->output_type(),
               operator_id,
               filter->id(),
               "PhyFilterBitsNode") {
    ExecContext* exec_context = operator_context_->get_exec_context();
    query_context_ = exec_context->get_query_context();
    std::vector<expr::TypedExprPtr> filters;
    filters.emplace_back(filter->filter());
    // This operator folds UNKNOWN predicate rows into the excluded set
    // (ConvertPredicateToFilteredBitset), i.e. it is a null-rejecting
    // consumer: let conjunctions in the predicate tree drop UNKNOWN rows
    // from their active sets early.
    exprs_ = std::make_unique<ExprSet>(
        filters, exec_context, /*null_rejecting=*/true);
    need_process_rows_ = query_context_->get_active_count();
    num_processed_rows_ = 0;

    enable_expr_cache_ = query_context_->get_enable_expr_cache();
    if (enable_expr_cache_) {
        // Only cache the predicate result when EVERY expression in it is
        // cacheable. A bloom_match subtree is non-cacheable (its slim ToString
        // cache key cannot distinguish distinct filter blobs), and that
        // propagates up, so a predicate containing bloom_match is never cached
        // and can never reuse another query's bitmap.
        for (const auto& e : exprs_->exprs()) {
            if (e && !e->IsCacheable()) {
                enable_expr_cache_ = false;
                break;
            }
        }
    }
    if (enable_expr_cache_) {
        expr_cache_key_ = BuildExprCacheKey(*filter, query_context_);
    }
}

void
PhyFilterBitsNode::AddInput(RowVectorPtr& input) {
    input_ = std::move(input);
}

bool
PhyFilterBitsNode::AllInputProcessed() {
    if (num_processed_rows_ == need_process_rows_) {
        input_ = nullptr;
        return true;
    }
    return false;
}

bool
PhyFilterBitsNode::IsFinished() {
    return AllInputProcessed();
}

RowVectorPtr
PhyFilterBitsNode::GetOutput() {
    milvus::exec::checkCancellation(query_context_);

    if (AllInputProcessed()) {
        return nullptr;
    }

    // Cache read: Stage 2 of two-stage search reuses the bitset cached by Stage 1.
    // Cache lives in the process-level ExprResCacheManager keyed by
    // (segment_id, FilterBitsNode signature + dynamic filter context), so
    // cross-query reuse is automatic only when the effective predicate matches.
    auto* cache_segment = query_context_->get_segment();
    const bool can_use_cache = enable_expr_cache_ && !expr_cache_key_.empty() &&
                               cache_segment != nullptr &&
                               cache_segment->type() == SegmentType::Sealed &&
                               ExprResCacheManager::IsEnabled();
    const auto put_cache = [&](const std::shared_ptr<BitmapVector>& bitmap) {
        if (!can_use_cache) {
            return;
        }
        ExprResCacheManager::Key key{cache_segment->get_segment_id(),
                                     expr_cache_key_};
        ExprResCacheManager::Value value;
        value.result = std::make_shared<Bitmap>(bitmap->result().clone());
        value.valid_result =
            std::make_shared<Bitmap>(bitmap->validity().clone());
        value.active_count = need_process_rows_;
        ExprResCacheManager::Instance().Put(key, value);
    };
    if (can_use_cache) {
        ExprResCacheManager::Key key{cache_segment->get_segment_id(),
                                     expr_cache_key_};
        ExprResCacheManager::Value cached;
        cached.active_count = need_process_rows_;
        if (ExprResCacheManager::Instance().Get(key, cached) &&
            cached.result != nullptr &&
            cached.result->size() == need_process_rows_) {
            num_processed_rows_ = need_process_rows_;
            auto valid = cached.valid_result ? cached.valid_result->clone()
                                             : Bitmap(need_process_rows_, true);
            return std::make_shared<RowVector>(
                std::vector<VectorPtr>{std::make_shared<BitmapVector>(
                    cached.result->clone(), std::move(valid))});
        }
    }

    tracer::AutoSpan span(
        "PhyFilterBitsNode::Execute", tracer::GetRootSpan(), true);
    tracer::AddEvent(fmt::format("input_rows: {}", need_process_rows_));

    exprs_->WaitPrefetch();

    std::chrono::high_resolution_clock::time_point scalar_start =
        std::chrono::high_resolution_clock::now();

    EvalCtx eval_ctx(operator_context_->get_exec_context());

    Bitmap bitset;
    Bitmap valid_bitset;

    // optimization: if all expressions can be executed at once,
    // execute in a single pass and flip in-place to avoid bitmap copies.
    if (exprs_->CanExecuteAllAtOnce()) {
        tracer::AddEvent("expr_execute_all_at_once");
        exprs_->SetExecuteAllAtOnce();

        exprs_->Eval(0, 1, true, eval_ctx, results_);
        AssertInfo(results_.size() == 1 && results_[0] != nullptr,
                   "PhyFilterBitsNode result size should be size one and not "
                   "be nullptr");
        auto bitmap_vec = GetBitmapVector(results_[0]);
        if (!bitmap_vec) {
            auto col_vec = GetColumnVector(results_[0]);
            AssertInfo(col_vec->IsBitmap(),
                       "PhyFilterBitsNode result should be bitmap vector");
            bitmap_vec = BitmapVector::FromColumnVector(col_vec);
        }
        ConvertPredicateToFilteredBitset(*bitmap_vec);
        num_processed_rows_ = bitmap_vec->size();

        AssertInfo(bitmap_vec->size() == need_process_rows_,
                   "bitset size: {}, need_process_rows_: {}",
                   bitmap_vec->size(),
                   need_process_rows_);

        put_cache(bitmap_vec);

        std::chrono::high_resolution_clock::time_point scalar_end =
            std::chrono::high_resolution_clock::now();
        double scalar_cost =
            std::chrono::duration<double, std::micro>(scalar_end - scalar_start)
                .count();
        milvus::monitor::internal_core_search_latency_scalar.Observe(
            scalar_cost / 1000);

        return std::make_shared<RowVector>(
            std::vector<VectorPtr>{std::move(bitmap_vec)});
    }

    while (num_processed_rows_ < need_process_rows_) {
        exprs_->Eval(0, 1, true, eval_ctx, results_);

        AssertInfo(results_.size() == 1 && results_[0] != nullptr,
                   "PhyFilterBitsNode result size should be size one and not "
                   "be nullptr");

        auto bitmap_vec = GetBitmapVector(results_[0]);
        if (!bitmap_vec) {
            auto col_vec = GetColumnVector(results_[0]);
            if (!col_vec->IsBitmap()) {
                ThrowInfo(UnexpectedError,
                          "PhyFilterBitsNode result should be bitmap");
            }
            bitmap_vec = BitmapVector::FromColumnVector(col_vec);
        }
        bitset.append(bitmap_vec->result());
        valid_bitset.append(bitmap_vec->validity());
        num_processed_rows_ += bitmap_vec->size();
    }
    auto bitmap_vec = std::make_shared<BitmapVector>(std::move(bitset),
                                                     std::move(valid_bitset));
    ConvertPredicateToFilteredBitset(*bitmap_vec);

    AssertInfo(bitmap_vec->size() == need_process_rows_,
               "bitset size: {}, need_process_rows_: {}",
               bitmap_vec->size(),
               need_process_rows_);
    Assert(bitmap_vec->validity().size() == need_process_rows_);

    // Cache before moving the output into the RowVector.
    put_cache(bitmap_vec);

    // num_processed_rows_ = need_process_rows_;
    std::chrono::high_resolution_clock::time_point scalar_end =
        std::chrono::high_resolution_clock::now();
    double scalar_cost =
        std::chrono::duration<double, std::micro>(scalar_end - scalar_start)
            .count();
    milvus::monitor::internal_core_search_latency_scalar.Observe(scalar_cost /
                                                                 1000);

    return std::make_shared<RowVector>(
        std::vector<VectorPtr>{std::move(bitmap_vec)});
}

}  // namespace exec
}  // namespace milvus
