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

#include "RandomSampleNode.h"

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <ratio>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bitset/bitset.h"
#include "common/EasyAssert.h"
#include "common/Tracer.h"
#include "common/Types.h"
#include "exec/QueryContext.h"
#include "exec/expression/Utils.h"
#include "fmt/core.h"
#include "monitor/Monitor.h"
#include "plan/PlanNode.h"
#include "prometheus/histogram.h"

namespace milvus {
namespace exec {
PhyRandomSampleNode::PhyRandomSampleNode(
    int32_t operator_id,
    DriverContext* ctx,
    const std::shared_ptr<const plan::RandomSampleNode>& random_sample_node)
    : Operator(ctx,
               random_sample_node->output_type(),
               operator_id,
               random_sample_node->id(),
               "PhyRandomSampleNode"),
      factor_(random_sample_node->factor()) {
    // We should intercept unexpected number of factor at proxy level, so it's impossible to trigger
    // the panic here theoretically.
    AssertInfo(
        factor_ > 0.0 && factor_ < 1.0, "Unexpected sample factor {}", factor_);
    active_count_ = operator_context_->get_exec_context()
                        ->get_query_context()
                        ->get_active_count();
    is_source_node_ = random_sample_node->sources().size() == 0;
}

void
PhyRandomSampleNode::AddInput(RowVectorPtr& input) {
    input_ = std::move(input);
}

FixedVector<uint32_t>
PhyRandomSampleNode::HashsetSample(const uint32_t N,
                                   const uint32_t M,
                                   std::mt19937& gen) {
    std::uniform_int_distribution<> dis(0, N - 1);
    std::unordered_set<uint32_t> sampled;
    sampled.reserve(M);
    while (sampled.size() < M) {
        sampled.insert(dis(gen));
    }

    return FixedVector<uint32_t>(sampled.begin(), sampled.end());
}

FixedVector<uint32_t>
PhyRandomSampleNode::StandardSample(const uint32_t N,
                                    const uint32_t M,
                                    std::mt19937& gen) {
    FixedVector<uint32_t> inputs(N);
    FixedVector<uint32_t> outputs(M);
    std::iota(inputs.begin(), inputs.end(), 0);

    std::sample(inputs.begin(), inputs.end(), outputs.begin(), M, gen);
    return outputs;
}

FixedVector<uint32_t>
PhyRandomSampleNode::Sample(const uint32_t N, const float factor) {
    const uint32_t M = std::max(static_cast<uint32_t>(N * factor), 1u);
    std::random_device rd;
    std::mt19937 gen(rd());
    float epsilon = std::numeric_limits<float>::epsilon();
    // It's derived from some experiments
    if (factor <= 0.02 + epsilon ||
        (N <= 10000000 && factor <= 0.045 + epsilon) ||
        (N <= 60000000 && factor <= 0.025 + epsilon)) {
        return HashsetSample(N, M, gen);
    }
    return StandardSample(N, M, gen);
}

RowVectorPtr
PhyRandomSampleNode::GetOutput() {
    auto* query_context =
        operator_context_->get_exec_context()->get_query_context();
    milvus::exec::checkCancellation(query_context);

    if (is_finished_) {
        return nullptr;
    }

    tracer::AutoSpan span(
        "PhyRandomSampleNode::Execute", tracer::GetRootSpan(), true);

    if (!is_source_node_ && input_ == nullptr) {
        return nullptr;
    }

    if (active_count_ == 0) {
        is_finished_ = true;
        return nullptr;
    }

    tracer::AddEvent(fmt::format(
        "sample_factor: {}, active_count: {}", factor_, active_count_));

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

    RowVectorPtr result = nullptr;
    if (!is_source_node_) {
        auto input_roaring = GetRoaringBitmapVector(input_);
        if (input_roaring == nullptr) {
            input_roaring =
                RoaringBitmapVector::FromColumnVector(GetColumnVector(input_));
        }
        // note: false means the element is hit
        size_t input_false_count =
            input_roaring->size() - input_roaring->count();
        auto sample_output =
            std::make_shared<RoaringBitmapVector>(input_roaring->size(), true);
        sample_output->SetAll();

        if (input_false_count > 0) {
            auto sampled = Sample(input_false_count, factor_);
            assert(sampled.back() < input_false_count);
            for (auto i = 0; i < sampled.size(); ++i) {
                sample_output->Remove(input_roaring->SelectFalse(sampled[i]));
            }
        }
        if (!input_roaring->valid_values_all_valid()) {
            TargetBitmap valid_values;
            input_roaring->AppendValidValuesTo(
                valid_values, 0, input_roaring->size());
            sample_output->set_valid_values(std::move(valid_values));
        }

        result = std::make_shared<RowVector>(
            std::vector<VectorPtr>{std::move(sample_output)});
    } else {
        auto sample_output =
            std::make_shared<RoaringBitmapVector>(active_count_, true);
        // true in TargetBitmap means we don't want this row.
        bool need_flip = true;
        float factor = factor_;
        if (factor > 0.5) {
            need_flip = false;
            factor = 1.0 - factor;
        }
        auto sampled = Sample(active_count_, factor);

        if (need_flip) {
            sample_output->SetAll();
            for (auto n : sampled) {
                sample_output->Remove(n);
            }
        } else {
            sample_output->AddMany(sampled.data(), sampled.size());
        }

        result = std::make_shared<RowVector>(
            std::vector<VectorPtr>{std::move(sample_output)});
    }

    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    double duration =
        std::chrono::duration<double, std::micro>(end - start).count();
    milvus::monitor::internal_core_search_latency_random_sample.Observe(
        duration / 1000);

    if (result) {
        auto result_roaring = GetRoaringBitmapVector(result);
        AssertInfo(result_roaring != nullptr,
                   "PhyRandomSampleNode result should be roaring bitmap");
        auto sampled_count = result_roaring->size() - result_roaring->count();
        tracer::AddEvent(fmt::format("sampled_count: {}, total_count: {}",
                                     sampled_count,
                                     active_count_));
    }

    is_finished_ = true;

    return result;
}

bool
PhyRandomSampleNode::IsFinished() {
    return is_finished_;
}

}  // namespace exec
}  // namespace milvus
