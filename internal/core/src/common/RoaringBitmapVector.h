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

#pragma once

#include <roaring/roaring.h>

#include <algorithm>
#include <limits>
#include <memory>

#include "common/EasyAssert.h"
#include "common/Vector.h"

namespace milvus {

class RoaringBitmapVector final : public BaseVector {
 public:
    ~RoaringBitmapVector() override;

    RoaringBitmapVector(TargetBitmap&& bitset, TargetBitmap&& valid_bitset)
        : BaseVector(DataType::INT8, bitset.size()),
          bitmap_(roaring_bitmap_create()),
          valid_values_all_valid_(false),
          valid_values_(std::move(valid_bitset)) {
        AssertInfo(bitmap_ != nullptr, "failed to create roaring bitmap");
        AssertInfo(valid_values_.size() == length_,
                   "valid bitmap size {} does not match bitmap size {}",
                   valid_values_.size(),
                   length_);

        for (auto offset = bitset.find_first(true); offset.has_value();
             offset = bitset.find_next(offset.value(), true)) {
            AssertInfo(offset.value() <= std::numeric_limits<uint32_t>::max(),
                       "roaring bitmap only supports uint32 ids, got {}",
                       offset.value());
            roaring_bitmap_add(bitmap_.get(),
                               static_cast<uint32_t>(offset.value()));
        }
    }

    explicit RoaringBitmapVector(size_t size, bool valid = true)
        : BaseVector(DataType::INT8, size),
          bitmap_(roaring_bitmap_create()),
          valid_values_all_valid_(valid),
          valid_values_(valid ? TargetBitmap{} : TargetBitmap(size, false)) {
        AssertInfo(bitmap_ != nullptr, "failed to create roaring bitmap");
    }

    RoaringBitmapVector(size_t size, TargetBitmap&& valid_values)
        : BaseVector(DataType::INT8, size),
          bitmap_(roaring_bitmap_create()),
          valid_values_all_valid_(false),
          valid_values_(std::move(valid_values)) {
        AssertInfo(bitmap_ != nullptr, "failed to create roaring bitmap");
        AssertInfo(valid_values_.size() == length_,
                   "valid bitmap size {} does not match bitmap size {}",
                   valid_values_.size(),
                   length_);
    }

    static std::shared_ptr<RoaringBitmapVector>
    FromColumnVector(const ColumnVectorPtr& column) {
        auto size = column->size();
        auto roaring = std::make_shared<RoaringBitmapVector>(size, true);
        TargetBitmapView data(column->GetRawData(), size);
        roaring->OrShifted(data, 0);

        TargetBitmapView valid(column->GetValidRawData(), size);
        if (!valid.all()) {
            TargetBitmap valid_values;
            valid_values.append(valid);
            roaring->set_valid_values(std::move(valid_values));
        }
        return roaring;
    }

    std::shared_ptr<RoaringBitmapVector>
    Clone() const {
        auto cloned = valid_values_all_valid_
                          ? std::make_shared<RoaringBitmapVector>(length_, true)
                          : std::make_shared<RoaringBitmapVector>(
                                length_, valid_values_.clone());
        cloned->bitmap_.reset(roaring_bitmap_copy(bitmap_.get()));
        AssertInfo(cloned->bitmap_ != nullptr, "failed to copy roaring bitmap");
        return cloned;
    }

    VectorPtr
    cloneEmpty(vector_size_t size) const override {
        return std::make_shared<RoaringBitmapVector>(size, true);
    }

    ColumnVectorPtr
    ToColumnVector() const {
        auto bitset = ToTargetBitmap();
        auto valid_values = ValidValuesBitmap().clone();
        return std::make_shared<ColumnVector>(std::move(bitset),
                                              std::move(valid_values));
    }

    TargetBitmap
    ToTargetBitmap() const {
        TargetBitmap bitset(length_, false);
        IterateTrueBits([&](uint32_t offset) {
            if (offset < length_) {
                bitset[offset] = true;
            }
            return true;
        });
        return bitset;
    }

    size_t
    size() const {
        return length_;
    }

    void
    Add(size_t offset) {
        AssertInfo(offset <= std::numeric_limits<uint32_t>::max(),
                   "roaring bitmap only supports uint32 ids, got {}",
                   offset);
        roaring_bitmap_add(bitmap_.get(), static_cast<uint32_t>(offset));
    }

    void
    Remove(size_t offset) {
        AssertInfo(offset <= std::numeric_limits<uint32_t>::max(),
                   "roaring bitmap only supports uint32 ids, got {}",
                   offset);
        roaring_bitmap_remove(bitmap_.get(), static_cast<uint32_t>(offset));
    }

    void
    RemoveRange(size_t begin, size_t end) {
        if (begin >= end) {
            return;
        }
        AssertInfo(end - 1 <= std::numeric_limits<uint32_t>::max(),
                   "roaring bitmap only supports uint32 ids, got {}",
                   end - 1);
        roaring_bitmap_remove_range(bitmap_.get(), begin, end);
    }

    void
    AddMany(const uint32_t* offsets, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            if (offsets[i] < length_) {
                roaring_bitmap_add(bitmap_.get(), offsets[i]);
            }
        }
    }

    void
    AddRange(size_t begin, size_t end) {
        if (begin >= end) {
            return;
        }
        AssertInfo(end - 1 <= std::numeric_limits<uint32_t>::max(),
                   "roaring bitmap only supports uint32 ids, got {}",
                   end - 1);
        roaring_bitmap_add_range(bitmap_.get(), begin, end);
        roaring_bitmap_run_optimize(bitmap_.get());
    }

    void
    Flip() {
        if (length_ == 0) {
            return;
        }
        roaring_bitmap_flip_inplace(bitmap_.get(), 0, length_);
        if (!valid_values_all_valid_) {
            RoaringBitmapVector valid_mask(length_, true);
            valid_mask.Or(valid_values_, length_);
            roaring_bitmap_and_inplace(bitmap_.get(), valid_mask.bitmap_.get());
        }
    }

    void
    And(const RoaringBitmapVector& other) {
        AssertSameSize(other);
        UpdateValidValuesForAnd(other);
        roaring_bitmap_and_inplace(bitmap_.get(), other.bitmap_.get());
    }

    void
    Or(const RoaringBitmapVector& other) {
        AssertSameSize(other);
        UpdateValidValuesForOr(other);
        roaring_bitmap_or_inplace(bitmap_.get(), other.bitmap_.get());
    }

    void
    AndNot(const RoaringBitmapVector& other) {
        AssertSameSize(other);
        roaring_bitmap_andnot_inplace(bitmap_.get(), other.bitmap_.get());
    }

    void
    SetAll() {
        AddRange(0, length_);
    }

    void
    Or(const TargetBitmap& bitset, size_t limit) {
        auto or_size = std::min({bitset.size(), length_, limit});
        for (auto offset = bitset.find_first(true);
             offset.has_value() && offset.value() < or_size;
             offset = bitset.find_next(offset.value(), true)) {
            Add(offset.value());
        }
    }

    void
    OrShifted(const TargetBitmapView& bitset, size_t offset) {
        if (offset >= length_) {
            return;
        }
        auto or_size = std::min(bitset.size(), length_ - offset);
        for (auto value = bitset.find_first(true);
             value.has_value() && value.value() < or_size;
             value = bitset.find_next(value.value(), true)) {
            Add(offset + value.value());
        }
    }

    void
    OrShifted(const RoaringBitmapVector& bitset, size_t offset) {
        bitset.IterateTrueBits([&](uint32_t value) {
            Add(offset + value);
            return true;
        });
    }

    std::shared_ptr<RoaringBitmapVector>
    Slice(size_t offset, size_t size) const {
        AssertInfo(offset + size <= length_,
                   "roaring bitmap slice [{}, {}) exceeds size {}",
                   offset,
                   offset + size,
                   length_);
        auto sliced = std::make_shared<RoaringBitmapVector>(size, false);
        if (valid_values_all_valid_) {
            sliced->valid_values_all_valid_ = true;
            sliced->valid_values_.clear();
        } else {
            sliced->valid_values_.clear();
            sliced->valid_values_.append(valid_values_, offset, size);
        }
        IterateTrueBits([&](uint32_t value) {
            if (value >= offset && value < offset + size) {
                sliced->Add(value - offset);
            }
            return value < offset + size;
        });
        return sliced;
    }

    size_t
    count() const {
        return roaring_bitmap_get_cardinality(bitmap_.get());
    }

    bool
    Contains(size_t offset) const {
        if (offset >= length_) {
            return false;
        }
        AssertInfo(offset <= std::numeric_limits<uint32_t>::max(),
                   "roaring bitmap only supports uint32 ids, got {}",
                   offset);
        return roaring_bitmap_contains(bitmap_.get(),
                                       static_cast<uint32_t>(offset));
    }

    uint32_t
    SelectFalse(uint32_t rank) const {
        AssertInfo(rank < length_ - count(),
                   "false rank {} out of bounds for bitmap with {} false bits",
                   rank,
                   length_ - count());
        uint32_t low = 0;
        uint32_t high = static_cast<uint32_t>(length_ - 1);
        while (low < high) {
            auto mid = low + (high - low) / 2;
            auto true_count = roaring_bitmap_rank(bitmap_.get(), mid);
            auto false_count = static_cast<uint64_t>(mid) + 1 - true_count;
            if (false_count > rank) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }

    uint32_t
    SelectTrue(uint32_t rank) const {
        AssertInfo(rank < count(),
                   "true rank {} out of bounds for bitmap with {} true bits",
                   rank,
                   count());
        uint32_t value = 0;
        auto found = roaring_bitmap_select(bitmap_.get(), rank, &value);
        AssertInfo(found, "failed to select true rank {}", rank);
        return value;
    }

    bool
    AllTrue() const {
        return valid_values_all_valid_ && count() == length_;
    }

    bool
    AllFalse() const {
        return valid_values_all_valid_ && count() == 0;
    }

    const roaring_bitmap_t*
    bitmap() const {
        return bitmap_.get();
    }

    const TargetBitmap&
    valid_values() const {
        return ValidValuesBitmap();
    }

    bool
    valid_values_all_valid() const {
        return valid_values_all_valid_;
    }

    void
    AppendValidValuesTo(TargetBitmap& dst, size_t offset, size_t size) const {
        AssertInfo(offset + size <= length_,
                   "valid bitmap slice [{}, {}) exceeds size {}",
                   offset,
                   offset + size,
                   length_);
        if (valid_values_all_valid_) {
            dst.append(TargetBitmap(size, true));
        } else {
            dst.append(valid_values_, offset, size);
        }
    }

    void
    set_valid_values(TargetBitmap&& valid_values) {
        AssertInfo(valid_values.size() == length_,
                   "valid bitmap size {} does not match bitmap size {}",
                   valid_values.size(),
                   length_);
        valid_values_all_valid_ = false;
        valid_values_ = std::move(valid_values);
    }

    void
    SetAllValid() {
        valid_values_all_valid_ = true;
        valid_values_.clear();
    }

 private:
    struct RoaringDeleter {
        void
        operator()(roaring_bitmap_t* ptr) const {
            roaring_bitmap_free(ptr);
        }
    };

    template <typename Func>
    void
    IterateTrueBits(Func&& func) const {
        struct Context {
            Func* func;
        } context{&func};

        roaring_iterate(
            bitmap_.get(),
            [](uint32_t value, void* param) -> bool {
                auto* context = static_cast<Context*>(param);
                return (*context->func)(value);
            },
            &context);
    }

    const TargetBitmap&
    ValidValuesBitmap() const {
        if (valid_values_all_valid_) {
            valid_values_ = TargetBitmap(length_, true);
            valid_values_all_valid_ = false;
        }
        return valid_values_;
    }

    void
    AssertSameSize(const RoaringBitmapVector& other) const {
        AssertInfo(length_ == other.length_,
                   "roaring bitmap size {} does not match {}",
                   length_,
                   other.length_);
    }

    void
    UpdateValidValuesForAnd(const RoaringBitmapVector& other) {
        if (valid_values_all_valid_ && other.valid_values_all_valid_) {
            return;
        }

        auto left_data = ToTargetBitmap();
        auto right_data = other.ToTargetBitmap();
        TargetBitmap left_valid = ValidValuesBitmap().clone();
        TargetBitmap right_valid = other.ValidValuesBitmap().clone();

        // result_valid = left_valid & (right_valid | ~left_data) |
        //                right_valid & ~right_data
        TargetBitmap tmp(length_, true);
        tmp.inplace_xor(left_data, length_);
        tmp.inplace_or(right_valid, length_);
        left_valid.inplace_and(tmp, length_);

        tmp.set();
        tmp.inplace_xor(right_data, length_);
        tmp.inplace_and(right_valid, length_);
        left_valid.inplace_or(tmp, length_);

        valid_values_ = std::move(left_valid);
        valid_values_all_valid_ = false;
    }

    void
    UpdateValidValuesForOr(const RoaringBitmapVector& other) {
        if (valid_values_all_valid_ && other.valid_values_all_valid_) {
            return;
        }

        auto left_data = ToTargetBitmap();
        auto right_data = other.ToTargetBitmap();
        TargetBitmap left_valid = ValidValuesBitmap().clone();
        TargetBitmap right_valid = other.ValidValuesBitmap().clone();

        // result_valid = left_valid & (right_valid | left_data) |
        //                right_valid & right_data
        TargetBitmap tmp(length_);
        tmp.inplace_or(left_data, length_);
        tmp.inplace_or(right_valid, length_);
        left_valid.inplace_and(tmp, length_);

        tmp.reset();
        tmp.inplace_or(right_data, length_);
        tmp.inplace_and(right_valid, length_);
        left_valid.inplace_or(tmp, length_);

        valid_values_ = std::move(left_valid);
        valid_values_all_valid_ = false;
    }

    std::unique_ptr<roaring_bitmap_t, RoaringDeleter> bitmap_;
    mutable bool valid_values_all_valid_{true};
    mutable TargetBitmap valid_values_;
};

using RoaringBitmapVectorPtr = std::shared_ptr<RoaringBitmapVector>;

}  // namespace milvus
