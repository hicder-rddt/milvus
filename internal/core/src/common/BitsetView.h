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

#include <fmt/core.h>

#include <boost_ext/dynamic_bitset_ext.hpp>
#include <roaring/roaring.h>

#include <cstdlib>
#include <deque>
#include <cstring>
#include <limits>
#include <memory>

#include "bitset/detail/element_wise.h"
#include "common/Types.h"
#include "common/EasyAssert.h"
#include "common/RoaringBitmapVector.h"
#include "knowhere/bitsetview.h"

namespace milvus {

class BitsetView : public knowhere::BitsetView {
 public:
    BitsetView() = default;
    ~BitsetView() = default;

    BitsetView(const std::nullptr_t value)  // NOLINT
        : knowhere::BitsetView(value) {     // NOLINT
    }

    BitsetView(const uint8_t* data, size_t num_bits)
        : knowhere::BitsetView(data, num_bits) {  // NOLINT
    }

    BitsetView(const knowhere::BitsetView& bitset)  // NOLINT
        : knowhere::BitsetView(bitset) {
    }

    BitsetView(const BitsetType& bitset)  // NOLINT
        : BitsetView((uint8_t*)(bitset.data()), bitset.size()) {
    }

    BitsetView(const BitsetTypePtr& bitset_ptr) {  // NOLINT
        if (bitset_ptr) {
            *this = BitsetView(*bitset_ptr);
        }
    }

    // Return whether all bits are set (vacuously true when empty).
    bool
    all() const {
        if (empty()) {
            return true;
        }
        if (has_out_ids()) {
            for (size_t i = 0; i < size(); ++i) {
                if (!test(i)) {
                    return false;
                }
            }
            return true;
        }
        return bitset::detail::ElementWiseBitsetPolicy<uint8_t>::op_all(
            data(), 0, size());
    }

    // Return whether no bit is set (vacuously true when empty).
    bool
    none() const {
        if (empty()) {
            return true;
        }
        if (has_out_ids()) {
            for (size_t i = 0; i < size(); ++i) {
                if (test(i)) {
                    return false;
                }
            }
            return true;
        }
        return bitset::detail::ElementWiseBitsetPolicy<uint8_t>::op_none(
            data(), 0, size());
    }

    BitsetView
    subview(size_t offset, size_t size) const {
        if (empty()) {
            return {};
        }

        AssertInfo(
            (offset & 0x7) == 0, "offset {} is not divisible by 8", offset);
        AssertInfo(offset + size <= this->size(),
                   "index out of range, offset={}, size={}, bitset.size={}",
                   offset,
                   size,
                   this->size());
        return {data() + (offset >> 3), size};
    }
};

class FrozenRoaringBitsetView {
 public:
    explicit FrozenRoaringBitsetView(const TargetBitmapView& bitset) {
        num_bits_ = bitset.size();
        num_filtered_out_bits_ = bitset.count();
        if (num_bits_ == 0) {
            return;
        }

        std::unique_ptr<roaring_bitmap_t, RoaringDeleter> bitmap(
            roaring_bitmap_create());
        AssertInfo(bitmap != nullptr, "failed to create roaring bitmap");

        for (auto offset = bitset.find_first(true); offset.has_value();
             offset = bitset.find_next(offset.value(), true)) {
            AssertInfo(offset.value() <= std::numeric_limits<uint32_t>::max(),
                       "roaring bitmap only supports uint32 ids, got {}",
                       offset.value());
            roaring_bitmap_add(bitmap.get(),
                               static_cast<uint32_t>(offset.value()));
        }

        Freeze(bitmap.get());
    }

    explicit FrozenRoaringBitsetView(const RoaringBitmapVector& bitset) {
        num_bits_ = bitset.size();
        num_filtered_out_bits_ = bitset.count();
        if (num_bits_ == 0) {
            return;
        }

        Freeze(bitset.bitmap());
    }

    const BitsetView&
    view() const {
        return view_;
    }

    size_t
    size() const {
        return num_bits_;
    }

    size_t
    count() const {
        return num_filtered_out_bits_;
    }

 private:
    struct FreeDeleter {
        void
        operator()(char* ptr) const {
            std::free(ptr);
        }
    };

    struct RoaringDeleter {
        void
        operator()(roaring_bitmap_t* ptr) const {
            roaring_bitmap_free(ptr);
        }
    };

    static constexpr size_t kFrozenAlignment = 32;

    void
    Freeze(const roaring_bitmap_t* bitmap) {
        frozen_size_ = roaring_bitmap_frozen_size_in_bytes(bitmap);
        auto aligned_size =
            ((frozen_size_ + kFrozenAlignment - 1) / kFrozenAlignment) *
            kFrozenAlignment;
        void* aligned_buffer = nullptr;
        auto alloc_status =
            posix_memalign(&aligned_buffer, kFrozenAlignment, aligned_size);
        AssertInfo(alloc_status == 0 && aligned_buffer != nullptr,
                   "failed to allocate aligned frozen roaring bitmap buffer");
        frozen_buffer_.reset(static_cast<char*>(aligned_buffer));
        std::memset(frozen_buffer_.get(), 0, aligned_size);

        roaring_bitmap_frozen_serialize(bitmap, frozen_buffer_.get());
        view_ = BitsetView(
            knowhere::BitsetView::FromFrozenRoaring(frozen_buffer_.get(),
                                                    frozen_size_,
                                                    num_bits_,
                                                    num_filtered_out_bits_));
    }

    std::unique_ptr<char, FreeDeleter> frozen_buffer_;
    size_t frozen_size_{0};
    size_t num_bits_{0};
    size_t num_filtered_out_bits_{0};
    BitsetView view_;
};

}  // namespace milvus
