// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <roaring/roaring.hh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

#include "common/EasyAssert.h"
#include "common/Vector.h"

namespace milvus {

using DenseBitmap = TargetBitmap;
using RoaringBitmap = roaring::Roaring;

// Fixed-domain, backend-neutral query bitmap. Roaring is selected by default;
// dense is used only when the logical domain exceeds uint32_t.
class __attribute__((visibility("default"))) Bitmap {
 public:
    enum class Backend { Dense, Roaring };

    explicit Bitmap(size_t size = 0, bool value = false)
        : size_(size), storage_(MakeStorage(size, value)) {
    }

    Bitmap(const TargetBitmap& bitmap) : Bitmap(bitmap.clone()) {
    }

    Bitmap(TargetBitmap&& bitmap)
        : size_(bitmap.size()),
          storage_(CanUseRoaring(size_) ? Storage(ToRoaring(bitmap))
                                        : Storage(std::move(bitmap))) {
    }

    Bitmap(size_t size, RoaringBitmap bitmap)
        : size_(size), storage_(MakeStorage(size, std::move(bitmap))) {
    }

    Bitmap(TargetBitmap bitmap, Backend backend)
        : size_(bitmap.size()),
          storage_(backend == Backend::Roaring && CanUseRoaring(size_)
                       ? Storage(ToRoaring(bitmap))
                       : Storage(std::move(bitmap))) {
    }

    Bitmap(const Bitmap& other)
        : size_(other.size_),
          storage_(std::visit(
              [](const auto& bitmap) -> Storage {
                  using T = std::decay_t<decltype(bitmap)>;
                  if constexpr (std::is_same_v<T, DenseBitmap>) {
                      return bitmap.clone();
                  } else {
                      return RoaringBitmap(bitmap);
                  }
              },
              other.storage_)) {
    }

    Bitmap(Bitmap&&) noexcept = default;

    Bitmap&
    operator=(const Bitmap& other) {
        if (this != &other) {
            Bitmap copy(other);
            *this = std::move(copy);
        }
        return *this;
    }

    Bitmap&
    operator=(Bitmap&&) noexcept = default;

    size_t
    size() const {
        return size_;
    }

    bool
    empty() const {
        return size_ == 0;
    }

    bool
    operator[](size_t offset) const {
        return test(offset);
    }

    static bool
    CanUseRoaring(size_t size) {
        return size <=
               static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
    }

    Backend
    backend() const {
        return std::holds_alternative<RoaringBitmap>(storage_)
                   ? Backend::Roaring
                   : Backend::Dense;
    }

    bool
    is_roaring() const {
        return backend() == Backend::Roaring;
    }

    bool
    test(size_t offset) const {
        if (offset >= size_) {
            return false;
        }
        return std::visit(
            [offset](const auto& bitmap) -> bool {
                using T = std::decay_t<decltype(bitmap)>;
                if constexpr (std::is_same_v<T, DenseBitmap>) {
                    return bitmap[offset];
                } else {
                    return bitmap.contains(static_cast<uint32_t>(offset));
                }
            },
            storage_);
    }

    void
    set(size_t offset, bool value = true) {
        AssertInfo(
            offset < size_, "bitmap offset {} out of range {}", offset, size_);
        std::visit(
            [offset, value](auto& bitmap) {
                using T = std::decay_t<decltype(bitmap)>;
                if constexpr (std::is_same_v<T, DenseBitmap>) {
                    bitmap.set(offset, value);
                } else if (value) {
                    bitmap.add(static_cast<uint32_t>(offset));
                } else {
                    bitmap.remove(static_cast<uint32_t>(offset));
                }
            },
            storage_);
    }

    void
    set_range(size_t begin, size_t end, bool value) {
        AssertInfo(begin <= end && end <= size_,
                   "bitmap range [{}, {}) out of range {}",
                   begin,
                   end,
                   size_);
        if (begin == end) {
            return;
        }
        std::visit(
            [begin, end, value](auto& bitmap) {
                using T = std::decay_t<decltype(bitmap)>;
                if constexpr (std::is_same_v<T, DenseBitmap>) {
                    bitmap.set(begin, end - begin, value);
                } else if (value) {
                    bitmap.addRange(begin, end);
                } else {
                    bitmap.removeRange(begin, end);
                }
            },
            storage_);
    }

    void
    set_all() {
        set_range(0, size_, true);
    }

    void
    reset_all() {
        set_range(0, size_, false);
    }

    size_t
    count() const {
        return std::visit(
            [](const auto& bitmap) -> size_t {
                using T = std::decay_t<decltype(bitmap)>;
                if constexpr (std::is_same_v<T, DenseBitmap>) {
                    return bitmap.count();
                } else {
                    return bitmap.cardinality();
                }
            },
            storage_);
    }

    bool
    all() const {
        return count() == size_;
    }

    bool
    none() const {
        return count() == 0;
    }

    Bitmap
    clone() const {
        return *this;
    }

    Bitmap
    slice(size_t offset, size_t length) const {
        AssertInfo(offset <= size_ && length <= size_ - offset,
                   "bitmap slice [{}, {}) out of range {}",
                   offset,
                   offset + length,
                   size_);
        if (length == 0) {
            return Bitmap(0, false);
        }
        if (const auto* dense = std::get_if<DenseBitmap>(&storage_)) {
            TargetBitmap result;
            result.append(*dense, offset, length);
            return Bitmap(std::move(result), Backend::Dense);
        }

        RoaringBitmap result;
        roaring_uint32_iterator_t it;
        roaring_init_iterator(&std::get<RoaringBitmap>(storage_).roaring, &it);
        roaring_move_uint32_iterator_equalorlarger(
            &it, static_cast<uint32_t>(offset));
        const auto end = offset + length;
        while (it.has_value && static_cast<size_t>(it.current_value) < end) {
            result.add(static_cast<uint32_t>(it.current_value - offset));
            roaring_advance_uint32_iterator(&it);
        }
        return Bitmap(length, std::move(result));
    }

    void
    append(const Bitmap& other) {
        const auto old_size = size_;
        Bitmap result(size_ + other.size_, false);
        iterate([&](size_t value) {
            result.set(value);
            return true;
        });
        other.iterate([&](size_t value) {
            result.set(old_size + value);
            return true;
        });
        *this = std::move(result);
    }

    void
    flip() {
        std::visit(
            [this](auto& bitmap) {
                using T = std::decay_t<decltype(bitmap)>;
                if constexpr (std::is_same_v<T, DenseBitmap>) {
                    bitmap.flip();
                } else {
                    bitmap.flip(0, size_);
                }
            },
            storage_);
    }

    void
    and_with(const Bitmap& other) {
        BinaryUpdate(
            other,
            [](DenseBitmap& left, const DenseBitmap& right) {
                left.inplace_and(right, left.size());
            },
            [](RoaringBitmap& left, const RoaringBitmap& right) {
                left &= right;
            });
    }

    void
    or_with(const Bitmap& other) {
        BinaryUpdate(
            other,
            [](DenseBitmap& left, const DenseBitmap& right) {
                left.inplace_or(right, left.size());
            },
            [](RoaringBitmap& left, const RoaringBitmap& right) {
                left |= right;
            });
    }

    void
    xor_with(const Bitmap& other) {
        BinaryUpdate(
            other,
            [](DenseBitmap& left, const DenseBitmap& right) {
                left.inplace_xor(right, left.size());
            },
            [](RoaringBitmap& left, const RoaringBitmap& right) {
                left ^= right;
            });
    }

    void
    and_not_with(const Bitmap& other) {
        BinaryUpdate(
            other,
            [](DenseBitmap& left, const DenseBitmap& right) {
                auto complement = right.clone();
                complement.flip();
                left.inplace_and(complement, left.size());
            },
            [](RoaringBitmap& left, const RoaringBitmap& right) {
                left -= right;
            });
    }

    TargetBitmap
    to_dense() const {
        if (const auto* dense = std::get_if<DenseBitmap>(&storage_)) {
            return dense->clone();
        }
        TargetBitmap result(size_, false);
        iterate([&](size_t value) {
            result.set(value);
            return true;
        });
        return result;
    }

    TargetBitmap
    to_dense(size_t offset, size_t length) const {
        AssertInfo(offset <= size_ && length <= size_ - offset,
                   "bitmap range [{}, {}) out of range {}",
                   offset,
                   offset + length,
                   size_);
        if (length == 0) {
            return TargetBitmap();
        }
        if (const auto* dense = std::get_if<DenseBitmap>(&storage_)) {
            TargetBitmap result;
            result.append(*dense, offset, length);
            return result;
        }

        TargetBitmap result(length, false);
        roaring_uint32_iterator_t it;
        roaring_init_iterator(&std::get<RoaringBitmap>(storage_).roaring, &it);
        roaring_move_uint32_iterator_equalorlarger(
            &it, static_cast<uint32_t>(offset));
        const auto end = offset + length;
        while (it.has_value && static_cast<size_t>(it.current_value) < end) {
            result.set(static_cast<size_t>(it.current_value) - offset);
            roaring_advance_uint32_iterator(&it);
        }
        return result;
    }

    void
    append_range_to(TargetBitmap& dst, size_t offset, size_t length) const {
        dst.append(to_dense(offset, length));
    }

    template <typename OffsetContainer>
    TargetBitmap
    gather_to_dense(const OffsetContainer& offsets) const {
        TargetBitmap result(offsets.size(), false);
        for (size_t i = 0; i < offsets.size(); ++i) {
            AssertInfo(
                offsets[i] >= 0 && static_cast<size_t>(offsets[i]) < size_,
                "bitmap offset {} out of range {}",
                offsets[i],
                size_);
            result.set(i, test(static_cast<size_t>(offsets[i])));
        }
        return result;
    }

    RoaringBitmap
    to_roaring() const {
        AssertInfo(CanUseRoaring(size_),
                   "logical bitmap size {} exceeds Roaring domain",
                   size_);
        if (const auto* roaring = std::get_if<RoaringBitmap>(&storage_)) {
            return *roaring;
        }
        return ToRoaring(std::get<DenseBitmap>(storage_));
    }

    const roaring_bitmap_t*
    roaring_bitmap() const {
        const auto* bitmap = std::get_if<RoaringBitmap>(&storage_);
        AssertInfo(bitmap != nullptr,
                   "dense bitmap has no live roaring_bitmap_t");
        return &bitmap->roaring;
    }

    const DenseBitmap&
    dense_bitmap() const {
        const auto* bitmap = std::get_if<DenseBitmap>(&storage_);
        AssertInfo(bitmap != nullptr, "roaring bitmap has no dense storage");
        return *bitmap;
    }

    template <typename Func>
    void
    iterate(Func&& func) const {
        std::visit(
            [&](const auto& bitmap) {
                using T = std::decay_t<decltype(bitmap)>;
                if constexpr (std::is_same_v<T, DenseBitmap>) {
                    for (auto offset = bitmap.find_first(true);
                         offset.has_value();
                         offset = bitmap.find_next(offset.value(), true)) {
                        if (!func(offset.value())) {
                            break;
                        }
                    }
                } else {
                    for (auto value : bitmap) {
                        if (!func(value)) {
                            break;
                        }
                    }
                }
            },
            storage_);
    }

 private:
    using Storage = std::variant<DenseBitmap, RoaringBitmap>;

    static Storage
    MakeStorage(size_t size, bool value) {
        if (!CanUseRoaring(size)) {
            return DenseBitmap(size, value);
        }
        RoaringBitmap bitmap;
        if (value && size != 0) {
            bitmap.addRange(0, size);
        }
        return bitmap;
    }

    static Storage
    MakeStorage(size_t size, RoaringBitmap bitmap) {
        if (CanUseRoaring(size)) {
            bitmap.removeRange(size, std::numeric_limits<uint64_t>::max());
            return bitmap;
        }
        DenseBitmap dense(size, false);
        for (auto offset : bitmap) {
            dense.set(offset);
        }
        return dense;
    }

    static RoaringBitmap
    ToRoaring(const DenseBitmap& bitmap) {
        RoaringBitmap result;
        for (auto offset = bitmap.find_first(true); offset.has_value();
             offset = bitmap.find_next(offset.value(), true)) {
            result.add(static_cast<uint32_t>(offset.value()));
        }
        return result;
    }

    template <typename DenseOp, typename RoaringOp>
    void
    BinaryUpdate(const Bitmap& other,
                 DenseOp&& dense_op,
                 RoaringOp&& roaring_op) {
        AssertInfo(size_ == other.size_,
                   "bitmap size {} does not match {}",
                   size_,
                   other.size_);
        std::visit(
            [&](auto& left, const auto& right) {
                using L = std::decay_t<decltype(left)>;
                using R = std::decay_t<decltype(right)>;
                if constexpr (std::is_same_v<L, DenseBitmap>) {
                    if constexpr (std::is_same_v<R, DenseBitmap>) {
                        dense_op(left, right);
                    } else {
                        auto converted = other.to_dense();
                        dense_op(left, converted);
                    }
                } else {
                    if constexpr (std::is_same_v<R, RoaringBitmap>) {
                        roaring_op(left, right);
                    } else {
                        auto converted = other.to_roaring();
                        roaring_op(left, converted);
                    }
                }
            },
            storage_,
            other.storage_);
    }

    size_t size_;
    Storage storage_;
};

class __attribute__((visibility("default"))) BitmapVector final
    : public BaseVector {
 public:
    ~BitmapVector() override;

    BitmapVector(TargetBitmap&& result, TargetBitmap&& validity)
        : BaseVector(DataType::INT8, result.size()),
          result_(std::move(result)),
          validity_(std::move(validity)) {
        AssertInfo(result_.size() == validity_.size(),
                   "bitmap result and validity size mismatch");
    }

    explicit BitmapVector(size_t size, bool valid = true)
        : BaseVector(DataType::INT8, size),
          result_(size, false),
          validity_(size, valid) {
    }

    BitmapVector(Bitmap result, Bitmap validity)
        : BaseVector(DataType::INT8, result.size()),
          result_(std::move(result)),
          validity_(std::move(validity)) {
        AssertInfo(result_.size() == validity_.size(),
                   "bitmap result and validity size mismatch");
    }

    size_t
    size() const {
        return length_;
    }

    static std::shared_ptr<BitmapVector>
    FromColumnVector(const ColumnVectorPtr& column) {
        const auto size = static_cast<size_t>(column->size());
        TargetBitmap result;
        result.append(TargetBitmapView(column->GetRawData(), size));
        TargetBitmap validity;
        validity.append(TargetBitmapView(column->GetValidRawData(), size));
        return std::make_shared<BitmapVector>(Bitmap(std::move(result)),
                                              Bitmap(std::move(validity)));
    }

    std::shared_ptr<BitmapVector>
    Clone() const {
        return std::make_shared<BitmapVector>(result_.clone(),
                                              validity_.clone());
    }

    VectorPtr
    cloneEmpty(vector_size_t size) const override {
        return std::make_shared<BitmapVector>(size, true);
    }

    ColumnVectorPtr
    ToColumnVector() const {
        return std::make_shared<ColumnVector>(result_.to_dense(),
                                              validity_.to_dense());
    }

    TargetBitmap
    ToTargetBitmap() const {
        return result_.to_dense();
    }

    Bitmap&
    result() {
        return result_;
    }
    const Bitmap&
    result() const {
        return result_;
    }
    Bitmap&
    validity() {
        return validity_;
    }
    const Bitmap&
    validity() const {
        return validity_;
    }

    void
    Add(size_t offset) {
        result_.set(offset);
    }
    void
    Remove(size_t offset) {
        result_.set(offset, false);
    }
    void
    AddMany(const uint32_t* offsets, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            if (offsets[i] < result_.size()) {
                result_.set(offsets[i]);
            }
        }
    }
    void
    AddRange(size_t begin, size_t end) {
        result_.set_range(begin, end, true);
    }
    void
    RemoveRange(size_t begin, size_t end) {
        result_.set_range(begin, end, false);
    }
    void
    SetAll() {
        result_.set_all();
    }
    void
    Flip() {
        result_.flip();
        // Keep the canonical representation of UNKNOWN as data=false,
        // validity=false.  Flipping the data bits alone turns UNKNOWN into a
        // set bit, which leaks through consumers that intentionally inspect
        // only the data bitmap.
        result_.and_with(validity_);
    }
    size_t
    count() const {
        return result_.count();
    }
    bool
    Contains(size_t offset) const {
        return result_.test(offset);
    }

    void
    And(const BitmapVector& other) {
        auto not_left = result_.clone();
        not_left.flip();
        not_left.or_with(other.validity_);
        auto new_validity = validity_.clone();
        new_validity.and_with(not_left);
        auto right_false = other.result_.clone();
        right_false.flip();
        right_false.and_with(other.validity_);
        new_validity.or_with(right_false);
        result_.and_with(other.result_);
        validity_ = std::move(new_validity);
    }

    void
    Or(const BitmapVector& other) {
        auto left_true_or_valid = other.validity_.clone();
        left_true_or_valid.or_with(result_);
        auto new_validity = validity_.clone();
        new_validity.and_with(left_true_or_valid);
        auto right_true = other.validity_.clone();
        right_true.and_with(other.result_);
        new_validity.or_with(right_true);
        result_.or_with(other.result_);
        validity_ = std::move(new_validity);
    }

    void
    AndNot(const BitmapVector& other) {
        result_.and_not_with(other.result_);
    }

    bool
    AllTrue() const {
        return validity_.all() && result_.all();
    }
    bool
    AllFalse() const {
        return validity_.all() && result_.none();
    }
    bool
    valid_values_all_valid() const {
        return validity_.all();
    }
    void
    SetAllValid() {
        validity_.set_all();
    }
    const roaring_bitmap_t*
    bitmap() const {
        return result_.roaring_bitmap();
    }

    void
    Or(const TargetBitmap& bitset, size_t limit) {
        const auto end = std::min({bitset.size(), result_.size(), limit});
        for (auto offset = bitset.find_first(true);
             offset.has_value() && offset.value() < end;
             offset = bitset.find_next(offset.value(), true)) {
            result_.set(offset.value());
        }
    }

    void
    OrShifted(const TargetBitmapView& bitset, size_t offset) {
        if (offset >= result_.size()) {
            return;
        }
        const auto end = std::min(bitset.size(), result_.size() - offset);
        for (auto value = bitset.find_first(true);
             value.has_value() && value.value() < end;
             value = bitset.find_next(value.value(), true)) {
            result_.set(offset + value.value());
        }
    }

    void
    OrShifted(const BitmapVector& other, size_t offset) {
        other.result_.iterate([&](size_t value) {
            if (offset + value < result_.size()) {
                result_.set(offset + value);
            }
            return true;
        });
    }

    std::shared_ptr<BitmapVector>
    Slice(size_t offset, size_t size) const {
        return std::make_shared<BitmapVector>(result_.slice(offset, size),
                                              validity_.slice(offset, size));
    }

    uint32_t
    SelectTrue(uint32_t rank) const {
        uint32_t selected = 0;
        uint64_t current = 0;
        result_.iterate([&](size_t value) {
            if (current++ == rank) {
                AssertInfo(value <= std::numeric_limits<uint32_t>::max(),
                           "selected bitmap offset {} exceeds uint32_t",
                           value);
                selected = static_cast<uint32_t>(value);
                return false;
            }
            return true;
        });
        AssertInfo(rank < current, "true rank {} out of bounds", rank);
        return selected;
    }

    uint32_t
    SelectFalse(uint32_t rank) const {
        uint32_t seen = 0;
        for (size_t value = 0; value < result_.size(); ++value) {
            if (!result_.test(value) && seen++ == rank) {
                return static_cast<uint32_t>(value);
            }
        }
        ThrowInfo(
            ErrorCode::UnexpectedError, "false rank {} out of bounds", rank);
    }

    void
    AppendValidValuesTo(TargetBitmap& dst, size_t offset, size_t size) const {
        dst.append(validity_.slice(offset, size).to_dense());
    }

    void
    set_valid_values(TargetBitmap&& validity) {
        AssertInfo(validity.size() == length_,
                   "valid bitmap size {} does not match {}",
                   validity.size(),
                   length_);
        validity_ = Bitmap(std::move(validity));
    }

 private:
    Bitmap result_;
    Bitmap validity_;
};

using BitmapVectorPtr = std::shared_ptr<BitmapVector>;

}  // namespace milvus
