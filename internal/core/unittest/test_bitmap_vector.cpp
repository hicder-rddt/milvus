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

#include <gtest/gtest.h>

#include <limits>

#include "common/BitmapVector.h"

namespace milvus {
namespace {

TargetBitmap
Dense(std::initializer_list<size_t> values, size_t size) {
    TargetBitmap bitmap(size, false);
    for (auto value : values) {
        bitmap.set(value);
    }
    return bitmap;
}

void
ExpectDenseEq(const TargetBitmap& actual, const TargetBitmap& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i], expected[i]) << "bit " << i;
    }
}

TEST(BitmapTest, DefaultsToRoaringAndSupportsUnalignedDomain) {
    Bitmap bitmap(67, false);
    EXPECT_TRUE(bitmap.is_roaring());
    bitmap.set(0);
    bitmap.set(65);
    bitmap.set(66);
    EXPECT_EQ(bitmap.count(), 3);

    bitmap.flip();
    EXPECT_EQ(bitmap.count(), 64);
    EXPECT_FALSE(bitmap.test(0));
    EXPECT_FALSE(bitmap.test(65));
    EXPECT_FALSE(bitmap.test(66));
}

TEST(BitmapTest, MixedBackendsHaveSetOperationParity) {
    Bitmap roaring(Dense({1, 2, 63, 64}, 65));
    Bitmap dense(Dense({2, 3, 64}, 65), Bitmap::Backend::Dense);

    auto intersection = roaring.clone();
    intersection.and_with(dense);
    ExpectDenseEq(intersection.to_dense(), Dense({2, 64}, 65));

    auto union_result = dense.clone();
    union_result.or_with(roaring);
    ExpectDenseEq(union_result.to_dense(), Dense({1, 2, 3, 63, 64}, 65));

    auto xor_result = roaring.clone();
    xor_result.xor_with(dense);
    ExpectDenseEq(xor_result.to_dense(), Dense({1, 3, 63}, 65));

    auto difference = roaring.clone();
    difference.and_not_with(dense);
    ExpectDenseEq(difference.to_dense(), Dense({1, 63}, 65));
}

TEST(BitmapTest, CloneSliceAppendAndRangesPreserveLogicalSize) {
    Bitmap bitmap(13, false);
    bitmap.set_range(2, 7, true);
    bitmap.set_range(4, 6, false);

    auto clone = bitmap.clone();
    clone.set(12);
    EXPECT_FALSE(bitmap.test(12));

    auto slice = bitmap.slice(1, 8);
    EXPECT_EQ(slice.size(), 8);
    ExpectDenseEq(slice.to_dense(), Dense({1, 2, 5}, 8));

    Bitmap suffix(Dense({0, 2}, 3));
    slice.append(suffix);
    EXPECT_EQ(slice.size(), 11);
    ExpectDenseEq(slice.to_dense(), Dense({1, 2, 5, 8, 10}, 11));
}

TEST(BitmapTest, BackendBoundaryDecisionIsExplicit) {
    constexpr auto max_domain =
        static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
    EXPECT_TRUE(Bitmap::CanUseRoaring(max_domain));
    if constexpr (std::numeric_limits<size_t>::max() > max_domain) {
        EXPECT_FALSE(Bitmap::CanUseRoaring(max_domain + 1));
    }
}

TEST(BitmapTest, RangeConversionHandlesBackendsAndContainerBoundaries) {
    auto source =
        Dense({0, 62, 63, 64, 65, 65534, 65535, 65536, 65537, 70000}, 70003);
    Bitmap roaring(source);
    Bitmap dense(source.clone(), Bitmap::Backend::Dense);

    for (const auto* bitmap : {&roaring, &dense}) {
        ExpectDenseEq(bitmap->to_dense(0, 0), Dense({}, 0));
        ExpectDenseEq(bitmap->to_dense(0, bitmap->size()), source);
        ExpectDenseEq(bitmap->to_dense(62, 5), Dense({0, 1, 2, 3}, 5));
        ExpectDenseEq(bitmap->to_dense(65534, 5), Dense({0, 1, 2, 3}, 5));
        ExpectDenseEq(bitmap->to_dense(70000, 3), Dense({0}, 3));

        TargetBitmap appended = Dense({0}, 1);
        bitmap->append_range_to(appended, 65534, 5);
        ExpectDenseEq(appended, Dense({0, 1, 2, 3, 4}, 6));
    }
}

TEST(BitmapTest, GatherSupportsOrderingDuplicatesAndEmptyInput) {
    Bitmap roaring(Dense({1, 3, 7, 64}, 67));
    Bitmap dense(Dense({1, 3, 7, 64}, 67), Bitmap::Backend::Dense);
    const std::vector<int32_t> offsets = {64, 1, 64, 2, 7, 3};
    const std::vector<int32_t> empty;

    for (const auto* bitmap : {&roaring, &dense}) {
        ExpectDenseEq(bitmap->gather_to_dense(offsets),
                      Dense({0, 1, 2, 4, 5}, offsets.size()));
        ExpectDenseEq(bitmap->gather_to_dense(empty), Dense({}, 0));
    }
}

TEST(BitmapVectorTest, ThreeValuedLogicStaysRoaring) {
    // TRUE, FALSE, NULL
    BitmapVector left(Bitmap(Dense({0}, 3)), Bitmap(Dense({0, 1}, 3)));
    // NULL, TRUE, FALSE
    BitmapVector right(Bitmap(Dense({1}, 3)), Bitmap(Dense({1, 2}, 3)));

    auto and_result = left.Clone();
    and_result->And(right);
    EXPECT_TRUE(and_result->result().is_roaring());
    EXPECT_TRUE(and_result->validity().is_roaring());
    ExpectDenseEq(and_result->result().to_dense(), Dense({}, 3));
    ExpectDenseEq(and_result->validity().to_dense(), Dense({1, 2}, 3));

    auto or_result = left.Clone();
    or_result->Or(right);
    EXPECT_TRUE(or_result->result().is_roaring());
    EXPECT_TRUE(or_result->validity().is_roaring());
    ExpectDenseEq(or_result->result().to_dense(), Dense({0, 1}, 3));
    ExpectDenseEq(or_result->validity().to_dense(), Dense({0, 1}, 3));
}

}  // namespace
}  // namespace milvus
