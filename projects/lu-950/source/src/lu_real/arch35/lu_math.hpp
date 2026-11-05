/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Experimental numerical code; not a CPU implementation of the public NPU API.
#pragma once

#ifndef LU950_INLINE
#define LU950_INLINE inline
#endif

namespace lu950
{
LU950_INLINE float Abs(float value) { return value < 0.0f ? -value : value; }

template <class Matrix>
LU950_INLINE float UpdatedElement(Matrix a, int row, int col, int terms)
{
    float value = a.Get(row, col);
    float correction = 0.0f;
    for (int k = 0; k < terms; ++k)
    {
        const float term = -a.Get(row, k) * a.Get(k, col) - correction;
        const float updated = value + term;
        correction = (updated - value) - term;
        value = updated;
    }
    return value;
}

// Matrix and vector accessors use Get/Set so this exact algorithm can operate
// on AscendC LocalTensor objects and independently checked host buffers.
template <class Matrix, class Pivots>
LU950_INLINE int Factor(Matrix a, Pivots pivots, int m, int n, bool pivoting)
{
    int info = 0;
    const int steps = m < n ? m : n;
    for (int k = 0; k < steps; ++k)
    {
        // Left-looking update: retain the unprocessed input and accumulate
        // previous-column contributions with compensation before pivoting.
        for (int row = k; row < m; ++row)
        {
            a.Set(row, k, UpdatedElement(a, row, k, k));
        }
        int pivot = k;
        if (pivoting)
        {
            float largest = Abs(a.Get(k, k));
            for (int row = k + 1; row < m; ++row)
            {
                const float candidate = Abs(a.Get(row, k));
                if (candidate > largest)
                {
                    largest = candidate;
                    pivot = row;
                }
            }
        }
        pivots.Set(k, pivot + 1);
        if (pivot != k)
        {
            for (int col = 0; col < n; ++col)
            {
                const float temp = a.Get(k, col);
                a.Set(k, col, a.Get(pivot, col));
                a.Set(pivot, col, temp);
            }
        }
        for (int col = k + 1; col < n; ++col)
        {
            a.Set(k, col, UpdatedElement(a, k, col, k));
        }
        const float diagonal = a.Get(k, k);
        if (diagonal == 0.0f)
        {
            if (info == 0)
            {
                info = k + 1;
            }
            continue;
        }
        // Divide directly: forming 1/diagonal first overflows for tiny pivots.
        for (int row = k + 1; row < m; ++row)
        {
            a.Set(row, k, a.Get(row, k) / diagonal);
        }
    }
    return info;
}

template <class Matrix, class Output, class Pivots, class Vector>
LU950_INLINE int InvertFactors(Matrix a, Output c, Pivots pivots, Vector rhs, int n, bool pivoting, int firstColumn = 0,
                               int columnStride = 1)
{
    for (int row = 0; row < n; ++row)
    {
        if (a.Get(row, row) == 0.0f)
        {
            return row + 1;
        }
        if (pivoting && (pivots.Get(row) < row + 1 || pivots.Get(row) > n))
        {
            return -1;  // Internal helper code, not a public API argument index.
        }
    }
    for (int col = firstColumn; col < n; col += columnStride)
    {
        for (int row = 0; row < n; ++row)
        {
            rhs.Set(row, row == col ? 1.0f : 0.0f);
        }
        if (pivoting)
        {
            for (int k = 0; k < n; ++k)
            {
                const int pivot = pivots.Get(k) - 1;
                const float temp = rhs.Get(k);
                rhs.Set(k, rhs.Get(pivot));
                rhs.Set(pivot, temp);
            }
        }
        for (int row = 0; row < n; ++row)
        {
            float value = rhs.Get(row);
            float correction = 0.0f;
            for (int k = 0; k < row; ++k)
            {
                const float term = -a.Get(row, k) * rhs.Get(k) - correction;
                const float updated = value + term;
                correction = (updated - value) - term;
                value = updated;
            }
            rhs.Set(row, value);
        }
        for (int row = n - 1; row >= 0; --row)
        {
            float value = rhs.Get(row);
            float correction = 0.0f;
            for (int k = row + 1; k < n; ++k)
            {
                const float term = -a.Get(row, k) * rhs.Get(k) - correction;
                const float updated = value + term;
                correction = (updated - value) - term;
                value = updated;
            }
            rhs.Set(row, value / a.Get(row, row));
        }
        for (int row = 0; row < n; ++row)
        {
            c.Set(row, col, rhs.Get(row));
        }
    }
    return 0;
}
// Small matrices follow the U-inverse, L-removal and reverse-pivot order of LAPACK.
template <class Matrix, class Output, class Pivots>
LU950_INLINE int InvertSmallLapack(Matrix a, Output c, Pivots pivots, int n, bool pivoting)
{
    for (int row = 0; row < n; ++row)
    {
        if (a.Get(row, row) == 0.0f) return row + 1;
        if (pivoting && (pivots.Get(row) < row + 1 || pivots.Get(row) > n)) return -1;
    }
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row) c.Set(row, col, 0.0f);
    for (int col = 0; col < n; ++col)
    {
        const float reciprocal = 1.0f / a.Get(col, col);
        for (int row = 0; row < col; ++row)
        {
            float value = 0.0f;
            for (int k = row; k < col; ++k) value -= c.Get(row, k) * a.Get(k, col);
            c.Set(row, col, value * reciprocal);
        }
        c.Set(col, col, reciprocal);
    }
    for (int col = n - 1; col >= 0; --col)
        for (int row = 0; row < n; ++row)
        {
            float value = 0.0f;
            for (int k = col + 1; k < n; ++k) value -= c.Get(row, k) * a.Get(k, col);
            c.Set(row, col, c.Get(row, col) + value);
        }
    if (pivoting)
        for (int col = n - 1; col >= 0; --col)
        {
            const int pivot = pivots.Get(col) - 1;
            for (int row = 0; row < n; ++row)
            {
                const float value = c.Get(row, col);
                c.Set(row, col, c.Get(row, pivot));
                c.Set(row, pivot, value);
            }
        }
    return 0;
}
}  // namespace lu950
