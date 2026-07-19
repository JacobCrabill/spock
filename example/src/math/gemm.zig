//! Naive DGEMM implementation

/// Routine similar to BLAS's `dgemm` but does not allow for transposes.
///
/// Performs C := alpha*A*B + beta*C
///
/// Matrices are given in column-major memory layout
///
/// Just as an alternative to the BLAS routines in case a standalone implementation is required
///
/// Arows - No. of rows of matrices A and C
/// Bcols - No. of columns of matrices B and C
/// Acols - No. of columns of A or No. of rows of B
pub fn dgemm(Arows: u32, Bcols: u32, Acols: u32, alpha: f64, beta: f64, A: []const f64, B: []const f64, C: []f64) void {
    const empy_matrix = (Arows == 0) or (Bcols == 0);
    const only_c = ((alpha == 0.0) or (Acols == 0)) and beta == 1.0;
    if (empy_matrix or only_c) {
        return;
    }

    if (beta == 0.0) {
        @memset(C, 0.0);
    }

    if (alpha == 0.0) {
        if (beta != 0.0) {
            for (0..Bcols) |j| {
                for (0..Arows) |i| {
                    C[i + j * Arows] = beta * C[i + j * Arows];
                }
            }
        }
        return;
    }

    // Otherwise, perform full operation
    var temp: f64 = undefined;
    for (0..Bcols) |j| {
        if (beta != 1.0) {
            for (0..Arows) |i| {
                C[i + j * Arows] = beta * C[i + j * Arows];
            }
        }

        for (0..Acols) |l| {
            temp = alpha * B[l + j * Acols];

            for (0..Arows) |i| {
                C[i + j * Arows] += temp * A[i + l * Arows];
            }
        }
    }
}

test dgemm {
    var A = [_]f64{ 1, 2, 3, 4 };
    var B = [_]f64{ 1, 1, 2, 2 };
    var C = [_]f64{ 0, 0, 0, 0 };

    dgemm(2, 2, 2, 1.0, 1.0, A[0..], B[0..], C[0..]);

    // 1 3   1 2  ==  4 8
    // 2 4   1 2      6 12

    const D = [_]f64{ 4, 6, 8, 12 };
    for (D, C) |d, c| {
        try std.testing.expectEqual(d, c);
    }
}

const std = @import("std");
