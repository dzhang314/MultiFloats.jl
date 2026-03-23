using LinearAlgebra: I, LowerTriangular, UnitLowerTriangular,
    UnitUpperTriangular, UpperTriangular, cholesky, cond, lu, norm, opnorm, qr


const SHIFT = 0


function test_lu(::Type{T}, n::Int, num_trials::Int) where {T}
    for _ = 1:num_trials
        A = randn(T, n, n)
        F = lu(A)
        residual = norm(F.L * F.U - A[F.p, :], Inf)
        @test residual <= 4 * opnorm(A, Inf) * eps(T) # 2 confirmed too tight
    end
end


@testset "LU decomposition (precompile)" begin
    for T in _MF_TYPES
        test_lu(T, 4, 1)
    end
end

@testset "LU decomposition (4x4)" begin
    for T in _MF_TYPES
        test_lu(T, 4, 2^(15 + SHIFT))
    end
end

@testset "LU decomposition (5x5)" begin
    for T in _MF_TYPES
        test_lu(T, 5, 2^(14 + SHIFT))
    end
end

@testset "LU decomposition (12x12)" begin
    for T in _MF_TYPES
        test_lu(T, 12, 2^(11 + SHIFT))
    end
end

@testset "LU decomposition (18x18)" begin
    for T in _MF_TYPES
        test_lu(T, 18, 2^(10 + SHIFT))
    end
end

@testset "LU decomposition (25x25)" begin
    for T in _MF_TYPES
        test_lu(T, 25, 2^(8 + SHIFT))
    end
end


function test_qr(::Type{T}, n::Int, num_trials::Int) where {T}
    for _ = 1:num_trials
        A = randn(T, n, n)
        F = qr(A)
        Q = Matrix(F.Q)
        residual = norm(Q * F.R - A, Inf)
        @test residual <= 8 * opnorm(A, Inf) * eps(T) # 4 confirmed too tight
        overlap = norm(Q' * Q - I, Inf)
        @test overlap <= 16 * eps(T) # 8 confirmed too tight
    end
end


@testset "QR decomposition (precompile)" begin
    for T in _MF_TYPES
        test_qr(T, 4, 1)
    end
end

@testset "QR decomposition (4x4)" begin
    for T in _MF_TYPES
        test_qr(T, 4, 2^(14 + SHIFT))
    end
end

@testset "QR decomposition (5x5)" begin
    for T in _MF_TYPES
        test_qr(T, 5, 2^(13 + SHIFT))
    end
end

@testset "QR decomposition (12x12)" begin
    for T in _MF_TYPES
        test_qr(T, 12, 2^(10 + SHIFT))
    end
end

@testset "QR decomposition (18x18)" begin
    for T in _MF_TYPES
        test_qr(T, 18, 2^(8 + SHIFT))
    end
end

@testset "QR decomposition (25x25)" begin
    for T in _MF_TYPES
        test_qr(T, 25, 2^(7 + SHIFT))
    end
end


function test_cholesky(::Type{T}, n::Int, num_trials::Int) where {T}
    for _ = 1:num_trials
        A = randn(T, n, n)
        A = A' * A + I
        F = cholesky(A)
        residual = norm(F.L * F.L' - A, Inf)
        @test residual <= 8 * opnorm(A, Inf) * eps(T) # 4 confirmed too tight
    end
end


@testset "Cholesky decomposition (precompile)" begin
    for T in _MF_TYPES
        test_cholesky(T, 4, 1)
    end
end

@testset "Cholesky decomposition (4x4)" begin
    for T in _MF_TYPES
        test_cholesky(T, 4, 2^(15 + SHIFT))
    end
end

@testset "Cholesky decomposition (5x5)" begin
    for T in _MF_TYPES
        test_cholesky(T, 5, 2^(14 + SHIFT))
    end
end

@testset "Cholesky decomposition (12x12)" begin
    for T in _MF_TYPES
        test_cholesky(T, 12, 2^(11 + SHIFT))
    end
end

@testset "Cholesky decomposition (18x18)" begin
    for T in _MF_TYPES
        test_cholesky(T, 18, 2^(9 + SHIFT))
    end
end

@testset "Cholesky decomposition (25x25)" begin
    for T in _MF_TYPES
        test_cholesky(T, 25, 2^(8 + SHIFT))
    end
end


function test_linear_solve(::Type{T}, n::Int, num_trials::Int) where {T}
    for _ = 1:num_trials
        A = randn(T, n, n)
        b = randn(T, n)
        x = A \ b
        residual = norm(A * x - b, Inf)
        # 2 confirmed too tight
        @test residual <= 4 * opnorm(A, Inf) * norm(x, Inf) * eps(T)
    end
end


@testset "linear solve (precompile)" begin
    for T in _MF_TYPES
        test_linear_solve(T, 4, 1)
    end
end

@testset "linear solve (4x4)" begin
    for T in _MF_TYPES
        test_linear_solve(T, 4, 2^(15 + SHIFT))
    end
end

@testset "linear solve (5x5)" begin
    for T in _MF_TYPES
        test_linear_solve(T, 5, 2^(14 + SHIFT))
    end
end

@testset "linear solve (12x12)" begin
    for T in _MF_TYPES
        test_linear_solve(T, 12, 2^(13 + SHIFT))
    end
end

@testset "linear solve (18x18)" begin
    for T in _MF_TYPES
        test_linear_solve(T, 18, 2^(11 + SHIFT))
    end
end

@testset "linear solve (25x25)" begin
    for T in _MF_TYPES
        test_linear_solve(T, 25, 2^(10 + SHIFT))
    end
end


function test_triangular_solve(::Type{T}, n::Int, num_trials::Int) where {T}
    for _ = 1:num_trials
        A = randn(T, n, n)
        b = randn(T, n)
        U = UpperTriangular(A)
        x = U \ b
        residual = norm(U * x - b, Inf)
        # 1 confirmed too tight
        @test residual <= 2 * opnorm(U, Inf) * norm(x, Inf) * eps(T)
        L = LowerTriangular(A)
        x = L \ b
        residual = norm(L * x - b, Inf)
        # 1 confirmed too tight
        @test residual <= 2 * opnorm(L, Inf) * norm(x, Inf) * eps(T)
    end
end


@testset "triangular solve (precompile)" begin
    for T in _MF_TYPES
        test_triangular_solve(T, 4, 1)
    end
end

@testset "triangular solve (4x4)" begin
    for T in _MF_TYPES
        test_triangular_solve(T, 4, 2^(15 + SHIFT))
    end
end

@testset "triangular solve (5x5)" begin
    for T in _MF_TYPES
        test_triangular_solve(T, 5, 2^(14 + SHIFT))
    end
end

@testset "triangular solve (12x12)" begin
    for T in _MF_TYPES
        test_triangular_solve(T, 12, 2^(13 + SHIFT))
    end
end

@testset "triangular solve (18x18)" begin
    for T in _MF_TYPES
        test_triangular_solve(T, 18, 2^(12 + SHIFT))
    end
end

@testset "triangular solve (25x25)" begin
    for T in _MF_TYPES
        test_triangular_solve(T, 25, 2^(11 + SHIFT))
    end
end


function test_unit_triangular_solve(
    ::Type{T},
    n::Int,
    num_trials::Int,
) where {T}
    for _ = 1:num_trials
        A = randn(T, n, n)
        b = randn(T, n)
        U = UnitUpperTriangular(A)
        x = U \ b
        residual = norm(U * x - b, Inf)
        # 1 confirmed too tight
        @test residual <= 2 * opnorm(U, Inf) * norm(x, Inf) * eps(T)
        L = UnitLowerTriangular(A)
        x = L \ b
        residual = norm(L * x - b, Inf)
        # 1 confirmed too tight
        @test residual <= 2 * opnorm(L, Inf) * norm(x, Inf) * eps(T)
    end
end


@testset "unit triangular solve (precompile)" begin
    for T in _MF_TYPES
        test_unit_triangular_solve(T, 4, 1)
    end
end

@testset "unit triangular solve (4x4)" begin
    for T in _MF_TYPES
        test_unit_triangular_solve(T, 4, 2^(15 + SHIFT))
    end
end

@testset "unit triangular solve (5x5)" begin
    for T in _MF_TYPES
        test_unit_triangular_solve(T, 5, 2^(14 + SHIFT))
    end
end

@testset "unit triangular solve (12x12)" begin
    for T in _MF_TYPES
        test_unit_triangular_solve(T, 12, 2^(13 + SHIFT))
    end
end

@testset "unit triangular solve (18x18)" begin
    for T in _MF_TYPES
        test_unit_triangular_solve(T, 18, 2^(12 + SHIFT))
    end
end

@testset "unit triangular solve (25x25)" begin
    for T in _MF_TYPES
        test_unit_triangular_solve(T, 25, 2^(11 + SHIFT))
    end
end


function test_matrix_inverse(::Type{T}, n::Int, num_trials::Int) where {T}
    for _ = 1:num_trials
        A = randn(T, n, n)
        inv_A = inv(A)
        residual = norm(inv_A * A - I, Inf)
        @test residual <= 4 * cond(A, Inf) * eps(T) # 2 confirmed too tight
    end
end


@testset "matrix inverse (precompile)" begin
    for T in _MF_TYPES
        test_matrix_inverse(T, 4, 1)
    end
end

@testset "matrix inverse (4x4)" begin
    for T in _MF_TYPES
        test_matrix_inverse(T, 4, 2^(13 + SHIFT))
    end
end

@testset "matrix inverse (5x5)" begin
    for T in _MF_TYPES
        test_matrix_inverse(T, 5, 2^(12 + SHIFT))
    end
end

@testset "matrix inverse (12x12)" begin
    for T in _MF_TYPES
        test_matrix_inverse(T, 12, 2^(9 + SHIFT))
    end
end

@testset "matrix inverse (18x18)" begin
    for T in _MF_TYPES
        test_matrix_inverse(T, 18, 2^(8 + SHIFT))
    end
end

@testset "matrix inverse (25x25)" begin
    for T in _MF_TYPES
        test_matrix_inverse(T, 25, 2^(7 + SHIFT))
    end
end
