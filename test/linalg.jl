using LinearAlgebra: I, lu, norm, opnorm, qr


function test_lu(::Type{T}, n::Int, num_trials::Int) where {T}
    for _ = 1:num_trials
        A = randn(T, n, n)
        F = lu(A)
        residual = norm(F.L * F.U - A[F.p, :], Inf)
        @test residual <= 2 * opnorm(A, Inf) * eps(T)
    end
end


@testset "LU decomposition (precompile)" begin
    for T in _MF_TYPES
        test_lu(T, 4, 1)
    end
end

@testset "LU decomposition (4x4)" begin
    for T in _MF_TYPES
        test_lu(T, 4, 2^15)
    end
end

@testset "LU decomposition (5x5)" begin
    for T in _MF_TYPES
        test_lu(T, 5, 2^14)
    end
end

@testset "LU decomposition (12x12)" begin
    for T in _MF_TYPES
        test_lu(T, 12, 2^11)
    end
end

@testset "LU decomposition (18x18)" begin
    for T in _MF_TYPES
        test_lu(T, 18, 2^10)
    end
end

@testset "LU decomposition (25x25)" begin
    for T in _MF_TYPES
        test_lu(T, 25, 2^8)
    end
end


function test_qr(::Type{T}, n::Int, num_trials::Int) where {T}
    for _ = 1:num_trials
        A = randn(T, n, n)
        F = qr(A)
        Q = Matrix(F.Q)
        residual = norm(Q * F.R - A, Inf)
        @test residual <= 8 * opnorm(A, Inf) * eps(T)
        overlap = norm(Q' * Q - I, Inf)
        @test overlap <= 8 * eps(T)
    end
end


@testset "QR decomposition (precompile)" begin
    for T in _MF_TYPES
        test_qr(T, 4, 1)
    end
end

@testset "QR decomposition (4x4)" begin
    for T in _MF_TYPES
        test_qr(T, 4, 2^14)
    end
end

@testset "QR decomposition (5x5)" begin
    for T in _MF_TYPES
        test_qr(T, 5, 2^13)
    end
end

@testset "QR decomposition (12x12)" begin
    for T in _MF_TYPES
        test_qr(T, 12, 2^10)
    end
end

@testset "QR decomposition (18x18)" begin
    for T in _MF_TYPES
        test_qr(T, 18, 2^8)
    end
end

@testset "QR decomposition (25x25)" begin
    for T in _MF_TYPES
        test_qr(T, 25, 2^7)
    end
end
