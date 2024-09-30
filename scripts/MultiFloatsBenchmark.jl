using Printf
using Random
using LinearAlgebra
using GenericLinearAlgebra

using MultiFloats  # Float64x2
using DoubleFloats # Double64
using Quadmath     # Float128
using DecFP        # Dec128
using ArbNumerics  # ArbFloat

setextrabits(0)
setprecision(BigFloat, 106)
setprecision(ArbFloat, 106)

function test_qr(::Type{T}, n::Int) where {T}
    Random.seed!(0)
    A = T.(rand(Float64, n, n))
    result = @timed qr(A)
    num_correct_digits = setprecision(BigFloat, 256) do
        A_big = BigFloat.(A)
        Q_big = BigFloat.(result.value.Q)
        R_big = BigFloat.(result.value.R)
        return -log10(Float64(sum(abs.(Q_big * R_big - A_big))))
    end
    @printf("%s |  qr  | %.1f | %.6f | %.6f | %d\n", rpad(T, 24),
        num_correct_digits, result.time, result.gctime, result.bytes)
end

function test_pinv(::Type{T}, m::Int, n::Int) where {T}
    Random.seed!(0)
    A = T.(rand(Float64, m, n))
    result = @timed pinv(A)
    num_correct_digits = setprecision(BigFloat, 256) do
        A_big = BigFloat.(A)
        Apinv = BigFloat.(result.value)
        return -log10(Float64(sum(abs.(A_big * Apinv * A_big - A_big))))
    end
    @printf("%s | pinv | %.1f | %.6f | %.6f | %d\n", rpad(T, 24),
        num_correct_digits, result.time, result.gctime, result.bytes)
end

const N = 400
const M = 250

function main()
    while true
        test_qr(Float64x2, N)
        test_qr(Double64, N)
        test_qr(Float128, N)
        # test_qr(Dec128, N)
        test_qr(BigFloat, N)
        # test_qr(ArbFloat, N)

        test_pinv(Float64x2, N, M)
        test_pinv(Double64, N, M)
        test_pinv(Float128, N, M)
        # test_pinv(Dec128, N, M)
        test_pinv(BigFloat, N, M)
        # test_pinv(ArbFloat, N, M)
    end
end

main()
