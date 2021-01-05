using Random
using LinearAlgebra
using GenericSVD

using ArbNumerics  # ArbFloat
using Decimals     # Decimal
using DecFP        # Decimal128
using DoubleFloats # Double64
using Quadmath     # Float128
using MultiFloats  # Float64x2

setextrabits(0)
setprecision(BigFloat, 106)
setprecision(ArbFloat, 106)

function test_qr(::Type{T}, n::Int) where {T}
    Random.seed!(0)
    A = T.(rand(Float64, n, n))
    result, elapsed_time, _, _, _ = @timed qr(A)
    num_correct_digits = -log10(Float64(sum(abs.(result.Q * result.R - A))))
    println(rpad(T, 30), " |  qr  | ", round(num_correct_digits, digits=1), " | ", elapsed_time)
end

function test_pinv(::Type{T}, m::Int, n::Int) where {T}
    Random.seed!(0)
    A = T.(rand(Float64, m, n))
    Apinv, elapsed_time, _, _, _ = @timed pinv(A)
    num_correct_digits = -log10(Float64(sum(abs.(A * Apinv * A - A))))
    println(rpad(T, 30), " | pinv | ", round(num_correct_digits, digits=1), " | ", elapsed_time)
end

const N = 40
const M = 25

function main()
    while true
        test_qr(Float64x2, N)
        test_qr(BigFloat, N)
        test_qr(ArbFloat, N)
        test_qr(Dec128, N)
        test_qr(Double64, N)
        test_qr(Float128, N)
        test_pinv(Float64x2, N, M)
        test_pinv(BigFloat, N, M)
        test_pinv(ArbFloat, N, M)
        test_pinv(Double64, N, M)
        test_pinv(Float128, N, M)
    end
end

main()
