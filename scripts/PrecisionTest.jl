using MultiFloats
using Printf


function sqrt_precision(x::MultiFloat{T,N}) where {T,N}
    setprecision(BigFloat, 2200) do
        approx = BigFloat(sqrt(x))
        exact = sqrt(BigFloat(x))
        -Float64(log2(abs((approx - exact) / exact)))
    end
end


function exp_precision(x::MultiFloat{T,N}) where {T,N}
    setprecision(BigFloat, 2200) do
        approx = BigFloat(exp(x))
        exact = exp(BigFloat(x))
        -Float64(log2(abs((approx - exact) / exact)))
    end
end


num_trials = 10^6

println("SQRT (CLEAN)")
use_clean_multifloat_arithmetic()
for N = 2 : 8
    correctbits = floor(Int, minimum(
            sqrt_precision(ldexp(rand(Float64x{N}), rand(-300:300)))
            for _ = 1 : num_trials))
    @printf("Float64x%d: %.3d bits (%2d lost, %.3f%% loss)\n", N, correctbits,
        precision(Float64x{N}) - correctbits,
        100 * abs(precision(Float64x{N}) - correctbits) / precision(Float64x{N}))
end
println()

println("SQRT (STANDARD)")
use_standard_multifloat_arithmetic()
for N = 2 : 8
    correctbits = floor(Int, minimum(
            sqrt_precision(ldexp(rand(Float64x{N}), rand(-300:300)))
            for _ = 1 : num_trials))
    @printf("Float64x%d: %.3d bits (%2d lost, %.3f%% loss)\n", N, correctbits,
        precision(Float64x{N}) - correctbits,
        100 * abs(precision(Float64x{N}) - correctbits) / precision(Float64x{N}))
end
println()

println("SQRT (SLOPPY)")
use_sloppy_multifloat_arithmetic()
for N = 2 : 8
    correctbits = floor(Int, minimum(
            sqrt_precision(ldexp(rand(Float64x{N}), rand(-300:300)))
            for _ = 1 : num_trials))
    @printf("Float64x%d: %.3d bits (%2d lost, %.3f%% loss)\n", N, correctbits,
        precision(Float64x{N}) - correctbits,
        100 * abs(precision(Float64x{N}) - correctbits) / precision(Float64x{N}))
end
println()


println("EXP (CLEAN)")
use_clean_multifloat_arithmetic()
for N = 2 : 8
    correctbits = floor(Int, minimum(
            exp_precision(400 * rand(Float64x{N}) - 200)
            for _ = 1 : num_trials))
    @printf("Float64x%d: %.3d bits (%2d lost, %.3f%% loss)\n", N, correctbits,
        precision(Float64x{N}) - correctbits,
        100 * abs(precision(Float64x{N}) - correctbits) / precision(Float64x{N}))
end
println()

println("EXP (STANDARD)")
use_standard_multifloat_arithmetic()
for N = 2 : 8
    correctbits = floor(Int, minimum(
            exp_precision(400 * rand(Float64x{N}) - 200)
            for _ = 1 : num_trials))
    @printf("Float64x%d: %.3d bits (%2d lost, %.3f%% loss)\n", N, correctbits,
        precision(Float64x{N}) - correctbits,
        100 * abs(precision(Float64x{N}) - correctbits) / precision(Float64x{N}))
end
println()

println("EXP (SLOPPY)")
use_sloppy_multifloat_arithmetic()
for N = 2 : 8
    correctbits = floor(Int, minimum(
            exp_precision(400 * rand(Float64x{N}) - 200)
            for _ = 1 : num_trials))
    @printf("Float64x%d: %.3d bits (%2d lost, %.3f%% loss)\n", N, correctbits,
        precision(Float64x{N}) - correctbits,
        100 * abs(precision(Float64x{N}) - correctbits) / precision(Float64x{N}))
end
println()
