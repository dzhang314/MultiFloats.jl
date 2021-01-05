push!(LOAD_PATH, "C:\\Users\\Zhang\\Documents\\GitHub")
set_zero_subnormals(true)

using DZLinearAlgebra: find_saturation_threshold
using MultiFloats
using Printf


const INVERSE_FACTORIALS_F64 = setprecision(() ->
    [Float64x{20}(inv(BigFloat(factorial(BigInt(i))))) for i = 1 : 170],
    BigFloat, 1200)


const LOG2_F64 = setprecision(() ->
    Float64x{20}(log(BigFloat(2))),
    BigFloat, 1200)


function test_multifloat_exp(x::Float64x{N}, num_terms::Int=170,
        reduction_power::Int=0, ::Val{B}=Val(false)) where {N,B}
    exponent_f = Base.rint_llvm(x._limbs[1] / 0.6931471805599453)
    exponent_i = Base.fptosi(Int, exponent_f)
    y = ldexp(Float64x{N}(Float64x{N+!B}(x) -
            exponent_f * Float64x{N+!B}(LOG2_F64)), -reduction_power)
    terms = Vector{Float64x{N}}(undef, num_terms)
    @inbounds terms[1] = y
    for i = 2 : num_terms
        @inbounds terms[i] = terms[div(i, 2)] * terms[div(i+1, 2)]
    end
    exp_y = terms[num_terms] * Float64x{N}(INVERSE_FACTORIALS_F64[num_terms])
    for i = num_terms-1 : -1 : 1
        exp_y += terms[i] * Float64x{N}(INVERSE_FACTORIALS_F64[i])
    end
    exp_y += 1.0
    for _ = 1 : reduction_power
        exp_y *= exp_y
    end
    return ldexp(exp_y, exponent_i)
end


function test_exp_accuracy(x::Float64x{N}, args...) where {N}
    setprecision(3000) do
        exp_x = test_multifloat_exp(x, args...)
        exp_exact = exp(BigFloat(x))
        -Float64(log2(abs((BigFloat(exp_x) - exp_exact) / exp_exact)))
    end
end


function run_test(num_trials::Int, max_reduction_power::Int, ::Val{B}) where {B}
    for N = 1 : 8
        for reduction_power = 0 : max_reduction_power
            data = [(Float64(num_terms), minimum(test_exp_accuracy(
                    100 * rand(Float64x{N}) - 50,
                    num_terms, reduction_power, Val{B}())
                for _ = 1 : num_trials)) for num_terms = 1 : 170]
            threshold = round(Int, find_saturation_threshold(data)[1]) + 1
            lost_bits = precision(Float64x{N}) - floor(Int,
                minimum(p[2] for p in data[threshold:end]))
            @printf("Float64x%d (power %2d): %3d terms; %3d bits\n",
                N, reduction_power, threshold, lost_bits)
        end
        println()
    end
end


num_trials = 1000
max_reduction_power = 20

use_clean_multifloat_arithmetic(9)
println("CLEAN (EXTRA BIT)")
run_test(num_trials, max_reduction_power, Val{false}())
println("CLEAN (NO EXTRA)")
run_test(num_trials, max_reduction_power, Val{true}())

use_standard_multifloat_arithmetic(9)
println("STANDARD (EXTRA BIT)")
run_test(num_trials, max_reduction_power, Val{false}())
println("STANDARD (NO EXTRA)")
run_test(num_trials, max_reduction_power, Val{true}())

use_sloppy_multifloat_arithmetic(9)
println("SLOPPY (EXTRA BIT)")
run_test(num_trials, max_reduction_power, Val{false}())
println("SLOPPY (NO EXTRA)")
run_test(num_trials, max_reduction_power, Val{true}())
