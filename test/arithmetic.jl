function common_bits(x::BigFloat, y::BigFloat)
    d = x - y
    if iszero(d)
        return min(precision(x), precision(y))
    end
    return max(exponent(x), exponent(y)) - exponent(d)
end

function test_unary_operation(
    f::F,
    ::Type{MultiFloat{T,N}},
    deficit::Int,
    num_trials::Int;
    precise_condition::P=(_, _, _) -> true,
    nan_condition::Q=_ -> false,
    positive_inputs::Bool=false,
) where {F,T,N,P,Q}
    e_lo = exponent(floatmin(T)) + (N - 1) * precision(T)
    e_hi = exponent(floatmax(T)) - (N - 1) * precision(T)
    p = precision(MultiFloat{T,N})
    setprecision(BigFloat, 2 * MultiFloats._full_precision(T) + 1) do
        setrounding(BigFloat, RoundNearest) do
            big_min = BigFloat(nextfloat(zero(T)))
            big_max = BigFloat(floatmax(MultiFloat{T,N}))
            for _ = 1:num_trials
                x = _bit_rand(MultiFloat{T,N})
                if positive_inputs
                    x = abs(x)
                end
                num_allocations = @allocated begin
                    z = f(x)
                end
                @test iszero(num_allocations)

                big_f = f(BigFloat(x))
                if iszero(z)
                    @test abs(big_f) < big_min
                elseif isfinite(z)
                    if precise_condition(e_lo, e_hi, exponent(x))
                        big_z = BigFloat(z)
                        @test (common_bits(big_z, big_f) >= p - deficit) ||
                              (abs(big_z - big_f) < N * big_min)
                    end
                else
                    @test (big_f < -big_max) || (big_f > +big_max) ||
                          nan_condition(x)
                end
            end
        end
    end
end

function test_binary_operation(
    f::F,
    ::Type{MultiFloat{T,N}},
    deficit::Int,
    num_trials::Int;
    precise_condition::P=(_, _, _, _) -> true,
    nan_condition::Q=(_, _) -> false,
) where {F,T,N,P,Q}
    e_lo = exponent(floatmin(T)) + (N - 1) * precision(T)
    e_hi = exponent(floatmax(T)) - (N - 1) * precision(T)
    p = precision(MultiFloat{T,N})
    setprecision(BigFloat, 2 * MultiFloats._full_precision(T) + 1) do
        setrounding(BigFloat, RoundNearest) do
            big_min = BigFloat(nextfloat(zero(T)))
            big_max = BigFloat(floatmax(MultiFloat{T,N}))
            for _ = 1:num_trials
                x = _bit_rand(MultiFloat{T,N})
                y = _bit_rand(MultiFloat{T,N})
                num_allocations = @allocated begin
                    z = f(x, y)
                end
                @test iszero(num_allocations)

                big_f = f(BigFloat(x), BigFloat(y))
                if iszero(z)
                    @test abs(big_f) < big_min
                elseif isfinite(z)
                    if precise_condition(e_lo, e_hi, exponent(x), exponent(y))
                        big_z = BigFloat(z)
                        @test (common_bits(big_z, big_f) >= p - deficit) ||
                              (abs(big_z - big_f) < N * big_min)
                    end
                else
                    @test (big_f < -big_max) || (big_f > +big_max) ||
                          nan_condition(x, y)
                end
            end
        end
    end
end

@testset "addition and subtraction" begin
    for T in _MF_TYPES
        test_binary_operation(+, T, 0, 2^18)
        test_binary_operation(-, T, 0, 2^18)
    end
end

@testset "multiplication" begin
    for T in _MF_TYPES
        test_binary_operation(*, T, 1, 2^19)
    end
end

@testset "reciprocal" begin
    for T in _MF_TYPES
        test_unary_operation(inv, T, 2, 2^17;
            precise_condition=(_, e_hi, ex) -> (ex <= e_hi))
        test_unary_operation(MultiFloats.inv_r, T, 2, 2^17;
            precise_condition=(_, e_hi, ex) -> (ex <= e_hi))
    end
end

@testset "division" begin
    for T in _MF_TYPES
        test_binary_operation(/, T, 2, 2^17;
            precise_condition=(e_lo, e_hi, ex, ey) ->
                (ex >= e_lo) & (ey <= e_hi) & (ex - ey >= e_lo),
            nan_condition=(_, y) -> issubnormal(y))
        test_binary_operation(MultiFloats.div_r, T, 2, 2^17;
            precise_condition=(e_lo, e_hi, ex, ey) ->
                (ex >= e_lo) & (ey <= e_hi) & (ex - ey >= e_lo),
            nan_condition=(_, y) -> issubnormal(y))
    end
end

@testset "reciprocal square root" begin
    for T in _MF_TYPES
        test_unary_operation(MultiFloats.rsqrt_r, T, 3, 2^18;
            precise_condition=(_, e_hi, ex) -> (ex <= e_hi),
            nan_condition=issubnormal, positive_inputs=true)
    end
end

@testset "square root" begin
    for T in _MF_TYPES
        test_unary_operation(sqrt, T, 3, 2^17;
            precise_condition=(e_lo, e_hi, ex) -> (e_lo <= ex <= e_hi),
            nan_condition=issubnormal, positive_inputs=true)
        test_unary_operation(MultiFloats.sqrt_r, T, 3, 2^17;
            precise_condition=(e_lo, e_hi, ex) -> (e_lo <= ex <= e_hi),
            nan_condition=issubnormal, positive_inputs=true)
    end
end

@testset "reciprocal cube root" begin
    for T in _MF_TYPES
        test_unary_operation(MultiFloats.rcbrt, T, 5, 2^17;
            precise_condition=(_, e_hi, ex) -> (ex <= e_hi),
            nan_condition=issubnormal)
    end
end

@testset "cube root" begin
    for T in _MF_TYPES
        test_unary_operation(cbrt, T, 6, 2^17;
            precise_condition=(e_lo, e_hi, ex) -> (e_lo <= ex <= e_hi),
            nan_condition=issubnormal)
    end
end

@testset "atan (one argument)" begin
    for T in _MF_TYPES
        test_unary_operation(atan, T, 4, 2^16)
    end
end

@testset "atan (two argument)" begin
    for T in _MF_TYPES
        # atan(y, x) computes y/x internally, so only assert accuracy where that
        # ratio is representable (same condition as the division testset; the
        # first argument y is the numerator, the second x the denominator).
        test_binary_operation(atan, T, 4, 2^16;
            precise_condition=(e_lo, e_hi, ey, ex) ->
                (ey >= e_lo) & (ex <= e_hi) & (ey - ex >= e_lo))
    end
    # quadrant, axis, and signed-zero origin cases against BigFloat.
    for T in _MF_TYPES
        S = T.parameters[1]
        p = precision(T)
        setprecision(BigFloat, 2 * MultiFloats._full_precision(S) + 1) do
            # Note: -0.0 second argument is omitted because Float32xN (N>=2)
            # does not preserve negative zero through construction.
            for (yv, xv) in ((3, 4), (3, -4), (-3, -4), (-3, 4),
                             (1, 0), (-1, 0), (0, 1), (0, -1), (0.0, 0.0))
                z = BigFloat(atan(T(yv), T(xv)))
                ref = atan(BigFloat(yv), BigFloat(xv))
                @test abs(z - ref) <= ldexp(abs(ref), 4 - p) + eps(BigFloat)
            end
            # Overflow-avoidance and sign/quadrant coverage at extreme magnitudes,
            # where y/x would overflow (so the smaller/larger swap is required)
            # and the result is an exact-angle limit. Cases whose true result is a
            # subnormal tiny ratio are omitted (they hit the division precision
            # limit, not atan's).
            hi = floatmax(T)
            tiny = floatmin(T)
            for (y, x) in ((hi, one(T)), (hi, -one(T)),     # -> +pi/2
                           (-hi, one(T)), (-hi, -one(T)),   # -> -pi/2
                           (one(T), -hi), (-one(T), -hi),   # -> +-pi
                           (hi, tiny), (-hi, tiny),         # -> +-pi/2 (huge ratio)
                           (hi, -tiny), (-hi, -tiny))       # -> +-pi/2 (huge ratio)
                z = BigFloat(atan(y, x))
                ref = atan(BigFloat(y), BigFloat(x))
                @test abs(z - ref) <= ldexp(abs(ref), 4 - p) + eps(BigFloat)
            end
        end
    end
end
