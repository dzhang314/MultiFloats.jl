using MultiFloats
using Test: @test, @testset


const _FIXED_INT_TYPES = (
    Bool, Int8, Int16, Int32, Int64, Int128,
    UInt8, UInt16, UInt32, UInt64, UInt128,
)


const _FIXED_FLOAT_TYPES = (Float16, Float32, Float64)


const _MF_TYPES = (
    Float32x1, Float32x2, Float32x3, Float32x4,
    Float64x1, Float64x2, Float64x3, Float64x4,
)


const _MFV_TYPES = (
    Vec1Float32x1, Vec1Float32x2, Vec1Float32x3, Vec1Float32x4,
    Vec1Float64x1, Vec1Float64x2, Vec1Float64x3, Vec1Float64x4,
    Vec2Float32x1, Vec2Float32x2, Vec2Float32x3, Vec2Float32x4,
    Vec2Float64x1, Vec2Float64x2, Vec2Float64x3, Vec2Float64x4,
    Vec4Float32x1, Vec4Float32x2, Vec4Float32x3, Vec4Float32x4,
    Vec4Float64x1, Vec4Float64x2, Vec4Float64x3, Vec4Float64x4,
    Vec8Float32x1, Vec8Float32x2, Vec8Float32x3, Vec8Float32x4,
    Vec8Float64x1, Vec8Float64x2, Vec8Float64x3, Vec8Float64x4,
)


@inline function _bit_rand(::Type{T}) where {T}
    while true
        x = reinterpret(T, rand(Base.uinttype(T)))
        if isfinite(x)
            return x
        end
    end
end


@inline function _bit_rand(::Type{MultiFloat{T,N}}) where {T,N}
    while true
        x = MultiFloats.renormalize(MultiFloat{T,N}(
            ntuple(_ -> _bit_rand(T), Val{N}())))
        if isfinite(x)
            return x
        end
    end
end


@inline _bit_rand(::Type{MultiFloatVec{M,T,N}}) where {M,T,N} =
    MultiFloatVec{M,T,N}(ntuple(_ -> _bit_rand(MultiFloat{T,N}), Val{M}()))


function test_construction(
    ::Type{MultiFloat{T,N}},
    ::Type{I},
    n::Int,
) where {T,N,I<:Integer}
    setprecision(BigFloat, 2 * MultiFloats._full_precision(T) + 1) do
        setrounding(BigFloat, RoundNearest) do
            for _ = 1:n
                x = rand(I)
                num_allocations = @allocated begin
                    y = MultiFloat{T,N}(x)
                end
                @test num_allocations == 0
                @test MultiFloats.isnormalized(y)
                @test MultiFloats.iscanonical(y)

                if isfinite(y)
                    big_x = BigFloat(x)
                    big_y = BigFloat(y)
                    big_prev_y = BigFloat(prevfloat(y))
                    big_next_y = BigFloat(nextfloat(y))
                    @test big_prev_y < big_y < big_next_y

                    center_diff = abs(big_y - big_x)
                    prev_diff = abs(big_prev_y - big_x)
                    next_diff = abs(big_next_y - big_x)
                    prev_cmp = cmp(center_diff, prev_diff)
                    next_cmp = cmp(center_diff, next_diff)
                    @test ((prev_cmp <= 0) & (next_cmp < 0)) |
                          ((prev_cmp < 0) & (next_cmp <= 0))
                end

                num_allocations = @allocated begin
                    v1 = MultiFloatVec{4,T,N}(x)
                    v2 = MultiFloatVec{4,T,N}(x, x, x, x)
                    v3 = MultiFloatVec{4,T,N}((x, x, x, x))
                    w1 = MultiFloatVec{4,T,N}(y)
                    w2 = MultiFloatVec{4,T,N}(y, y, y, y)
                    w3 = MultiFloatVec{4,T,N}((y, y, y, y))
                end
                @test iszero(num_allocations)
                @test v1 === v2 === v3 === w1 === w2 === w3
            end
        end
    end
end


@testset "construction from integers" begin
    for T in _MF_TYPES, I in _FIXED_INT_TYPES
        test_construction(T, I, 2^12)
    end
end


function test_construction(
    ::Type{MultiFloat{T,N}},
    ::Type{F},
    n::Int,
) where {T,N,F<:AbstractFloat}
    setprecision(BigFloat, 2 * MultiFloats._full_precision(T) + 1) do
        setrounding(BigFloat, RoundNearest) do
            big_max = BigFloat(floatmax(MultiFloat{T,N}))
            for _ = 1:n
                x = _bit_rand(F)
                num_allocations = @allocated begin
                    y = MultiFloat{T,N}(x)
                end
                @test iszero(num_allocations)

                z = MultiFloats._clear_signed_zeros(y)
                @test MultiFloats.isnormalized(z)
                @test MultiFloats.iscanonical(z)

                big_x = BigFloat(x)
                if isfinite(y)
                    big_y = BigFloat(y)
                    big_prev_y = BigFloat(prevfloat(y))
                    big_next_y = BigFloat(nextfloat(y))
                    @test big_prev_y < big_y < big_next_y

                    center_diff = abs(big_y - big_x)
                    prev_diff = abs(big_prev_y - big_x)
                    next_diff = abs(big_next_y - big_x)
                    prev_cmp = cmp(center_diff, prev_diff)
                    next_cmp = cmp(center_diff, next_diff)
                    @test ((prev_cmp <= 0) & (next_cmp < 0)) |
                          ((prev_cmp < 0) & (next_cmp <= 0))
                else
                    @test !isnan(y)
                    if signbit(y)
                        @test big_x < -big_max
                    else
                        @test big_x > +big_max
                    end
                end

                num_allocations = @allocated begin
                    v1 = MultiFloatVec{4,T,N}(x)
                    v2 = MultiFloatVec{4,T,N}(x, x, x, x)
                    v3 = MultiFloatVec{4,T,N}((x, x, x, x))
                    w1 = MultiFloatVec{4,T,N}(y)
                    w2 = MultiFloatVec{4,T,N}(y, y, y, y)
                    w3 = MultiFloatVec{4,T,N}((y, y, y, y))
                end
                @test iszero(num_allocations)
                @test v1 === v2 === v3 === w1 === w2 === w3
            end
        end
    end
end


@testset "construction from floating-point numbers" begin
    for T in _MF_TYPES, F in _FIXED_FLOAT_TYPES
        test_construction(T, F, 2^16)
    end
end


function test_prev_next(x::MultiFloat{T,N}, k::Int) where {T,N}
    y = x
    if k > 0
        for _ = 1:k
            y = prevfloat(y)
        end
        for _ = 1:k
            y = nextfloat(y)
        end
    elseif k < 0
        for _ = k:-1
            y = nextfloat(y)
        end
        for _ = k:-1
            y = prevfloat(y)
        end
    end
    @test (y == x) || (y == MultiFloats.canonize(x))
end


@testset "prevfloat/nextfloat round-trip" begin
    for T in _MF_TYPES
        for _ = 1:4096
            test_prev_next(_bit_rand(T), rand(-256:+256))
        end
        for k = -4:+4
            test_prev_next(+floatmin(T), k)
            test_prev_next(-floatmin(T), k)
            if k >= -1
                test_prev_next(+floatmax(T), k)
            end
            if k <= +1
                test_prev_next(-floatmax(T), k)
            end
        end
    end
end


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
    n::Int;
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
            for _ = 1:n
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
    n::Int;
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
            for _ = 1:n
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
        test_binary_operation(*, T, 1, 2^20)
    end
end


@testset "reciprocal" begin
    for T in _MF_TYPES
        test_unary_operation(inv, T, 2, 2^18;
            precise_condition=(_, e_hi, ex) -> (ex <= e_hi))
    end
end


@testset "division" begin
    for T in _MF_TYPES
        test_binary_operation(/, T, 2, 2^18;
            precise_condition=(e_lo, e_hi, ex, ey) ->
                (ex >= e_lo) & (ey <= e_hi) & (ex - ey >= e_lo),
            nan_condition=(_, y) -> issubnormal(y))
    end
end


@testset "reciprocal square root" begin
    for T in _MF_TYPES
        test_unary_operation(MultiFloats.rsqrt, T, 3, 2^18;
            precise_condition=(_, e_hi, ex) -> (ex <= e_hi),
            nan_condition=issubnormal, positive_inputs=true)
    end
end


@testset "square root" begin
    for T in _MF_TYPES
        test_unary_operation(sqrt, T, 3, 2^18;
            precise_condition=(e_lo, e_hi, ex) -> (e_lo <= ex <= e_hi),
            nan_condition=issubnormal, positive_inputs=true)
    end
end


function test_string_round_trip(x::MultiFloat{T,N}) where {T,N}
    y = MultiFloat{T,N}(string(x))
    @test (y == x) || (y == MultiFloats.canonize(x))
end


function test_string_round_trip(x::MultiFloatVec{M,T,N}) where {M,T,N}
    s = string(x)
    i = only(findall(==('['), s))
    j = only(findall(==(']'), s))
    y = MultiFloatVec{M,T,N}(MultiFloat{T,N}.(split(s[i+1:j-1], ", "))...)
    @test all((y == x) | (y == MultiFloats.canonize(x)))
end


@testset "scalar string conversion round-trip" begin
    for T in _MF_TYPES
        for _ = 1:1024
            test_string_round_trip(_bit_rand(T))
        end
        test_string_round_trip(+floatmin(T))
        test_string_round_trip(-floatmin(T))
        test_string_round_trip(+floatmax(T))
        test_string_round_trip(-floatmax(T))
    end
end


@testset "vector string conversion round-trip" begin
    for T in _MFV_TYPES
        for _ = 1:64
            test_string_round_trip(_bit_rand(T))
        end
    end
end


# @testset "elementary functions" begin
#     for T in _MF_TYPES
#         setprecision(BigFloat, precision(T) * 2) do
#             @testset "$func" for func in (exp2, log2, cbrt)
#                 for _ = 1:64
#                     x = rand(T)
#                     xbig = big(x)
#                     try
#                         ybig = func(xbig)
#                         @test abs(func(x) - ybig) < 10 * eps(T(ybig))
#                     catch e
#                         e isa DomainError || retrhow(e)
#                     end
#                 end
#             end
#         end
#     end
# end
