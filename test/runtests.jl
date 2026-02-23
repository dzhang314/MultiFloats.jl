using MultiFloats
using Test: @test, @testset


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


@testset "elementary functions" begin
    for T in _MF_TYPES
        setprecision(BigFloat, precision(T)*2) do
            @testset "$func" for func in (exp2, log2, sqrt, cbrt)
                for _ = 1:64
                    x = rand(T)
                    xbig = big(x)
                    try
                        ybig = func(xbig)
                        @test abs(func(x)-ybig) < 10 * eps(T(ybig))
                    catch e
                        e isa DomainError || retrhow(e)
                    end
                end
            end
            @testset "$func" for func in (+, *, ^, (x, y) -> x^Int(big(round(10y-5))))
                for _ = 1:64
                    x = rand(T)
                    y = rand(T)
                    xbig = big(x)
                    ybig = big(y)
                    try
                        ybig = func(xbig, ybig)
                        @test (func(x, y)-ybig)/eps(T(ybig)) < 10
                    catch e
                        e isa DomainError || retrhow(e)
                    end
                end
            end
        end
    end
end

