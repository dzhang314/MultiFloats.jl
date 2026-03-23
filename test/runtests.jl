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


include("conversion.jl")
include("arithmetic.jl")
include("linalg.jl")
