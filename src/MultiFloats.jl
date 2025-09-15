module MultiFloats

using SIMD: Vec, vifelse, vgather, vscatter
using SIMD.Intrinsics: extractelement


############################################################### TYPE DEFINITIONS


export MultiFloat, MultiFloatVec, PreciseMultiFloat, PreciseMultiFloatVec


struct MultiFloat{T,N} <: AbstractFloat
    _limbs::NTuple{N,T}
end


struct PreciseMultiFloat{T,N} <: AbstractFloat
    _limbs::NTuple{N,T}
end


struct MultiFloatVec{M,T,N}
    _limbs::NTuple{N,Vec{M,T}}
end


struct PreciseMultiFloatVec{M,T,N}
    _limbs::NTuple{N,Vec{M,T}}
end


# Private aliases for brevity.
const _MF = MultiFloat
const _MFV = MultiFloatVec
const _PMF = PreciseMultiFloat
const _PMFV = PreciseMultiFloatVec
const _GMF{T,N} = Union{_MF{T,N},_PMF{T,N}}
const _GMFV{M,T,N} = Union{_MFV{M,T,N},_PMFV{M,T,N}}


################################################################### TYPE ALIASES


export Float16x, Float32x, Float64x,
    PreciseFloat16x, PreciseFloat32x, PreciseFloat64x,
    Float64x1, Float64x2, Float64x3, Float64x4,
    PreciseFloat64x1, PreciseFloat64x2, PreciseFloat64x3, PreciseFloat64x4,
    Vec1Float64x1, Vec1Float64x2, Vec1Float64x3, Vec1Float64x4,
    Vec2Float64x1, Vec2Float64x2, Vec2Float64x3, Vec2Float64x4,
    Vec4Float64x1, Vec4Float64x2, Vec4Float64x3, Vec4Float64x4,
    Vec8Float64x1, Vec8Float64x2, Vec8Float64x3, Vec8Float64x4,
    Vec1PreciseFloat64x1, Vec1PreciseFloat64x2,
    Vec1PreciseFloat64x3, Vec1PreciseFloat64x4,
    Vec2PreciseFloat64x1, Vec2PreciseFloat64x2,
    Vec2PreciseFloat64x3, Vec2PreciseFloat64x4,
    Vec4PreciseFloat64x1, Vec4PreciseFloat64x2,
    Vec4PreciseFloat64x3, Vec4PreciseFloat64x4,
    Vec8PreciseFloat64x1, Vec8PreciseFloat64x2,
    Vec8PreciseFloat64x3, Vec8PreciseFloat64x4


const Float16x{N} = MultiFloat{Float16,N}
const Float32x{N} = MultiFloat{Float32,N}
const Float64x{N} = MultiFloat{Float64,N}


const PreciseFloat16x{N} = PreciseMultiFloat{Float16,N}
const PreciseFloat32x{N} = PreciseMultiFloat{Float32,N}
const PreciseFloat64x{N} = PreciseMultiFloat{Float64,N}


const Float64x1 = MultiFloat{Float64,1}
const Float64x2 = MultiFloat{Float64,2}
const Float64x3 = MultiFloat{Float64,3}
const Float64x4 = MultiFloat{Float64,4}


const PreciseFloat64x1 = PreciseMultiFloat{Float64,1}
const PreciseFloat64x2 = PreciseMultiFloat{Float64,2}
const PreciseFloat64x3 = PreciseMultiFloat{Float64,3}
const PreciseFloat64x4 = PreciseMultiFloat{Float64,4}


const Vec1Float64x1 = MultiFloatVec{1,Float64,1}
const Vec1Float64x2 = MultiFloatVec{1,Float64,2}
const Vec1Float64x3 = MultiFloatVec{1,Float64,3}
const Vec1Float64x4 = MultiFloatVec{1,Float64,4}
const Vec2Float64x1 = MultiFloatVec{2,Float64,1}
const Vec2Float64x2 = MultiFloatVec{2,Float64,2}
const Vec2Float64x3 = MultiFloatVec{2,Float64,3}
const Vec2Float64x4 = MultiFloatVec{2,Float64,4}
const Vec4Float64x1 = MultiFloatVec{4,Float64,1}
const Vec4Float64x2 = MultiFloatVec{4,Float64,2}
const Vec4Float64x3 = MultiFloatVec{4,Float64,3}
const Vec4Float64x4 = MultiFloatVec{4,Float64,4}
const Vec8Float64x1 = MultiFloatVec{8,Float64,1}
const Vec8Float64x2 = MultiFloatVec{8,Float64,2}
const Vec8Float64x3 = MultiFloatVec{8,Float64,3}
const Vec8Float64x4 = MultiFloatVec{8,Float64,4}


const Vec1PreciseFloat64x1 = PreciseMultiFloatVec{1,Float64,1}
const Vec1PreciseFloat64x2 = PreciseMultiFloatVec{1,Float64,2}
const Vec1PreciseFloat64x3 = PreciseMultiFloatVec{1,Float64,3}
const Vec1PreciseFloat64x4 = PreciseMultiFloatVec{1,Float64,4}
const Vec2PreciseFloat64x1 = PreciseMultiFloatVec{2,Float64,1}
const Vec2PreciseFloat64x2 = PreciseMultiFloatVec{2,Float64,2}
const Vec2PreciseFloat64x3 = PreciseMultiFloatVec{2,Float64,3}
const Vec2PreciseFloat64x4 = PreciseMultiFloatVec{2,Float64,4}
const Vec4PreciseFloat64x1 = PreciseMultiFloatVec{4,Float64,1}
const Vec4PreciseFloat64x2 = PreciseMultiFloatVec{4,Float64,2}
const Vec4PreciseFloat64x3 = PreciseMultiFloatVec{4,Float64,3}
const Vec4PreciseFloat64x4 = PreciseMultiFloatVec{4,Float64,4}
const Vec8PreciseFloat64x1 = PreciseMultiFloatVec{8,Float64,1}
const Vec8PreciseFloat64x2 = PreciseMultiFloatVec{8,Float64,2}
const Vec8PreciseFloat64x3 = PreciseMultiFloatVec{8,Float64,3}
const Vec8PreciseFloat64x4 = PreciseMultiFloatVec{8,Float64,4}


###################################################################### CONSTANTS


@inline Base.zero(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(ntuple(
    _ -> zero(T), Val{N}()))
@inline Base.zero(::Type{_PMF{T,N}}) where {T,N} = _PMF{T,N}(ntuple(
    _ -> zero(T), Val{N}()))
@inline Base.zero(::Type{_MFV{M,T,N}}) where {M,T,N} = _MFV{M,T,N}(ntuple(
    _ -> zero(Vec{M,T}), Val{N}()))
@inline Base.zero(::Type{_PMFV{M,T,N}}) where {M,T,N} = _PMFV{M,T,N}(ntuple(
    _ -> zero(Vec{M,T}), Val{N}()))


@inline Base.zero(::_MF{T,N}) where {T,N} = zero(_MF{T,N})
@inline Base.zero(::_PMF{T,N}) where {T,N} = zero(_PMF{T,N})
@inline Base.zero(::_MFV{M,T,N}) where {M,T,N} = zero(_MFV{M,T,N})
@inline Base.zero(::_PMFV{M,T,N}) where {M,T,N} = zero(_PMFV{M,T,N})


@inline Base.one(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(ntuple(
    i -> ifelse(isone(i), one(T), zero(T)), Val{N}()))
@inline Base.one(::Type{_PMF{T,N}}) where {T,N} = _PMF{T,N}(ntuple(
    i -> ifelse(isone(i), one(T), zero(T)), Val{N}()))
@inline Base.one(::Type{_MFV{M,T,N}}) where {M,T,N} = _MFV{M,T,N}(ntuple(
    i -> ifelse(isone(i), one(Vec{M,T}), zero(Vec{M,T})), Val{N}()))
@inline Base.one(::Type{_PMFV{M,T,N}}) where {M,T,N} = _PMFV{M,T,N}(ntuple(
    i -> ifelse(isone(i), one(Vec{M,T}), zero(Vec{M,T})), Val{N}()))


@inline Base.one(::_MF{T,N}) where {T,N} = one(_MF{T,N})
@inline Base.one(::_PMF{T,N}) where {T,N} = one(_PMF{T,N})
@inline Base.one(::_MFV{M,T,N}) where {M,T,N} = one(_MFV{M,T,N})
@inline Base.one(::_PMFV{M,T,N}) where {M,T,N} = one(_PMFV{M,T,N})


################################################################### CONSTRUCTORS


# TODO: Add constructors for PreciseMultiFloat types.
# TODO: Add converting constructors between MultiFloat and PreciseMultiFloat.


# Construct from a single limb: pad remaining limbs with zeroes.
@inline _MF{T,N}(x::T) where {T,N} = _MF{T,N}(
    ntuple(i -> ifelse(isone(i), x, zero(T)), Val{N}()))
@inline _PMF{T,N}(x::T) where {T,N} = _PMF{T,N}(
    ntuple(i -> ifelse(isone(i), x, zero(T)), Val{N}()))
@inline _MFV{M,T,N}(x::Vec{M,T}) where {M,T,N} = _MFV{M,T,N}(
    ntuple(i -> ifelse(isone(i), x, zero(Vec{M,T})), Val{N}()))
@inline _PMFV{M,T,N}(x::Vec{M,T}) where {M,T,N} = _PMFV{M,T,N}(
    ntuple(i -> ifelse(isone(i), x, zero(Vec{M,T})), Val{N}()))
@inline _MFV{M,T,N}(x::NTuple{M,T}) where {M,T,N} = _MFV{M,T,N}(Vec{M,T}(x))
@inline _MFV{M,T,N}(x::Vararg{T,M}) where {M,T,N} = _MFV{M,T,N}(Vec{M,T}(x))
@inline _PMFV{M,T,N}(x::NTuple{M,T}) where {M,T,N} = _PMFV{M,T,N}(Vec{M,T}(x))
@inline _PMFV{M,T,N}(x::Vararg{T,M}) where {M,T,N} = _PMFV{M,T,N}(Vec{M,T}(x))


# Construct from multiple limbs: truncate or pad with zeroes.
@inline _MF{T,N1}(x::_GMF{T,N2}) where {T,N1,N2} = _MF{T,N1}(
    tuple(ntuple(i -> x._limbs[i], Val{min(N1, N2)}())...,
        ntuple(_ -> zero(T), Val{max(N1 - N2, 0)}())...))
@inline _PMF{T,N1}(x::_GMF{T,N2}) where {T,N1,N2} = _PMF{T,N1}(
    tuple(ntuple(i -> x._limbs[i], Val{min(N1, N2)}())...,
        ntuple(_ -> zero(T), Val{max(N1 - N2, 0)}())...))
@inline _MFV{M,T,N1}(x::_GMF{T,N2}) where {M,T,N1,N2} = _MFV{M,T,N1}(
    tuple(ntuple(i -> Vec{M,T}(x._limbs[i]), Val{min(N1, N2)}())...,
        ntuple(_ -> zero(Vec{M,T}), Val{max(N1 - N2, 0)}())...))
@inline _PMFV{M,T,N1}(x::_GMF{T,N2}) where {M,T,N1,N2} = _PMFV{M,T,N1}(
    tuple(ntuple(i -> Vec{M,T}(x._limbs[i]), Val{min(N1, N2)}())...,
        ntuple(_ -> zero(Vec{M,T}), Val{max(N1 - N2, 0)}())...))
@inline _MFV{M,T,N1}(x::_GMFV{M,T,N2}) where {M,T,N1,N2} = _MFV{M,T,N1}(
    tuple(ntuple(i -> x._limbs[i], Val{min(N1, N2)}())...,
        ntuple(_ -> zero(Vec{M,T}), Val{max(N1 - N2, 0)}())...))
@inline _PMFV{M,T,N1}(x::_GMFV{M,T,N2}) where {M,T,N1,N2} = _PMFV{M,T,N1}(
    tuple(ntuple(i -> x._limbs[i], Val{min(N1, N2)}())...,
        ntuple(_ -> zero(Vec{M,T}), Val{max(N1 - N2, 0)}())...))


# Construct vector from scalar: broadcast.
@inline _MFV{M,T,N}(x::T) where {M,T,N} = _MFV{M,T,N}(Vec{M,T}(x))
@inline _PMFV{M,T,N}(x::T) where {M,T,N} = _PMFV{M,T,N}(Vec{M,T}(x))
@inline _MFV{M,T,N}(x::_MF{T,N}) where {M,T,N} = _MFV{M,T,N}(
    ntuple(i -> Vec{M,T}(x._limbs[i]), Val{N}()))
@inline _MFV{1,T,N}(x::_MF{T,N}) where {T,N} = _MFV{1,T,N}(
    ntuple(i -> Vec{1,T}(x._limbs[i]), Val{N}()))
@inline _PMFV{M,T,N}(x::_PMF{T,N}) where {M,T,N} = _PMFV{M,T,N}(
    ntuple(i -> Vec{M,T}(x._limbs[i]), Val{N}()))
@inline _PMFV{1,T,N}(x::_PMF{T,N}) where {T,N} = _PMFV{1,T,N}(
    ntuple(i -> Vec{1,T}(x._limbs[i]), Val{N}()))


# Construct vector from tuple of scalars: transpose.
@inline _MFV{M,T,N}(xs::NTuple{M,_GMF{T,N}}) where {M,T,N} = _MFV{M,T,N}(
    ntuple(j -> Vec{M,T}(ntuple(i -> xs[i]._limbs[j], Val{M}())), Val{N}()))
@inline _MFV{M,T,N}(xs::Vararg{_GMF{T,N},M}) where {M,T,N} = _MFV{M,T,N}(
    ntuple(j -> Vec{M,T}(ntuple(i -> xs[i]._limbs[j], Val{M}())), Val{N}()))
@inline _PMFV{M,T,N}(xs::NTuple{M,_GMF{T,N}}) where {M,T,N} = _PMFV{M,T,N}(
    ntuple(j -> Vec{M,T}(ntuple(i -> xs[i]._limbs[j], Val{M}())), Val{N}()))
@inline _PMFV{M,T,N}(xs::Vararg{_GMF{T,N},M}) where {M,T,N} = _PMFV{M,T,N}(
    ntuple(j -> Vec{M,T}(ntuple(i -> xs[i]._limbs[j], Val{M}())), Val{N}()))


################################################################ VECTOR INDEXING


# export mfvgather, mfvscatter


@inline Base.length(::_GMFV{M,T,N}) where {M,T,N} = M


@inline Base.getindex(x::_MFV{M,T,N}, i::I) where {M,T,N,I} = _MF{T,N}(
    ntuple(j -> extractelement(x._limbs[j].data, i - one(I)), Val{N}()))
@inline Base.getindex(x::_PMFV{M,T,N}, i::I) where {M,T,N,I} = _PMF{T,N}(
    ntuple(j -> extractelement(x._limbs[j].data, i - one(I)), Val{N}()))


# TODO: Add support for PreciseMultiFloatVec types.


# @inline function mfvgather(
#     pointer::Ptr{_MF{T,N}}, index::Vec{M,I}
# ) where {M,T,N,I<:Integer}
#     base = reinterpret(Ptr{T}, pointer) + N * sizeof(T) * index
#     return _MFV{M,T,N}(ntuple(
#         i -> vgather(base + (i - 1) * sizeof(T)), Val{N}()))
# end


# @inline function mfvscatter(
#     x::_MFV{M,T,N}, pointer::Ptr{_MF{T,N}}, index::Vec{M,I}
# ) where {M,T,N,I<:Integer}
#     base = reinterpret(Ptr{T}, pointer) + N * sizeof(T) * index
#     for i = 1:N
#         vscatter(x._limbs[i], base + (i - 1) * sizeof(T), nothing)
#     end
#     return nothing
# end


# @inline mfvgather(array::Array{_MF{T,N},D}, index::Vec{M,I}
# ) where {M,T,N,D,I<:Integer} = mfvgather(pointer(array), index - one(I))


# @inline mfvscatter(x::_MFV{M,T,N}, array::Array{_MF{T,N},D}, index::Vec{M,I}
# ) where {M,T,N,D,I<:Integer} = mfvscatter(x, pointer(array), index - one(I))


################################################ CONVERSION FROM PRIMITIVE TYPES


# TODO: Add converting constructors for PreciseMultiFloat types.


# Bool, Int8, UInt8, Int16, UInt16, Float16, Int32, UInt32, and Float32
# can be directly converted to Float64 without losing precision.


@inline Float64x{N}(
    x::Union{Bool,Int8,Int16,Int32,UInt8,UInt16,UInt32,Float16,Float32},
) where {N} = Float64x{N}(Float64(x))
@inline PreciseFloat64x{N}(
    x::Union{Bool,Int8,Int16,Int32,UInt8,UInt16,UInt32,Float16,Float32},
) where {N} = PreciseFloat64x{N}(Float64(x))


# Int64, UInt64, Int128, and UInt128 cannot be directly converted to Float64
# without losing precision, so they must be split into multiple components.


# @inline Float64x1(x::Int64) = Float64x1(Float64(x))
# @inline Float64x1(x::UInt64) = Float64x1(Float64(x))
# @inline Float64x1(x::Int128) = Float64x1(Float64(x))
# @inline Float64x1(x::UInt128) = Float64x1(Float64(x))


# @inline function Float64x{N}(x::Int64) where {N}
#     x0 = Float64(x)
#     x1 = Float64(x - Int64(x0))
#     return Float64x{N}((x0, x1, ntuple(_ -> 0.0, Val{N - 2}())...))
# end


# @inline function Float64x{N}(x::UInt64) where {N}
#     x0 = Float64(x)
#     x1 = Float64(reinterpret(Int64, x - UInt64(x0)))
#     return Float64x{N}((x0, x1, ntuple(_ -> 0.0, Val{N - 2}())...))
# end


# @inline function Float64x2(x::Int128)
#     x0 = Float64(x)
#     x1 = Float64(x - Int128(x0))
#     return Float64x2((x0, x1))
# end


# @inline function Float64x2(x::UInt128)
#     x0 = Float64(x)
#     x1 = Float64(reinterpret(Int128, x - UInt128(x0)))
#     return Float64x2((x0, x1))
# end


# @inline function Float64x{N}(x::Int128) where {N}
#     x0 = Float64(x)
#     r1 = x - Int128(x0)
#     x1 = Float64(r1)
#     x2 = Float64(r1 - Int128(x1))
#     return Float64x{N}((x0, x1, x2, ntuple(_ -> 0.0, Val{N - 3}())...))
# end


# @inline function Float64x{N}(x::UInt128) where {N}
#     x0 = Float64(x)
#     r1 = reinterpret(Int128, x - UInt128(x0))
#     x1 = Float64(r1)
#     x2 = Float64(r1 - Int128(x1))
#     return Float64x{N}((x0, x1, x2, ntuple(_ -> 0.0, Val{N - 3}())...))
# end


# @inline _MFV{M,T,N}(x::Bool) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::Int8) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::Int16) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::Int32) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::Int64) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::Int128) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::UInt8) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::UInt16) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::UInt32) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::UInt64) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::UInt128) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::Float16) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::Float32) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::Float64) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))


###################################################### CONVERSION FROM BIG TYPES


# @inline _extract_limbs(::Type{T}, x::BigFloat, ::Val{0}) where {T} = ()
# @inline _extract_limbs(::Type{T}, x::BigFloat, ::Val{1}) where {T} = (T(x),)
# @inline function _extract_limbs(::Type{T}, x::BigFloat, ::Val{N}) where {T,N}
#     head = T(x)
#     tail = _extract_limbs(T, x - head, Val{N - 1}())
#     return (head, tail...)
# end


# @inline function _MF{T,N}(x::BigFloat) where {T,N}
#     if !isfinite(x)
#         value = T(x)
#         return _MF{T,N}(ntuple(_ -> value, Val{N}()))
#     elseif x > +floatmax(T)
#         value = T(+Inf)
#         return _MF{T,N}(ntuple(_ -> value, Val{N}()))
#     elseif x < -floatmax(T)
#         value = T(-Inf)
#         return _MF{T,N}(ntuple(_ -> value, Val{N}()))
#     else
#         return setprecision(BigFloat, precision(x)) do
#             setrounding(BigFloat, RoundNearest) do
#                 _MF{T,N}(_extract_limbs(T, x, Val{N}()))
#             end
#         end
#     end
# end


# @inline _extract_limbs(::Type{T}, x::BigInt, ::Val{0}) where {T} = ()
# @inline _extract_limbs(::Type{T}, x::BigInt, ::Val{1}) where {T} = (T(x),)
# @inline function _extract_limbs(::Type{T}, x::BigInt, ::Val{N}) where {T,N}
#     head = T(x)
#     tail = _extract_limbs(T, x - BigInt(head), Val{N - 1}())
#     return (head, tail...)
# end


# @inline function _MF{T,N}(x::BigInt) where {T,N}
#     if x > +floatmax(T)
#         value = T(+Inf)
#         return _MF{T,N}(ntuple(_ -> value, Val{N}()))
#     elseif x < -floatmax(T)
#         value = T(-Inf)
#         return _MF{T,N}(ntuple(_ -> value, Val{N}()))
#     else
#         return _MF{T,N}(_extract_limbs(T, x, Val{N}()))
#     end
# end


# @inline function _MF{T,N}(x::Rational{U}) where {T,N,U}
#     return setrounding(BigFloat, RoundNearest) do
#         _MF{T,N}(BigFloat(x; precision=_full_precision(T)))
#     end
# end


# @inline _MFV{M,T,N}(x::BigFloat) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::BigInt) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
# @inline _MFV{M,T,N}(x::Rational{U}) where {M,T,N,U} = _MFV{M,T,N}(_MF{T,N}(x))


########################################## CONVERSION FROM STRING AND IRRATIONAL


# @inline _full_precision(::Type{T}) where {T} =
#     exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)


# function _MF{T,N}(s::AbstractString) where {T,N}
#     return setrounding(BigFloat, RoundNearest) do
#         _MF{T,N}(BigFloat(s; precision=_full_precision(T)))
#     end
# end


# _MFV{M,T,N}(s::AbstractString) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(s))


# function _MF{T,N}(x::Irrational) where {T,N}
#     return setrounding(BigFloat, RoundNearest) do
#         _MF{T,N}(BigFloat(x; precision=_full_precision(T)))
#     end
# end


# _MFV{M,T,N}(x::Irrational) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))


######################################################################## SCALING


# Note: MultiFloats.scale is not exported because it is only useful for
# MultiFloats. Users are expected to call it as MultiFloats.scale(a, x).


@inline scale(a, x) = a * x
@inline scale(a::T, x::_MF{T,N}) where {T,N} = _MF{T,N}(
    ntuple(i -> a * x._limbs[i], Val{N}()))
@inline scale(a::T, x::_PMF{T,N}) where {T,N} = _PMF{T,N}(
    ntuple(i -> a * x._limbs[i], Val{N}()))
@inline scale(a::T, x::_MFV{M,T,N}) where {M,T,N} = _MFV{M,T,N}(
    ntuple(i -> a * x._limbs[i], Val{N}()))
@inline scale(a::T, x::_PMFV{M,T,N}) where {M,T,N} = _PMFV{M,T,N}(
    ntuple(i -> a * x._limbs[i], Val{N}()))


########################################################## ERROR-FREE ARITHMETIC


@inline function fast_two_sum(a::T, b::T) where {T}
    sum = a + b
    b_prime = sum - a
    b_err = b - b_prime
    return (sum, b_err)
end


@inline function two_sum(a::T, b::T) where {T}
    sum = a + b
    a_prime = sum - b
    b_prime = sum - a_prime
    a_err = a - a_prime
    b_err = b - b_prime
    err = a_err + b_err
    return (sum, err)
end


@inline function two_prod(a::T, b::T) where {T}
    prod = a * b
    err = fma(a, b, -prod)
    return (prod, err)
end


################################################################ RENORMALIZATION


# Note: MultiFloats.renormalize is not exported because it is a
# MultiFloat-specific operation. Users are expected to call it as
# MultiFloats.renormalize(x).


# # This function is needed to work around the following SIMD bug:
# # https://github.com/eschnett/SIMD.jl/issues/115
# @inline _ntuple_equal(x::NTuple{N,T}, y::NTuple{N,T}
# ) where {N,T} = all(x .== y)
# @inline _ntuple_equal(x::NTuple{N,Vec{M,T}}, y::NTuple{N,Vec{M,T}}
# ) where {N,M,T} = all(all.(x .== y))


# @inline function renormalize(xs::NTuple{N,T}) where {T,N}
#     total = sum(xs)
#     if !isfinite(total)
#         return ntuple(_ -> total, Val{N}())
#     end
#     while true
#         xs_new = _two_pass_renorm(Val{N}(), xs...)
#         if _ntuple_equal(xs, xs_new)
#             return xs
#         else
#             xs = xs_new
#         end
#     end
# end


# @inline _mask_each(
#     mask::Vec{M,Bool}, x::NTuple{N,Vec{M,T}}, y::Vec{M,T}
# ) where {M,T,N} = ntuple(i -> vifelse(mask, x[i], y), Val{N}())
# @inline _mask_each(
#     mask::Vec{M,Bool}, x::NTuple{N,Vec{M,T}}, y::NTuple{N,Vec{M,T}}
# ) where {M,T,N} = ntuple(i -> vifelse(mask, x[i], y[i]), Val{N}())


# @inline function renormalize(xs::NTuple{N,Vec{M,T}}) where {M,T,N}
#     total = sum(xs)
#     mask = isfinite(total)
#     xs = _mask_each(mask, xs, zero(Vec{M,T}))
#     while true
#         xs_new = _two_pass_renorm(Val{N}(), xs...)
#         if _ntuple_equal(xs, xs_new)
#             return _mask_each(mask, xs, total)
#         else
#             xs = xs_new
#         end
#     end
# end


# @inline renormalize(x::_MF{T,N}) where {T,N} =
#     _MF{T,N}(renormalize(x._limbs))
# @inline renormalize(x::_MFV{M,T,N}) where {M,T,N} =
#     _MFV{M,T,N}(renormalize(x._limbs))
# @inline renormalize(x::T) where {T<:Real} = x


################################################### FLOATING-POINT INTROSPECTION


# @inline _and(::Tuple{}) = true
# @inline _and(x::NTuple{1,Bool}) = x[1]
# @inline _and(x::NTuple{N,Bool}) where {N} = (&)(x...)


# @inline _vand(::Val{M}, ::Tuple{}) where {M} = one(Vec{M,Bool})
# @inline _vand(::Val{M}, x::NTuple{1,Vec{M,Bool}}) where {M} = x[1]
# @inline _vand(::Val{M}, x::NTuple{N,Vec{M,Bool}}) where {M,N} = (&)(x...)


# @inline _iszero(x::_MF{T,N}) where {T,N} = _and(
#     ntuple(i -> iszero(x._limbs[i]), Val{N}()))
# @inline _iszero(x::_MFV{M,T,N}) where {M,T,N} = _vand(
#     Val{M}(), ntuple(i -> iszero(x._limbs[i]), Val{N}()))
# @inline _isone(x::_MF{T,N}) where {T,N} = isone(x._limbs[1]) & _and(
#     ntuple(i -> iszero(x._limbs[i+1]), Val{N - 1}()))
# @inline _isone(x::_MFV{M,T,N}) where {M,T,N} = isone(x._limbs[1]) & _vand(
#     Val{M}(), ntuple(i -> iszero(x._limbs[i+1]), Val{N - 1}()))


# @inline Base.iszero(x::_MF{T,N}) where {T,N} = _iszero(renormalize(x))
# @inline Base.iszero(x::_MFV{M,T,N}) where {M,T,N} = _iszero(renormalize(x))
# @inline Base.isone(x::_MF{T,N}) where {T,N} = _isone(renormalize(x))
# @inline Base.isone(x::_MFV{M,T,N}) where {M,T,N} = _isone(renormalize(x))
# @inline Base.iszero(x::_MF{T,2}) where {T} = _iszero(x)
# @inline Base.iszero(x::_MFV{M,T,2}) where {M,T} = _iszero(x)
# @inline Base.isone(x::_MF{T,2}) where {T} = _isone(x)
# @inline Base.isone(x::_MFV{M,T,2}) where {M,T} = _isone(x)


# @inline _head(x::_MF{T,N}) where {T,N} = renormalize(x)._limbs[1]
# @inline _head(x::_MFV{M,T,N}) where {M,T,N} = renormalize(x)._limbs[1]
# @inline _head(x::_MF{T,2}) where {T} = x._limbs[1]
# @inline _head(x::_MFV{M,T,2}) where {M,T} = x._limbs[1]


# @inline Base.issubnormal(x::_MF{T,N}) where {T,N} = issubnormal(_head(x))
# @inline Base.issubnormal(x::_MFV{M,T,N}) where {M,T,N} = issubnormal(_head(x))
# @inline Base.isfinite(x::_MF{T,N}) where {T,N} = isfinite(_head(x))
# @inline Base.isfinite(x::_MFV{M,T,N}) where {M,T,N} = isfinite(_head(x))
# @inline Base.isinf(x::_MF{T,N}) where {T,N} = isinf(_head(x))
# @inline Base.isinf(x::_MFV{M,T,N}) where {M,T,N} = isinf(_head(x))
# @inline Base.isnan(x::_MF{T,N}) where {T,N} = isnan(_head(x))
# @inline Base.isnan(x::_MFV{M,T,N}) where {M,T,N} = isnan(_head(x))
# @inline Base.signbit(x::_MF{T,N}) where {T,N} = signbit(_head(x))
# @inline Base.signbit(x::_MFV{M,T,N}) where {M,T,N} = signbit(_head(x))


# # Note: SIMD.jl does not define Base.exponent or Base.isinteger for vectors.
# @inline Base.exponent(x::_MF{T,N}) where {T,N} = exponent(_head(x))
# @inline Base.isinteger(x::_MF{T,N}) where {T,N} =
#     all(isinteger.(renormalize(x)._limbs))


# # Note: SIMD.jl does not define Base.ldexp for vectors.
# @inline function Base.ldexp(x::_MF{T,N}, n::I) where {T,N,I}
#     x = renormalize(x)
#     return _MF{T,N}(ntuple(i -> ldexp(x._limbs[i], n), Val{N}()))
# end

# # needed for hashing
# Base.decompose(x::MultiFloat) = Base.decompose(BigFloat(x))

# # Note: SIMD.jl does not define Base.prevfloat or Base.nextfloat for vectors.
# _prevfloat(x::_MF{T,N}) where {T,N} = renormalize(_MF{T,N}((ntuple(
#         i -> x._limbs[i], Val{N - 1}())..., prevfloat(x._limbs[N]))))
# _nextfloat(x::_MF{T,N}) where {T,N} = renormalize(_MF{T,N}((ntuple(
#         i -> x._limbs[i], Val{N - 1}())..., nextfloat(x._limbs[N]))))
# @inline Base.prevfloat(x::_MF{T,N}) where {T,N} = _prevfloat(renormalize(x))
# @inline Base.nextfloat(x::_MF{T,N}) where {T,N} = _nextfloat(renormalize(x))


# # Note: SIMD.jl does not define Base.precision for vectors.
# if isdefined(Base, :_precision)
#     @inline Base._precision(::Type{_MF{T,N}}) where {T,N} =
#         N * precision(T) + (N - 1) # implicit bits of precision between limbs
# else
#     @inline Base.precision(::Type{_MF{T,N}}) where {T,N} =
#         N * precision(T) + (N - 1) # implicit bits of precision between limbs
# end


# # Note: SIMD.jl does not define Base.eps for vectors.
# @inline Base.eps(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(eps(T)^N)


# # Note: SIMD.jl does not define Base.floatmin or Base.floatmax for vectors.
# @inline Base.floatmin(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(floatmin(T))
# @inline Base.floatmax(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(floatmax(T))


# # Note: SIMD.jl does not define Base.typemin or Base.typemax for vectors.
# @inline Base.typemin(::Type{_MF{T,N}}) where {T,N} =
#     _MF{T,N}(ntuple(_ -> typemin(T), Val{N}()))
# @inline Base.typemax(::Type{_MF{T,N}}) where {T,N} =
#     _MF{T,N}(ntuple(_ -> typemax(T), Val{N}()))


################################################## CONVERSION TO PRIMITIVE TYPES


# @inline Base.Float16(x::Float16x{N}) where {N} = _head(x)
# @inline Base.Float32(x::Float32x{N}) where {N} = _head(x)
# @inline Base.Float64(x::Float64x{N}) where {N} = _head(x)


# @inline Base.Float16(x::Float32x{N}) where {N} = Float16(_head(x))
# @inline Base.Float16(x::Float64x{N}) where {N} = Float16(_head(x))
# @inline Base.Float32(x::Float64x{N}) where {N} = Float32(_head(x))


# TODO: Conversion from Float32x{N} to Float64.
# TODO: Conversion from Float16x{N} to Float32.
# TODO: Conversion from Float16x{N} to Float64.


######################################################## CONVERSION TO BIG TYPES


# Base.BigFloat(x::_MF{T,N}; precision::Integer=precision(BigFloat)) where {T,N} =
#     setprecision(BigFloat, precision) do
#         setrounding(BigFloat, RoundNearest) do
#             +(BigFloat.(renormalize(x)._limbs)...)
#         end
#     end


# Base.Rational{BigInt}(x::_MF{T,N}) where {T,N} =
#     +(Rational{BigInt}.(renormalize(x)._limbs)...)


####################################################################### PRINTING


# function _call_big(f::F, x::_MF{T,N}) where {F,T,N}
#     x = renormalize(x)
#     total = +(x._limbs...)
#     if !isfinite(total)
#         return setprecision(BigFloat, precision(T)) do
#             setrounding(BigFloat, RoundNearest) do
#                 f(BigFloat(total))
#             end
#         end
#     end
#     i = N
#     while (i > 0) && iszero(x._limbs[i])
#         i -= 1
#     end
#     if iszero(i)
#         return setprecision(BigFloat, precision(T)) do
#             setrounding(BigFloat, RoundNearest) do
#                 f(zero(BigFloat))
#             end
#         end
#     else
#         p = precision(T) + exponent(x._limbs[1]) - exponent(x._limbs[i])
#         return setprecision(BigFloat, p) do
#             setrounding(BigFloat, RoundNearest) do
#                 f(BigFloat(x))
#             end
#         end
#     end
# end


# function _call_big(f::F, x::_MF{T,N}, p::Int) where {F,T,N}
#     x = renormalize(x)
#     total = +(x._limbs...)
#     if !isfinite(total)
#         return setprecision(BigFloat, p) do
#             setrounding(BigFloat, RoundNearest) do
#                 f(BigFloat(total))
#             end
#         end
#     end
#     i = N
#     while (i > 0) && iszero(x._limbs[i])
#         i -= 1
#     end
#     if iszero(i)
#         return setprecision(BigFloat, p) do
#             setrounding(BigFloat, RoundNearest) do
#                 f(zero(BigFloat))
#             end
#         end
#     else
#         return setprecision(BigFloat, p) do
#             setrounding(BigFloat, RoundNearest) do
#                 f(BigFloat(x))
#             end
#         end
#     end
# end


# function Base.print(io::IO, x::_MF{T,N}) where {T,N}
#     _call_big(y -> print(io, y), x)
#     return nothing
# end


# function Base.print(io::IO, x::_MFV{M,T,N}) where {M,T,N}
#     write(io, '<')
#     print(io, M)
#     write(io, " x ")
#     print(io, T)
#     write(io, " x ")
#     print(io, N)
#     write(io, ">[")
#     for i = 1:M
#         if i > 1
#             write(io, ", ")
#         end
#         _call_big(y -> print(io, y), x[i])
#     end
#     write(io, ']')
#     return nothing
# end


# function Base.show(io::IO, ::MIME"text/plain", x::_MF{T,N}) where {T,N}
#     _call_big(y -> show(io, y), x)
#     return nothing
# end


# function Base.show(io::IO, x::_MFV{M,T,N}) where {M,T,N}
#     show(io, _MFV{M,T,N})
#     write(io, "((")
#     for i = 1:N
#         if i > 1
#             write(io, ", ")
#         end
#         show(io, Vec{M,T})
#         write(io, "((")
#         for j = 1:M
#             if j > 1
#                 write(io, ", ")
#             end
#             show(io, x._limbs[i][j])
#         end
#         write(io, "))")
#     end
#     write(io, "))")
#     return nothing
# end


# function Base.show(io::IO, ::MIME"text/plain", x::_MFV{M,T,N}) where {M,T,N}
#     write(io, '<')
#     show(io, M)
#     write(io, " x ")
#     show(io, T)
#     write(io, " x ")
#     show(io, N)
#     write(io, ">[")
#     for i = 1:M
#         if i > 1
#             write(io, ", ")
#         end
#         _call_big(y -> show(io, y), x[i])
#     end
#     write(io, ']')
#     return nothing
# end


################################################# STANDARD LIBRARY COMPATIBILITY


# (::Type{I})(x::_MF{T,N}) where {I<:Integer,T,N} =
#     setrounding(BigFloat, RoundNearest) do
#         I(BigFloat(x; precision=_full_precision(T)))
#     end


# MultiFloat{T,N}(z::Complex) where {T,N} =
#     isreal(z) ? MultiFloat{T,N}(real(z)) :
#     throw(InexactError(nameof(MultiFloat{T,N}), MultiFloat{T,N}, z))


# import LinearAlgebra: floatmin2
# @inline floatmin2(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(ldexp(one(T),
#     div(exponent(floatmin(T)) - N * exponent(eps(T)), 2)))


# import Printf: tofloat
# @inline tofloat(x::_MF{T,N}) where {T,N} = _call_big(BigFloat, x)


##################################################################### COMPARISON


# # TODO: MultiFloat-to-Float comparison.
# # TODO: Implement Base.cmp.


# _eq_expr(n::Int) = (n == 1) ? :(x._limbs[1] == y._limbs[1]) : :(
#     $(_eq_expr(n - 1)) & (x._limbs[$n] == y._limbs[$n]))
# _ne_expr(n::Int) = (n == 1) ? :(x._limbs[1] != y._limbs[1]) : :(
#     $(_ne_expr(n - 1)) | (x._limbs[$n] != y._limbs[$n]))
# _lt_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$i] < y._limbs[$i]) : :(
#     (x._limbs[$i] < y._limbs[$i]) |
#     ((x._limbs[$i] == y._limbs[$i]) & $(_lt_expr(i + 1, n))))
# _gt_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$i] > y._limbs[$i]) : :(
#     (x._limbs[$i] > y._limbs[$i]) |
#     ((x._limbs[$i] == y._limbs[$i]) & $(_gt_expr(i + 1, n))))
# _le_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$i] <= y._limbs[$i]) : :(
#     (x._limbs[$i] < y._limbs[$i]) |
#     ((x._limbs[$i] == y._limbs[$i]) & $(_le_expr(i + 1, n))))
# _ge_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$i] >= y._limbs[$i]) : :(
#     (x._limbs[$i] > y._limbs[$i]) |
#     ((x._limbs[$i] == y._limbs[$i]) & $(_ge_expr(i + 1, n))))


# @generated _eq(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _eq_expr(N)
# @generated _eq(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _eq_expr(N)
# @generated _ne(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _ne_expr(N)
# @generated _ne(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _ne_expr(N)
# @generated _lt(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _lt_expr(1, N)
# @generated _lt(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _lt_expr(1, N)
# @generated _gt(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _gt_expr(1, N)
# @generated _gt(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _gt_expr(1, N)
# @generated _le(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _le_expr(1, N)
# @generated _le(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _le_expr(1, N)
# @generated _ge(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _ge_expr(1, N)
# @generated _ge(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _ge_expr(1, N)


# @inline Base.:(==)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
#     _eq(renormalize(x), renormalize(y))
# @inline Base.:(==)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
#     _eq(renormalize(x), renormalize(y))
# @inline Base.:(!=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
#     _ne(renormalize(x), renormalize(y))
# @inline Base.:(!=)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
#     _ne(renormalize(x), renormalize(y))
# @inline Base.:(<)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
#     _lt(renormalize(x), renormalize(y))
# @inline Base.:(<)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
#     _lt(renormalize(x), renormalize(y))
# @inline Base.:(>)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
#     _gt(renormalize(x), renormalize(y))
# @inline Base.:(>)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
#     _gt(renormalize(x), renormalize(y))
# @inline Base.:(<=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
#     _le(renormalize(x), renormalize(y))
# @inline Base.:(<=)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
#     _le(renormalize(x), renormalize(y))
# @inline Base.:(>=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
#     _ge(renormalize(x), renormalize(y))
# @inline Base.:(>=)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
#     _ge(renormalize(x), renormalize(y))


# @inline Base.:(==)(x::_MF{T,2}, y::_MF{T,2}) where {T} = _eq(x, y)
# @inline Base.:(==)(x::_MFV{M,T,2}, y::_MFV{M,T,2}) where {M,T} = _eq(x, y)
# @inline Base.:(!=)(x::_MF{T,2}, y::_MF{T,2}) where {T} = _ne(x, y)
# @inline Base.:(!=)(x::_MFV{M,T,2}, y::_MFV{M,T,2}) where {M,T} = _ne(x, y)
# @inline Base.:(<)(x::_MF{T,2}, y::_MF{T,2}) where {T} = _lt(x, y)
# @inline Base.:(<)(x::_MFV{M,T,2}, y::_MFV{M,T,2}) where {M,T} = _lt(x, y)
# @inline Base.:(>)(x::_MF{T,2}, y::_MF{T,2}) where {T} = _gt(x, y)
# @inline Base.:(>)(x::_MFV{M,T,2}, y::_MFV{M,T,2}) where {M,T} = _gt(x, y)
# @inline Base.:(<=)(x::_MF{T,2}, y::_MF{T,2}) where {T} = _le(x, y)
# @inline Base.:(<=)(x::_MFV{M,T,2}, y::_MFV{M,T,2}) where {M,T} = _le(x, y)
# @inline Base.:(>=)(x::_MF{T,2}, y::_MF{T,2}) where {T} = _ge(x, y)
# @inline Base.:(>=)(x::_MFV{M,T,2}, y::_MFV{M,T,2}) where {M,T} = _ge(x, y)


########################################################### ARITHMETIC OVERLOADS


# @inline Base.:-(x::_MF{T,N}) where {T,N} =
#     _MF{T,N}(ntuple(i -> -x._limbs[i], Val{N}()))
# @inline Base.:-(x::_MFV{M,T,N}) where {M,T,N} =
#     _MFV{M,T,N}(ntuple(i -> -x._limbs[i], Val{N}()))


# @inline _abs(x::_MF{T,N}) where {T,N} =
#     ifelse(signbit(x._limbs[1]), -x, x)
# @inline _abs(x::_MFV{M,T,N}) where {M,T,N} =
#     _MFV{M,T,N}(_mask_each(signbit(x._limbs[1]), (-x)._limbs, x._limbs))


# @inline Base.abs(x::_MF{T,N}) where {T,N} = _abs(renormalize(x))
# @inline Base.abs(x::_MFV{M,T,N}) where {M,T,N} = _abs(renormalize(x))
# @inline Base.abs(x::_MF{T,2}) where {T} = _abs(x)
# @inline Base.abs(x::_MFV{M,T,2}) where {M,T} = _abs(x)


# @inline Base.abs2(x::_MF{T,N}) where {T,N} = x * x
# @inline Base.abs2(x::_MFV{M,T,N}) where {M,T,N} = x * x
# @inline Base.abs2(x::_MF{T,2}) where {T} = _abs2(x)
# @inline Base.abs2(x::_MFV{M,T,2}) where {M,T} = _abs2(x)


# @inline Base.inv(x::_MF{T,N}) where {T,N} = one(_MF{T,N}) / x
# @inline Base.inv(x::_MFV{M,T,N}) where {M,T,N} = one(_MFV{M,T,N}) / x
# @inline Base.inv(x::_MF{T,2}) where {T} = _inv(x)
# @inline Base.inv(x::_MFV{M,T,2}) where {M,T} = _inv(x)


# @inline Base.:^(x::_MF{T,N}, p::Integer) where {T,N} =
#     signbit(p) ? Base.power_by_squaring(inv(x), -p) : Base.power_by_squaring(x, p)
# @inline Base.:^(x::_MFV{M,T,N}, p::Integer) where {M,T,N} =
#     signbit(p) ? Base.power_by_squaring(inv(x), -p) : Base.power_by_squaring(x, p)


# @inline Base.sum(x::_MFV{M,T,N}) where {M,T,N} =
#     +(ntuple(i -> x[i], Val{M}())...)


#################################################################### SQUARE ROOT


# Note: MultiFloats.unsafe_sqrt and MultiFloats.rsqrt are not exported to avoid
# potential name conflicts, but they are intended for external use by end users.


# @inline unsafe_sqrt(x::Float16) = Base.sqrt_llvm(x)
# @inline unsafe_sqrt(x::Float32) = Base.sqrt_llvm(x)
# @inline unsafe_sqrt(x::Float64) = Base.sqrt_llvm(x)
# @inline unsafe_sqrt(x::BigFloat) = sqrt(x)


# @inline rsqrt(x::Float16) = inv(unsafe_sqrt(x))
# @inline rsqrt(x::Float32) = inv(unsafe_sqrt(x))
# @inline rsqrt(x::Float64) = inv(unsafe_sqrt(x))
# @inline rsqrt(x::BigFloat) = inv(sqrt(x))


# @inline function _rsqrt(x::_MF{T,N}, ::Val{I}) where {T,N,I}
#     _one = one(T)
#     _half = inv(_one + _one)
#     r = _MF{T,N}(inv(unsafe_sqrt(x._limbs[1])))
#     h = scale(_half, x)
#     for _ = 1:I
#         r += r * (_half - h * (r * r))
#     end
#     return r
# end


# @inline function _rsqrt(x::_MFV{M,T,N}, ::Val{I}) where {M,T,N,I}
#     _one = one(T)
#     _half = inv(_one + _one)
#     _half_vec = Vec{M,T}(ntuple(_ -> _half, Val{M}()))
#     r = _MFV{M,T,N}(inv(sqrt(x._limbs[1])))
#     h = scale(_half, x)
#     for _ = 1:I
#         r += r * (_half_vec - h * (r * r))
#     end
#     return r
# end


# @inline rsqrt(x::_MF{T,1}) where {T} = _rsqrt(x, Val{0}())
# @inline rsqrt(x::_MF{T,N}) where {T,N} = _rsqrt(x, Val{(N + 1) >> 1}())
# @inline rsqrt(x::_MFV{M,T,1}) where {M,T} = _rsqrt(x, Val{0}())
# @inline rsqrt(x::_MFV{M,T,N}) where {M,T,N} = _rsqrt(x, Val{(N + 1) >> 1}())


# @inline unsafe_sqrt(x::_MF{T,N}) where {T,N} = inv(rsqrt(x))
# @inline unsafe_sqrt(x::_MFV{M,T,N}) where {M,T,N} = inv(rsqrt(x))


# @inline Base.sqrt(x::_MF{T,N}) where {T,N} =
#     iszero(x) ? zero(_MF{T,N}) : unsafe_sqrt(x)
# @inline Base.sqrt(x::_MFV{M,T,N}) where {M,T,N} =
#     _MFV{M,T,N}(_mask_each(!iszero(x), unsafe_sqrt(x)._limbs, zero(Vec{M,T})))


################################################################ PROMOTION RULES


# Base.promote_rule(::Type{_MF{T,N}}, ::Type{T}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{T}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{Vec{M,T}}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{_MF{T,N}}) where {M,T,N} = _MFV{M,T,N}


# Base.promote_rule(::Type{_MF{T,N}}, ::Type{Bool}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{Bool}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int8}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{Int8}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int16}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{Int16}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int32}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{Int32}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int64}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{Int64}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int128}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{Int128}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt8}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{UInt8}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt16}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{UInt16}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt32}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{UInt32}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt64}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{UInt64}) where {M,T,N} = _MFV{M,T,N}
# Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt128}) where {T,N} = _MF{T,N}
# Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{UInt128}) where {M,T,N} = _MFV{M,T,N}


# Base.promote_rule(::Type{_MF{T,N}}, ::Type{BigInt}) where {T,N} = BigFloat
# Base.promote_rule(::Type{_MF{T,N}}, ::Type{BigFloat}) where {T,N} = BigFloat


# Base.promote_rule(::Type{Float32x{N}}, ::Type{Float16}) where {N} = Float32x{N}
# Base.promote_rule(::Type{Float64x{N}}, ::Type{Float16}) where {N} = Float64x{N}
# Base.promote_rule(::Type{Float64x{N}}, ::Type{Float32}) where {N} = Float64x{N}


# @inline Base.:+(x::_MFV{M,T,N}, y::Number) where {M,T,N} = +(promote(x, y)...)
# @inline Base.:+(x::Number, y::_MFV{M,T,N}) where {M,T,N} = +(promote(x, y)...)
# @inline Base.:-(x::_MFV{M,T,N}, y::Number) where {M,T,N} = -(promote(x, y)...)
# @inline Base.:-(x::Number, y::_MFV{M,T,N}) where {M,T,N} = -(promote(x, y)...)
# @inline Base.:*(x::_MFV{M,T,N}, y::Number) where {M,T,N} = *(promote(x, y)...)
# @inline Base.:*(x::Number, y::_MFV{M,T,N}) where {M,T,N} = *(promote(x, y)...)
# @inline Base.:/(x::_MFV{M,T,N}, y::Number) where {M,T,N} = /(promote(x, y)...)
# @inline Base.:/(x::Number, y::_MFV{M,T,N}) where {M,T,N} = /(promote(x, y)...)


# @inline Base.convert(::Type{_MFV{M,T,N}}, x::Number) where {M,T,N} = _MFV{M,T,N}(x)


####################################################### TRANSCENDENTAL FUNCTIONS


# # TODO: frexp, modf, isqrt
# const _BASE_TRANSCENDENTAL_FUNCTIONS = Symbol[
#     :cbrt, :exp, :exp2, :exp10, :expm1, :log, :log2, :log10, :log1p,
#     :sin, :cos, :tan, :sec, :csc, :cot,
#     :sind, :cosd, :tand, :secd, :cscd, :cotd,
#     :asin, :acos, :atan, :asec, :acsc, :acot,
#     :asind, :acosd, :atand, :asecd, :acscd, :acotd,
#     :sinh, :cosh, :tanh, :sech, :csch, :coth,
#     :asinh, :acosh, :atanh, :asech, :acsch, :acoth,
#     :sinpi, :cospi, :sinc, :cosc, :deg2rad, :rad2deg,
# ]


# const _BASE_TRANSCENDENTAL_TUPLE_FUNCTIONS = Symbol[
#     :sincos, :sincosd, :sincospi,
# ]


# for name in _BASE_TRANSCENDENTAL_FUNCTIONS
#     eval(:(Base.$name(::MultiFloat{T,N}) where {T,N} = error($(
#         "$name(MultiFloat) is not yet implemented. For a temporary workaround,\n" *
#         "call MultiFloats.use_bigfloat_transcendentals() immediately after\n" *
#         "importing MultiFloats. This will use the BigFloat implementation of\n" *
#         "$name, which will not be as fast as a pure-MultiFloat implementation.\n"
#     ))))
# end


# for name in _BASE_TRANSCENDENTAL_TUPLE_FUNCTIONS
#     eval(:(Base.$name(::MultiFloat{T,N}) where {T,N} = error($(
#         "$name(MultiFloat) is not yet implemented. For a temporary workaround,\n" *
#         "call MultiFloats.use_bigfloat_transcendentals() immediately after\n" *
#         "importing MultiFloats. This will use the BigFloat implementation of\n" *
#         "$name, which will not be as fast as a pure-MultiFloat implementation.\n"
#     ))))
# end


# function use_bigfloat_transcendentals(num_extra_bits::Int=20)
#     for name in _BASE_TRANSCENDENTAL_FUNCTIONS
#         eval(:(Base.$name(x::MultiFloat{T,N}) where {T,N} = MultiFloat{T,N}(
#             _call_big($name, x, precision(MultiFloat{T,N}) + $num_extra_bits))))
#     end
#     for name in _BASE_TRANSCENDENTAL_TUPLE_FUNCTIONS
#         eval(:(Base.$name(x::MultiFloat{T,N}) where {T,N} = MultiFloat{T,N}.(
#             _call_big($name, x, precision(MultiFloat{T,N}) + $num_extra_bits))))
#     end
# end


################################################################# RANDOM NUMBERS


# using Random: AbstractRNG, CloseOpen01, SamplerTrivial, UInt52
# import Random: rand


# @inline function _rand_f64(rng::AbstractRNG, k::Int)
#     # Subnormal numbers are intentionally not generated.
#     if k < exponent(floatmin(Float64))
#         return zero(Float64)
#     end
#     expnt = reinterpret(UInt64,
#         exponent(floatmax(Float64)) + k) << (precision(Float64) - 1)
#     mntsa = rand(rng, UInt52())
#     return reinterpret(Float64, expnt | mntsa)
# end


# @inline function _rand_sf64(rng::AbstractRNG, k::Int)
#     # Subnormal numbers are intentionally not generated.
#     if k < exponent(floatmin(Float64))
#         return zero(Float64)
#     end
#     expnt = reinterpret(UInt64,
#         exponent(floatmax(Float64)) + k) << (precision(Float64) - 1)
#     mntsa = rand(rng, UInt64) & 0x800FFFFFFFFFFFFF
#     return reinterpret(Float64, expnt | mntsa)
# end


# @inline function _rand_mf64(
#     rng::AbstractRNG, offset::Int, padding::NTuple{N,Int}
# ) where {N}
#     exponents = cumsum(padding) .+ (precision(Float64) + 1) .* ntuple(identity, Val{N}())
#     return Float64x{N + 1}((
#         _rand_f64(rng, offset),
#         _rand_sf64.(rng, offset .- exponents)...
#     ))
# end


# @inline function rand(
#     rng::AbstractRNG, ::SamplerTrivial{CloseOpen01{Float64x{N}}}
# ) where {N}
#     offset = -leading_zeros(rand(rng, UInt64)) - 1
#     padding = ntuple(_ -> leading_zeros(rand(rng, UInt64)), Val{N - 1}())
#     return _rand_mf64(rng, offset, padding)
# end


end # module MultiFloats
