module MultiFloats

using Base.MPFR: CdoubleMax, MPFRRoundingMode, MPFRRoundNearest
using MPFR_jll: libmpfr
using SIMD: FastContiguousArray, Vec, vgather, vscatter
using SIMD.Intrinsics: extractelement

import SIMD: vifelse


############################################################### TYPE DEFINITIONS


export MultiFloat, MultiFloatVec


struct MultiFloat{T,N} <: AbstractFloat
    _limbs::NTuple{N,T}
    @inline MultiFloat{T,N}(limbs::NTuple{N,T}) where {T,N} = new(limbs)
end


struct MultiFloatVec{M,T,N}
    _limbs::NTuple{N,Vec{M,T}}
    @inline MultiFloatVec{M,T,N}(
        limbs::NTuple{N,Vec{M,T}}) where {M,T,N} = new(limbs)
end


# Private aliases for brevity.
const _MF = MultiFloat
const _MFV = MultiFloatVec


################################################################### TYPE ALIASES


export Float16x, Float32x, Float64x,
    Float32x1, Float32x2, Float32x3, Float32x4,
    Float64x1, Float64x2, Float64x3, Float64x4,
    Vec1Float32x1, Vec1Float32x2, Vec1Float32x3, Vec1Float32x4,
    Vec1Float64x1, Vec1Float64x2, Vec1Float64x3, Vec1Float64x4,
    Vec2Float32x1, Vec2Float32x2, Vec2Float32x3, Vec2Float32x4,
    Vec2Float64x1, Vec2Float64x2, Vec2Float64x3, Vec2Float64x4,
    Vec4Float32x1, Vec4Float32x2, Vec4Float32x3, Vec4Float32x4,
    Vec4Float64x1, Vec4Float64x2, Vec4Float64x3, Vec4Float64x4,
    Vec8Float32x1, Vec8Float32x2, Vec8Float32x3, Vec8Float32x4,
    Vec8Float64x1, Vec8Float64x2, Vec8Float64x3, Vec8Float64x4,
    Vec16Float32x1, Vec16Float32x2, Vec16Float32x3, Vec16Float32x4,
    Vec16Float64x1, Vec16Float64x2, Vec16Float64x3, Vec16Float64x4,
    Vec32Float32x1, Vec32Float32x2, Vec32Float32x3, Vec32Float32x4,
    Vec32Float64x1, Vec32Float64x2, Vec32Float64x3, Vec32Float64x4


const Float16x{N} = MultiFloat{Float16,N}
const Float32x{N} = MultiFloat{Float32,N}
const Float64x{N} = MultiFloat{Float64,N}


const Float32x1 = MultiFloat{Float32,1}
const Float32x2 = MultiFloat{Float32,2}
const Float32x3 = MultiFloat{Float32,3}
const Float32x4 = MultiFloat{Float32,4}
const Float64x1 = MultiFloat{Float64,1}
const Float64x2 = MultiFloat{Float64,2}
const Float64x3 = MultiFloat{Float64,3}
const Float64x4 = MultiFloat{Float64,4}


const Vec1Float32x1 = MultiFloatVec{1,Float32,1}
const Vec1Float32x2 = MultiFloatVec{1,Float32,2}
const Vec1Float32x3 = MultiFloatVec{1,Float32,3}
const Vec1Float32x4 = MultiFloatVec{1,Float32,4}
const Vec1Float64x1 = MultiFloatVec{1,Float64,1}
const Vec1Float64x2 = MultiFloatVec{1,Float64,2}
const Vec1Float64x3 = MultiFloatVec{1,Float64,3}
const Vec1Float64x4 = MultiFloatVec{1,Float64,4}
const Vec2Float32x1 = MultiFloatVec{2,Float32,1}
const Vec2Float32x2 = MultiFloatVec{2,Float32,2}
const Vec2Float32x3 = MultiFloatVec{2,Float32,3}
const Vec2Float32x4 = MultiFloatVec{2,Float32,4}
const Vec2Float64x1 = MultiFloatVec{2,Float64,1}
const Vec2Float64x2 = MultiFloatVec{2,Float64,2}
const Vec2Float64x3 = MultiFloatVec{2,Float64,3}
const Vec2Float64x4 = MultiFloatVec{2,Float64,4}
const Vec4Float32x1 = MultiFloatVec{4,Float32,1}
const Vec4Float32x2 = MultiFloatVec{4,Float32,2}
const Vec4Float32x3 = MultiFloatVec{4,Float32,3}
const Vec4Float32x4 = MultiFloatVec{4,Float32,4}
const Vec4Float64x1 = MultiFloatVec{4,Float64,1}
const Vec4Float64x2 = MultiFloatVec{4,Float64,2}
const Vec4Float64x3 = MultiFloatVec{4,Float64,3}
const Vec4Float64x4 = MultiFloatVec{4,Float64,4}
const Vec8Float32x1 = MultiFloatVec{8,Float32,1}
const Vec8Float32x2 = MultiFloatVec{8,Float32,2}
const Vec8Float32x3 = MultiFloatVec{8,Float32,3}
const Vec8Float32x4 = MultiFloatVec{8,Float32,4}
const Vec8Float64x1 = MultiFloatVec{8,Float64,1}
const Vec8Float64x2 = MultiFloatVec{8,Float64,2}
const Vec8Float64x3 = MultiFloatVec{8,Float64,3}
const Vec8Float64x4 = MultiFloatVec{8,Float64,4}
const Vec16Float32x1 = MultiFloatVec{16,Float32,1}
const Vec16Float32x2 = MultiFloatVec{16,Float32,2}
const Vec16Float32x3 = MultiFloatVec{16,Float32,3}
const Vec16Float32x4 = MultiFloatVec{16,Float32,4}
const Vec16Float64x1 = MultiFloatVec{16,Float64,1}
const Vec16Float64x2 = MultiFloatVec{16,Float64,2}
const Vec16Float64x3 = MultiFloatVec{16,Float64,3}
const Vec16Float64x4 = MultiFloatVec{16,Float64,4}
const Vec32Float32x1 = MultiFloatVec{32,Float32,1}
const Vec32Float32x2 = MultiFloatVec{32,Float32,2}
const Vec32Float32x3 = MultiFloatVec{32,Float32,3}
const Vec32Float32x4 = MultiFloatVec{32,Float32,4}
const Vec32Float64x1 = MultiFloatVec{32,Float64,1}
const Vec32Float64x2 = MultiFloatVec{32,Float64,2}
const Vec32Float64x3 = MultiFloatVec{32,Float64,3}
const Vec32Float64x4 = MultiFloatVec{32,Float64,4}


###################################################################### CONSTANTS


@inline Base.zero(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(ntuple(
    _ -> zero(T), Val{N}()))
@inline Base.zero(::Type{_MFV{M,T,N}}) where {M,T,N} = _MFV{M,T,N}(ntuple(
    _ -> zero(Vec{M,T}), Val{N}()))


@inline Base.zero(::_MF{T,N}) where {T,N} = zero(_MF{T,N})
@inline Base.zero(::_MFV{M,T,N}) where {M,T,N} = zero(_MFV{M,T,N})


@inline Base.one(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(ntuple(
    i -> (isone(i) ? one(T) : zero(T)), Val{N}()))
@inline Base.one(::Type{_MFV{M,T,N}}) where {M,T,N} = _MFV{M,T,N}(ntuple(
    i -> (isone(i) ? one(Vec{M,T}) : zero(Vec{M,T})), Val{N}()))


@inline Base.one(::_MF{T,N}) where {T,N} = one(_MF{T,N})
@inline Base.one(::_MFV{M,T,N}) where {M,T,N} = one(_MFV{M,T,N})


################################################################### CONSTRUCTORS


# Construct MultiFloat scalar from single scalar limb.
@inline _MF{T,N}(x::T) where {T<:Number,N} = _MF{T,N}(
    ntuple(i -> (isone(i) ? x : zero(T)), Val{N}()))


# Construct MultiFloat vector from single vector limb (SIMD.Vec).
@inline _MFV{M,T,N}(x::Vec{M,T}) where {M,T<:Number,N} = _MFV{M,T,N}(
    ntuple(i -> (isone(i) ? x : zero(Vec{M,T})), Val{N}()))


# Construct MultiFloat vector from single scalar limb.
@inline _MFV{M,T,N}(x::T) where {M,T,N} = _MFV{M,T,N}(Vec{M,T}(x))


# Construct MultiFloat vector from single vector limb (NTuple/Vararg).
@inline _MFV{M,T,N}(x::NTuple{M,T}) where {M,T,N} = _MFV{M,T,N}(Vec{M,T}(x))
@inline _MFV{M,T,N}(x::Vararg{T,M}) where {M,T,N} = _MFV{M,T,N}(Vec{M,T}(x))


# Construct MultiFloat scalar from MultiFloat scalar.
@inline _MF{T,N1}(x::_MF{T,N2}) where {T,N1,N2} = _MF{T,N1}(
    tuple(ntuple(i -> x._limbs[i], Val{min(N1, N2)}())...,
        ntuple(_ -> zero(T), Val{max(N1 - N2, 0)}())...))


# Construct MultiFloat vector from MultiFloat scalar.
@inline _MFV{M,T,N1}(x::_MF{T,N2}) where {M,T,N1,N2} = _MFV{M,T,N1}(
    tuple(ntuple(i -> Vec{M,T}(x._limbs[i]), Val{min(N1, N2)}())...,
        ntuple(_ -> zero(Vec{M,T}), Val{max(N1 - N2, 0)}())...))


# Construct MultiFloat vector from MultiFloat vector.
@inline _MFV{M,T,N1}(x::_MFV{M,T,N2}) where {M,T,N1,N2} = _MFV{M,T,N1}(
    tuple(ntuple(i -> x._limbs[i], Val{min(N1, N2)}())...,
        ntuple(_ -> zero(Vec{M,T}), Val{max(N1 - N2, 0)}())...))


# Construct MultiFloat vector from multiple MultiFloat scalars.
@inline _MFV{M,T,N}(xs::NTuple{M,_MF{T,N}}) where {M,T,N} = _MFV{M,T,N}(
    ntuple(j -> Vec{M,T}(ntuple(i -> xs[i]._limbs[j], Val{M}())), Val{N}()))
@inline _MFV{M,T,N}(xs::Vararg{_MF{T,N},M}) where {M,T,N} = _MFV{M,T,N}(
    ntuple(j -> Vec{M,T}(ntuple(i -> xs[i]._limbs[j], Val{M}())), Val{N}()))


################################################ CONVERSION FROM PRIMITIVE TYPES


@inline _split_impl(::Integer, ::Type, ::Val{0}) = ()
@inline _split_impl(::AbstractFloat, ::Type, ::Val{0}) = ()
@inline _split_impl(x::Integer, ::Type{T}, ::Val{1}) where {T} = (T(x),)
@inline _split_impl(x::AbstractFloat, ::Type{T}, ::Val{1}) where {T} = (T(x),)

@inline function _split_impl(x::I, ::Type{T}, ::Val{N}) where {I<:Integer,T,N}
    next_limb = T(x)
    remainder = reinterpret(signed(I), x - I(next_limb))
    return (next_limb, _split_impl(remainder, T, Val{N - 1}())...)
end

@inline function _split_impl(
    x::F,
    ::Type{T},
    ::Val{N},
) where {F<:AbstractFloat,T,N}
    next_limb = T(x)
    remainder = x - F(next_limb)
    return (next_limb, _split_impl(remainder, T, Val{N - 1}())...)
end

@inline function _split(x::Number, ::Type{T}, ::Val{N}, ::Val{K}) where {T,N,K}
    result = tuple(_split_impl(x, T, Val{min(N, K)}())...,
        ntuple(_ -> zero(T), Val{max(N - K, 0)}())...)
    first_limb = first(result)
    return ifelse(isfinite(first_limb), result,
        ntuple(_ -> first_limb, Val{N}()))
end


const _F16 = Float16
const _Fits16 = Union{Bool,Int8,UInt8}
const _Fits16x2 = Union{Int16,UInt16}
const _Fits16x3 = Union{Int32,UInt32,Float32}
const _Fits16x4 = Union{Int64,UInt64,Float64,Int128,UInt128}
const _Fits16xN = Union{_Fits16x2,_Fits16x3,_Fits16x4}

const _F32 = Float32
const _Fits32 = Union{Bool,Int8,UInt8,Int16,UInt16,Float16}
const _Fits32x2 = Union{Int32,UInt32}
const _Fits32x3 = Union{Int64,UInt64,Float64}
const _Fits32x6 = Union{Int128,UInt128}
const _Fits32xN = Union{_Fits32x2,_Fits32x3,_Fits32x6}

const _F64 = Float64
const _Fits64 = Union{Bool,Int8,UInt8,Int16,UInt16,Float16,Int32,UInt32,Float32}
const _Fits64x2 = Union{Int64,UInt64}
const _Fits64x3 = Union{Int128,UInt128}
const _Fits64xN = Union{_Fits64x2,_Fits64x3}


@inline _split(x::_Fits16x2, ::Type{_F16}, ::Val{N}) where {N} =
    _split(x, _F16, Val{N}(), Val{2}())
@inline _split(x::_Fits16x3, ::Type{_F16}, ::Val{N}) where {N} =
    _split(x, _F16, Val{N}(), Val{3}())
@inline _split(x::_Fits16x4, ::Type{_F16}, ::Val{N}) where {N} =
    _split(x, _F16, Val{N}(), Val{4}())
@inline _split(x::_Fits32x2, ::Type{_F32}, ::Val{N}) where {N} =
    _split(x, _F32, Val{N}(), Val{2}())
@inline _split(x::_Fits32x3, ::Type{_F32}, ::Val{N}) where {N} =
    _split(x, _F32, Val{N}(), Val{3}())
@inline _split(x::_Fits32x6, ::Type{_F32}, ::Val{N}) where {N} =
    _split(x, _F32, Val{N}(), Val{6}())
@inline _split(x::_Fits64x2, ::Type{_F64}, ::Val{N}) where {N} =
    _split(x, _F64, Val{N}(), Val{2}())
@inline _split(x::_Fits64x3, ::Type{_F64}, ::Val{N}) where {N} =
    _split(x, _F64, Val{N}(), Val{3}())


# Construct MultiFloat scalar from primitive scalar (single limb).
@inline _MF{_F16,N}(x::_Fits16) where {N} = _MF{_F16,N}(_F16(x))
@inline _MF{_F32,N}(x::_Fits32) where {N} = _MF{_F32,N}(_F32(x))
@inline _MF{_F64,N}(x::_Fits64) where {N} = _MF{_F64,N}(_F64(x))


# Construct MultiFloat vector from primitive scalar (single limb).
@inline _MFV{M,_F16,N}(x::_Fits16) where {M,N} = _MFV{M,_F16,N}(_F16(x))
@inline _MFV{M,_F32,N}(x::_Fits32) where {M,N} = _MFV{M,_F32,N}(_F32(x))
@inline _MFV{M,_F64,N}(x::_Fits64) where {M,N} = _MFV{M,_F64,N}(_F64(x))


# Construct MultiFloat scalar from primitive scalar (multiple limbs).
@inline _MF{_F16,N}(x::_Fits16xN) where {N} =
    _MF{_F16,N}(_split(x, _F16, Val{N}()))
@inline _MF{_F32,N}(x::_Fits32xN) where {N} =
    _MF{_F32,N}(_split(x, _F32, Val{N}()))
@inline _MF{_F64,N}(x::_Fits64xN) where {N} =
    _MF{_F64,N}(_split(x, _F64, Val{N}()))


################################################################ RENORMALIZATION


@inline function two_sum(a::T, b::T) where {T}
    sum = a + b
    a_prime = sum - b
    b_prime = sum - a_prime
    a_err = a - a_prime
    b_err = b - b_prime
    err = a_err + b_err
    return (sum, err)
end


@generated function _renorm_pass(x::NTuple{N,T}) where {N,T}
    xs = [Symbol('x', i) for i = 1:N]
    body = Expr[]
    push!(body, Expr(:meta, :inline))
    push!(body, Expr(:(=), Expr(:tuple, xs...), :x))
    for i = 1:2:N-1
        push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[i+1]),
            Expr(:call, two_sum, xs[i], xs[i+1])))
    end
    for i = 2:2:N-1
        push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[i+1]),
            Expr(:call, two_sum, xs[i], xs[i+1])))
    end
    push!(body, Expr(:return, Expr(:tuple, xs...)))
    return Expr(:block, body...)
end


@inline isnormalized(x::NTuple{N,T}) where {N,T} = (x === _renorm_pass(x))
@inline isnormalized(x::_MF{T,N}) where {T,N} = isnormalized(x._limbs)
@inline isnormalized(x::_MFV{M,T,N}) where {M,T,N} = isnormalized(x._limbs)


@inline function renormalize(x::NTuple{N,T}) where {N,T}
    total = +(reverse(x)...)
    if !isfinite(total)
        return ntuple(_ -> total, Val{N}())
    end
    while true
        x_next = _renorm_pass(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


@inline renormalize(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(renormalize(x._limbs))
@inline renormalize(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(renormalize(x._limbs))


####################################################### CONVERSION FROM BIGFLOAT


@inline function mpfr_sub!(x::BigFloat, y::CdoubleMax, rounding::RoundingMode)
    ccall((:mpfr_sub_d, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Cdouble, MPFRRoundingMode),
        x, x, y, convert(MPFRRoundingMode, rounding))
    return x
end


@inline function mpfr_sub!(
    x::BigFloat,
    y::BigFloat,
    z::CdoubleMax,
    rounding::RoundingMode,
)
    ccall((:mpfr_sub_d, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Cdouble, MPFRRoundingMode),
        x, y, z, convert(MPFRRoundingMode, rounding))
    return x
end


function _split!(x::BigFloat, ::Type{T}, ::Val{N}) where {T,N}
    value = T(x)
    if iszero(value) | !isfinite(value)
        return ntuple(_ -> value, Val{N}())
    else
        _zero = zero(T)
        result = ntuple(i -> isone(i) ? value : _zero, Val{N}())
        mpfr_sub!(x, value, RoundNearest)
        for i = 2:N
            limb = T(x)
            result = Base.setindex(result, limb, i)
            mpfr_sub!(x, limb, RoundNearest)
        end
        return renormalize(result)
    end
end


function _split(x::BigFloat, ::Type{T}, ::Val{N}) where {T,N}
    value = T(x)
    if iszero(value) | !isfinite(value)
        return ntuple(_ -> value, Val{N}())
    else
        _zero = zero(T)
        result = ntuple(i -> isone(i) ? value : _zero, Val{N}())
        temp = BigFloat(; precision=precision(x))
        mpfr_sub!(temp, x, value, RoundNearest)
        for i = 2:N
            limb = T(temp)
            result = Base.setindex(result, limb, i)
            mpfr_sub!(temp, limb, RoundNearest)
        end
        return renormalize(result)
    end
end


_MF{T,N}(x::BigFloat) where {T,N} = _MF{T,N}(_split(x, T, Val{N}()))


#################################################### CONVERSION FROM OTHER TYPES


@inline _full_precision(::Type{T}) where {T} =
    exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)


# Construct MultiFloat scalar from any other type by passing through BigFloat.
function _from_big(x::Any, ::Type{T}, ::Val{N}) where {T,N}
    p = 2 * _full_precision(T) + 1
    try
        return _MF{T,N}(_split!(
            BigFloat(x, RoundNearest; precision=p), T, Val{N}()))
    catch e
        if e isa MethodError
            return _MF{T,N}(_split!(BigFloat(x; precision=p), T, Val{N}()))
        end
        rethrow()
    end
end

_MF{T,N}(x::AbstractString) where {T,N} = _from_big(x, T, Val{N}())
_MF{T,N}(x::Rational) where {T,N} = _from_big(x, T, Val{N}())
_MF{T,N}(x::Number) where {T,N} = _from_big(x, T, Val{N}())


function Base.tryparse(::Type{_MF{T,N}}, x::AbstractString) where {T,N}
    try
        return _MF{T,N}(x)
    catch e
        if e isa ArgumentError
            return nothing
        end
        rethrow()
    end
end


# Construct MultiFloat vector from non-MultiFloat scalar.
@inline _MFV{M,T,N}(x::Union{AbstractString,Number}) where {M,T,N} =
    _MFV{M,T,N}(_MF{T,N}(x))


# Construct MultiFloat vector from multiple non-MultiFloat scalars.
@inline _MFV{M,T,N}(xs::NTuple{M,Union{AbstractString,Number}}) where {M,T,N} =
    _MFV{M,T,N}(_MF{T,N}.(xs))
@inline _MFV{M,T,N}(xs::Vararg{Union{AbstractString,Number},M}) where {M,T,N} =
    _MFV{M,T,N}(_MF{T,N}.(xs))
_MFV{M,T,N}(::NTuple{K,Union{AbstractString,Number}}) where {M,T,N,K} =
    error("MultiFloatVec constructor requires tuple of length $M.")
_MFV{M,T,N}(::Vararg{Union{AbstractString,Number},K}) where {M,T,N,K} =
    error("MultiFloatVec constructor requires 1 or $M arguments.")


######################################################## CONVERSION TO LIMB TYPE


@inline (::Type{T})(x::_MF{T,N}) where {T,N} = first(x._limbs)
@inline (::Type{Vec{M,T}})(x::_MFV{M,T,N}) where {M,T,N} = first(x._limbs)


######################################################### CONVERSION TO BIGFLOAT


@inline function mpfr_zero!(x::BigFloat)
    ccall((:mpfr_set_zero, libmpfr), Cvoid, (Ref{BigFloat}, Cint), x, 0)
    return x
end


@inline function mpfr_add!(x::BigFloat, y::CdoubleMax, rounding::RoundingMode)
    ccall((:mpfr_add_d, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Cdouble, MPFRRoundingMode),
        x, x, y, convert(MPFRRoundingMode, rounding))
    return x
end


function Base.BigFloat(
    x::_MF{T,N},
    rounding::RoundingMode=rounding(BigFloat);
    precision::Integer=precision(BigFloat),
) where {T,N}
    result = BigFloat(; precision)
    mpfr_zero!(result)
    for limb in x._limbs
        mpfr_add!(result, limb, rounding)
    end
    return result
end


###################################################### CONVERSION TO OTHER TYPES


@inline Base.AbstractFloat(x::_MF{T,N}) where {T,N} = x


Base.Rational{I}(x::_MF{T,N}) where {I<:Integer,T,N} =
    sum(Rational{I}.(x._limbs); init=zero(Rational{I}))
# This specialization eliminates ambiguity with the
# Rational{BigInt}(::AbstractFloat) method defined in Base.MPFR.
Base.Rational{BigInt}(x::_MF{T,N}) where {T,N} =
    sum(Rational{BigInt}.(x._limbs); init=zero(Rational{BigInt}))
Base.Rational(x::_MF{T,N}) where {T,N} = Rational{BigInt}(x)


################################################################### CANONIZATION


function canonize(x::_MF{T,N}) where {T,N}
    temp = BigFloat(; precision=(_full_precision(T) + ndigits(N; base=2)))
    mpfr_zero!(temp)
    for limb in x._limbs
        mpfr_add!(temp, limb, RoundNearest)
    end
    return _MF{T,N}(_split!(temp, T, Val{N}()))
end


function canonize(x::_MFV{M,T,N}) where {M,T,N}
    temp = BigFloat(; precision=(_full_precision(T) + ndigits(N; base=2)))
    result = ntuple(_ -> zero(_MF{T,N}), Val{M}())
    for i = 1:M
        mpfr_zero!(temp)
        for j = 1:N
            limb = extractelement(x._limbs[j].data, i - 1)
            mpfr_add!(temp, limb, RoundNearest)
        end
        result = Base.setindex(result, _MF{T,N}(_split!(temp, T, Val{N}())), i)
    end
    return _MFV{M,T,N}(result)
end


iscanonical(x::_MF{T,N}) where {T,N} = (x === canonize(x))
iscanonical(x::_MFV{M,T,N}) where {M,T,N} = (x === canonize(x))


###################################################### FLOATING-POINT PROPERTIES


@inline Base.eps(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(eps(T)^N)


@static if isdefined(Base, :_precision_with_base_2) # Julia 1.11+
    @inline Base._precision_with_base_2(::Type{_MF{T,N}}) where {T,N} =
        N * precision(T) - (N - 1)
elseif isdefined(Base, :_precision) # Julia 1.8-1.10
    @inline Base._precision(::Type{_MF{T,N}}) where {T,N} =
        N * precision(T) - (N - 1)
else # Julia 1.7 and earlier
    @inline Base.precision(::Type{_MF{T,N}}) where {T,N} =
        N * precision(T) - (N - 1)
end
# NOTE: SIMD.jl does not define Base.precision for vectors.


@inline Base.floatmin(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(floatmin(T))
@inline Base.floatmax(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(ntuple(
    i -> ldexp(floatmax(T), -((i - 1) * (precision(T) + 1))),
    Val{N}()))
# NOTE: SIMD.jl does not define Base.floatmin or Base.floatmax for vectors.


@inline Base.typemin(::Type{_MF{T,N}}) where {T,N} =
    _MF{T,N}(ntuple(_ -> typemin(T), Val{N}()))
@inline Base.typemax(::Type{_MF{T,N}}) where {T,N} =
    _MF{T,N}(ntuple(_ -> typemax(T), Val{N}()))
# NOTE: SIMD.jl does not define Base.typemin or Base.typemax for vectors.


################################################## FLOATING-POINT CLASSIFICATION


@inline Base.signbit(x::_MF{T,N}) where {T,N} = signbit(first(x._limbs))
@inline Base.signbit(x::_MFV{M,T,N}) where {M,T,N} = signbit(first(x._limbs))
@inline Base.exponent(x::_MF{T,N}) where {T,N} = exponent(first(x._limbs))
# NOTE: SIMD.jl does not define Base.exponent for vectors.
@inline Base.issubnormal(x::_MF{T,N}) where {T,N} =
    issubnormal(first(x._limbs))
@inline Base.issubnormal(x::_MFV{M,T,N}) where {M,T,N} =
    issubnormal(first(x._limbs))


@inline _vany(::Tuple{}, ::Val{M}) where {M} = zero(Vec{M,Bool})
@inline _vany(x::Tuple{Vec{M,Bool}}, ::Val{M}) where {M} = x[1]
@inline _vany(x::NTuple{N,Vec{M,Bool}}, ::Val{M}) where {M,N} = (|)(x...)


@inline _vall(::Tuple{}, ::Val{M}) where {M} = one(Vec{M,Bool})
@inline _vall(x::Tuple{Vec{M,Bool}}, ::Val{M}) where {M} = x[1]
@inline _vall(x::NTuple{N,Vec{M,Bool}}, ::Val{M}) where {M,N} = (&)(x...)


@inline Base.iszero(x::_MF{T,N}) where {T,N} =
    all(iszero.(x._limbs))
@inline Base.iszero(x::_MFV{M,T,N}) where {M,T,N} =
    _vall(iszero.(x._limbs), Val{M}())


@inline Base.isone(x::_MF{T,N}) where {T,N} = all(ntuple(
    i -> (isone(i) ? isone(x._limbs[i]) : iszero(x._limbs[i])),
    Val{N}()))
@inline Base.isone(x::_MFV{M,T,N}) where {M,T,N} = _vall(ntuple(
        i -> (isone(i) ? isone(x._limbs[i]) : iszero(x._limbs[i])),
        Val{N}()), Val{M}())


@inline Base.isfinite(x::_MF{T,N}) where {T,N} =
    all(isfinite.(x._limbs))
@inline Base.isfinite(x::_MFV{M,T,N}) where {M,T,N} =
    _vall(isfinite.(x._limbs), Val{M}())


@inline _has_pos_inf(x::_MF{T,N}) where {T,N} = any(ntuple(
    i -> isinf(x._limbs[i]) & !signbit(x._limbs[i]),
    Val{N}()))
@inline _has_pos_inf(x::_MFV{M,T,N}) where {M,T,N} = _vany(ntuple(
        i -> isinf(x._limbs[i]) & !signbit(x._limbs[i]),
        Val{N}()), Val{M}())


@inline _has_neg_inf(x::_MF{T,N}) where {T,N} = any(ntuple(
    i -> isinf(x._limbs[i]) & signbit(x._limbs[i]),
    Val{N}()))
@inline _has_neg_inf(x::_MFV{M,T,N}) where {M,T,N} = _vany(ntuple(
        i -> isinf(x._limbs[i]) & signbit(x._limbs[i]),
        Val{N}()), Val{M}())


@inline _has_nan(x::_MF{T,N}) where {T,N} =
    any(isnan.(x._limbs))
@inline _has_nan(x::_MFV{M,T,N}) where {M,T,N} =
    _vany(isnan.(x._limbs), Val{M}())


@inline Base.isinf(x::_MF{T,N}) where {T,N} =
    xor(_has_pos_inf(x), _has_neg_inf(x)) & !_has_nan(x)
@inline Base.isinf(x::_MFV{M,T,N}) where {M,T,N} =
    xor(_has_pos_inf(x), _has_neg_inf(x)) & !_has_nan(x)


@inline Base.isnan(x::_MF{T,N}) where {T,N} =
    _has_nan(x) | (_has_pos_inf(x) & _has_neg_inf(x))
@inline Base.isnan(x::_MFV{M,T,N}) where {M,T,N} =
    _has_nan(x) | (_has_pos_inf(x) & _has_neg_inf(x))


@inline Base.isinteger(x::_MF{T,N}) where {T,N} = all(isinteger.(x._limbs))
# NOTE: SIMD.jl does not define Base.isinteger for vectors.


#################################################### FLOATING-POINT MANIPULATION


@inline function mpfr_add!(
    x::BigFloat,
    y::BigFloat,
    z::CdoubleMax,
    rounding::RoundingMode,
)
    ccall((:mpfr_add_d, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Cdouble, MPFRRoundingMode),
        x, y, z, convert(MPFRRoundingMode, rounding))
    return x
end


@inline Base.ldexp(x::_MF{T,N}, n::Integer) where {T,N} =
    _MF{T,N}(ntuple(i -> ldexp(x._limbs[i], n), Val{N}()))
# NOTE: SIMD.jl does not define Base.ldexp for vectors.


function Base.decompose(x::_MF{T,N}) where {T,N}
    if iszero(x)
        return (zero(BigInt), 0, ifelse(signbit(x), -1, +1))
    end
    has_pos_inf = _has_pos_inf(x)
    has_neg_inf = _has_neg_inf(x)
    if _has_nan(x) | (has_pos_inf & has_neg_inf)
        return (zero(BigInt), 0, 0)
    elseif has_pos_inf
        return (+one(BigInt), 0, 0)
    elseif has_neg_inf
        return (-one(BigInt), 0, 0)
    end
    p = precision(T)
    e_min = typemax(Int)
    for limb in x._limbs
        if !iszero(limb)
            e_min = min(e_min, exponent(limb) - (p - 1))
        end
    end
    num = zero(BigInt)
    for limb in x._limbs
        if !iszero(limb)
            e = exponent(limb) - (p - 1)
            m = BigInt(ldexp(significand(limb), p - 1))
            num += m << (e - e_min)
        end
    end
    return (num, e_min, 1)
end


function Base.prevfloat(x::_MF{T,N}) where {T,N}
    _one = one(T)
    _two = _one + _one
    _half = inv(_two)

    has_pos_inf = _has_pos_inf(x)
    has_neg_inf = _has_neg_inf(x)
    if _has_nan(x) | (has_pos_inf & has_neg_inf)
        return x
    elseif has_pos_inf
        return floatmax(_MF{T,N})
    elseif has_neg_inf
        return -typemax(_MF{T,N})
    end

    total = BigFloat(; precision=(_full_precision(T) + ndigits(N; base=2)))
    mpfr_zero!(total)
    for limb in x._limbs
        mpfr_add!(total, limb, RoundNearest)
    end
    reference = _split(total, T, Val{N}())

    perturbation = max(_half * eps(reference[N]), eps(zero(T)))
    temp = BigFloat(; precision=(_full_precision(T) + ndigits(N; base=2)))
    while perturbation <= floatmax(T)
        mpfr_sub!(temp, total, perturbation, RoundNearest)
        candidate = _split!(temp, T, Val{N}())
        if candidate !== reference
            return _MF{T,N}(candidate)
        end
        perturbation *= _two
    end

    @assert false
end


function Base.nextfloat(x::_MF{T,N}) where {T,N}
    _one = one(T)
    _two = _one + _one
    _half = inv(_two)

    has_pos_inf = _has_pos_inf(x)
    has_neg_inf = _has_neg_inf(x)
    if _has_nan(x) | (has_pos_inf & has_neg_inf)
        return x
    elseif has_pos_inf
        return typemax(_MF{T,N})
    elseif has_neg_inf
        return -floatmax(_MF{T,N})
    end

    total = BigFloat(; precision=(_full_precision(T) + ndigits(N; base=2)))
    mpfr_zero!(total)
    for limb in x._limbs
        mpfr_add!(total, limb, RoundNearest)
    end
    reference = _split(total, T, Val{N}())

    perturbation = max(_half * eps(reference[N]), eps(zero(T)))
    temp = BigFloat(; precision=(_full_precision(T) + ndigits(N; base=2)))
    while perturbation <= floatmax(T)
        mpfr_add!(temp, total, perturbation, RoundNearest)
        candidate = _split!(temp, T, Val{N}())
        if candidate !== reference
            return _MF{T,N}(candidate)
        end
        perturbation *= _two
    end

    @assert false
end


# NOTE: SIMD.jl does not define Base.prevfloat or Base.nextfloat for vectors.


############################################################## VECTOR OPERATIONS


export mfvgather, mfvscatter


@inline Base.length(::_MFV{M,T,N}) where {M,T,N} = M


@inline Base.getindex(x::_MFV{M,T,N}, i::I) where {M,T,N,I} = _MF{T,N}(
    ntuple(j -> extractelement(x._limbs[j].data, i - one(I)), Val{N}()))


@inline vifelse(
    mask::Vec{M,Bool}, x::_MFV{M,T,N}, y::_MFV{M,T,N},
) where {M,T,N} = _MFV{M,T,N}(
    ntuple(i -> vifelse(mask, x._limbs[i], y._limbs[i]), Val{N}()))


@inline function mfvgather(
    pointer::Ptr{_MF{T,N}}, index::Vec{M,I}
) where {M,T,N,I<:Integer}
    base = reinterpret(Ptr{T}, pointer) + N * sizeof(T) * index
    return _MFV{M,T,N}(ntuple(
        i -> vgather(base + (i - 1) * sizeof(T)), Val{N}()))
end


@inline function mfvscatter(
    x::_MFV{M,T,N}, pointer::Ptr{_MF{T,N}}, index::Vec{M,I}
) where {M,T,N,I<:Integer}
    base = reinterpret(Ptr{T}, pointer) + N * sizeof(T) * index
    for i = 1:N
        vscatter(x._limbs[i], base + (i - 1) * sizeof(T), nothing)
    end
    return nothing
end


@inline mfvgather(
    array::FastContiguousArray{_MF{T,N},D},
    index::Vec{M,I},
) where {M,T,N,D,I<:Integer} = mfvgather(pointer(array), index - one(I))


@inline mfvscatter(
    x::_MFV{M,T,N},
    array::FastContiguousArray{_MF{T,N},D},
    index::Vec{M,I},
) where {M,T,N,D,I<:Integer} = mfvscatter(x, pointer(array), index - one(I))


##################################################################### COMPARISON


# TODO: Implement Base.cmp.


_eq_expr(n::Int) = (n == 1) ? :(x._limbs[1] == y._limbs[1]) : :(
    $(_eq_expr(n - 1)) & (x._limbs[$n] == y._limbs[$n]))
_ne_expr(n::Int) = (n == 1) ? :(x._limbs[1] != y._limbs[1]) : :(
    $(_ne_expr(n - 1)) | (x._limbs[$n] != y._limbs[$n]))
_lt_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$n] < y._limbs[$n]) : :(
    (x._limbs[$i] < y._limbs[$i]) |
    ((x._limbs[$i] == y._limbs[$i]) & $(_lt_expr(i + 1, n))))
_gt_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$n] > y._limbs[$n]) : :(
    (x._limbs[$i] > y._limbs[$i]) |
    ((x._limbs[$i] == y._limbs[$i]) & $(_gt_expr(i + 1, n))))
_le_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$n] <= y._limbs[$n]) : :(
    (x._limbs[$i] < y._limbs[$i]) |
    ((x._limbs[$i] == y._limbs[$i]) & $(_le_expr(i + 1, n))))
_ge_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$n] >= y._limbs[$n]) : :(
    (x._limbs[$i] > y._limbs[$i]) |
    ((x._limbs[$i] == y._limbs[$i]) & $(_ge_expr(i + 1, n))))


@generated Base.:(==)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _eq_expr(N)
@generated Base.:(==)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _eq_expr(N)
@generated Base.:(!=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _ne_expr(N)
@generated Base.:(!=)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _ne_expr(N)
@generated Base.:(<)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _lt_expr(1, N)
@generated Base.:(<)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _lt_expr(1, N)
@generated Base.:(>)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _gt_expr(1, N)
@generated Base.:(>)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _gt_expr(1, N)
@generated Base.:(<=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _le_expr(1, N)
@generated Base.:(<=)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _le_expr(1, N)
@generated Base.:(>=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _ge_expr(1, N)
@generated Base.:(>=)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _ge_expr(1, N)


################################################## LEVEL 0 ARITHMETIC OPERATIONS


@inline Base.copy(x::_MF{T,N}) where {T,N} = _MF{T,N}((copy).(x._limbs))
@inline Base.copy(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}((copy).(x._limbs))


@inline Base.:+(x::_MF{T,N}) where {T,N} = _MF{T,N}((+).(x._limbs))
@inline Base.:+(x::_MFV{M,T,N}) where {M,T,N} = _MFV{M,T,N}((+).(x._limbs))


@inline Base.:-(x::_MF{T,N}) where {T,N} = _MF{T,N}((-).(x._limbs))
@inline Base.:-(x::_MFV{M,T,N}) where {M,T,N} = _MFV{M,T,N}((-).(x._limbs))


@inline Base.abs(x::_MF{T,N}) where {T,N} = ifelse(signbit(x), -x, x)
@inline Base.abs(x::_MFV{M,T,N}) where {M,T,N} = vifelse(signbit(x), -x, x)


# NOTE: MultiFloats.scale is not exported to avoid name conflicts.
# Users are expected to call it as MultiFloats.scale(a, x).
@inline scale(a, x) = a * x
@inline scale(a::T, x::NTuple{N,T}) where {T,N} =
    ntuple(i -> a * x[i], Val{N}())
@inline scale(a::T, x::_MF{T,N}) where {T,N} =
    _MF{T,N}(scale(a, x._limbs))
@inline scale(a::T, x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(scale(a, x._limbs))


############################################################## ADDITION NETWORKS


@inline function fast_two_sum(a::T, b::T) where {T}
    sum = a + b
    b_prime = sum - a
    b_err = b - b_prime
    return (sum, b_err)
end


@inline function mfadd(
    x::NTuple{1,T},
    y::NTuple{1,T},
    ::Val{1},
) where {T}
    return (x[1] + y[1],)
end


@inline function mfadd(
    x::NTuple{2,T},
    y::NTuple{2,T},
    ::Val{2},
) where {T}
    a, b = two_sum(x[1], y[1])
    c, d = two_sum(x[2], y[2])
    a, c = fast_two_sum(a, c)
    b += d
    b += c
    a, b = fast_two_sum(a, b)
    return (a, b)
end


@inline function mfadd(
    x::NTuple{3,T},
    y::NTuple{3,T},
    ::Val{3},
) where {T}
    a, b = two_sum(x[1], y[1])
    c, d = two_sum(x[2], y[2])
    e, f = two_sum(x[3], y[3])
    a, c = fast_two_sum(a, c)
    b += f
    d, e = two_sum(d, e)
    a, d = fast_two_sum(a, d)
    b, c = two_sum(b, c)
    c += e
    c, d = two_sum(c, d)
    b, c = two_sum(b, c)
    a, b = fast_two_sum(a, b)
    c += d
    b, c = fast_two_sum(b, c)
    a, b = fast_two_sum(a, b)
    b, c = fast_two_sum(b, c)
    return (a, b, c)
end


@inline function mfadd(
    x::NTuple{4,T},
    y::NTuple{4,T},
    ::Val{4},
) where {T}
    a, b = two_sum(x[1], y[1])
    c, d = two_sum(x[2], y[2])
    e, f = two_sum(x[3], y[3])
    g, h = two_sum(x[4], y[4])
    a, c = fast_two_sum(a, c)
    b += h
    d, e = two_sum(d, e)
    f, g = two_sum(f, g)
    b, g = two_sum(b, g)
    c, d = fast_two_sum(c, d)
    e, f = two_sum(e, f)
    a, c = fast_two_sum(a, c)
    d, e = fast_two_sum(d, e)
    b, d = two_sum(b, d)
    c, g = fast_two_sum(c, g)
    e += f
    b, c = two_sum(b, c)
    d, e = two_sum(d, e)
    a, b = fast_two_sum(a, b)
    c, d = two_sum(c, d)
    e += g
    b, c = fast_two_sum(b, c)
    d, e = two_sum(d, e)
    a, b = fast_two_sum(a, b)
    c, d = fast_two_sum(c, d)
    b, c = fast_two_sum(b, c)
    d += e
    a, b = fast_two_sum(a, b)
    c, d = fast_two_sum(c, d)
    b, c = fast_two_sum(b, c)
    c, d = fast_two_sum(c, d)
    return (a, b, c, d)
end


######################################################## MULTIPLICATION NETWORKS


@inline one_prod(a::T, b::T) where {T} = a * b


@inline function two_prod(a::T, b::T) where {T}
    prod = a * b
    err = fma(a, b, -prod)
    return (prod, err)
end


@inline function mfmul(
    x::NTuple{1,T},
    y::NTuple{1,T},
    ::Val{1},
) where {T}
    return (x[1] * y[1],)
end


@inline function mfmul(
    x::NTuple{2,T},
    y::NTuple{2,T},
    ::Val{2},
) where {T}
    p00, e00 = two_prod(x[1], y[1])
    p01 = one_prod(x[1], y[2])
    p10 = one_prod(x[2], y[1])
    p01 += p10
    e00 += p01
    p00, e00 = fast_two_sum(p00, e00)
    return (p00, e00)
end


@inline function mfmul(
    x::NTuple{3,T},
    y::NTuple{3,T},
    ::Val{3},
) where {T}
    p00, e00 = two_prod(x[1], y[1])
    p01, e01 = two_prod(x[1], y[2])
    p10, e10 = two_prod(x[2], y[1])
    p02 = one_prod(x[1], y[3])
    p11 = one_prod(x[2], y[2])
    p20 = one_prod(x[3], y[1])
    p01, p10 = two_sum(p01, p10)
    e01 += e10
    p02 += p20
    e00, p01 = two_sum(e00, p01)
    p02 += p11
    p00, e00 = fast_two_sum(p00, e00)
    p01 += p10
    e01 += p02
    p01 += e01
    e00, p01 = two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    e00, p01 = fast_two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    return (p00, e00, p01)
end


@inline function mfmul(
    x::NTuple{4,T},
    y::NTuple{4,T},
    ::Val{4},
) where {T}
    p00, e00 = two_prod(x[1], y[1])
    p01, e01 = two_prod(x[1], y[2])
    p10, e10 = two_prod(x[2], y[1])
    p02, e02 = two_prod(x[1], y[3])
    p11, e11 = two_prod(x[2], y[2])
    p20, e20 = two_prod(x[3], y[1])
    p03 = one_prod(x[1], y[4])
    p12 = one_prod(x[2], y[3])
    p21 = one_prod(x[3], y[2])
    p30 = one_prod(x[4], y[1])
    p01, p10 = two_sum(p01, p10)
    e01, e10 = two_sum(e01, e10)
    p02, p20 = two_sum(p02, p20)
    e02 += e20
    p03 += p30
    p12 += p21
    e00, p01 = two_sum(e00, p01)
    e01, p11 = two_sum(e01, p11)
    e10 += e02
    p20 += e11
    p03 += p12
    p00, e00 = fast_two_sum(p00, e00)
    p01, p10 = fast_two_sum(p01, p10)
    e01, p02 = two_sum(e01, p02)
    e10 += p03
    p11 += p20
    p01, e01 = two_sum(p01, e01)
    p10 += p11
    e10 += p02
    p10 += e01
    p01, p10 = two_sum(p01, p10)
    e00, p01 = two_sum(e00, p01)
    p10 += e10
    p00, e00 = fast_two_sum(p00, e00)
    p01, p10 = two_sum(p01, p10)
    e00, p01 = two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    p01, p10 = fast_two_sum(p01, p10)
    e00, p01 = fast_two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    p01, p10 = fast_two_sum(p01, p10)
    e00, p01 = fast_two_sum(e00, p01)
    p01, p10 = fast_two_sum(p01, p10)
    return (p00, e00, p01, p10)
end


############################################################## SQUARING NETWORKS


# NOTE: MultiFloats.twice is not exported to avoid name conflicts.
# Users are expected to call it as MultiFloats.twice(x).
@inline twice(x) = x + x
@inline twice(x::_MF{T,N}) where {T,N} = _MF{T,N}(twice.(x._limbs))
@inline twice(x::_MFV{M,T,N}) where {M,T,N} = _MFV{M,T,N}(twice.(x._limbs))


@inline function mfsqr(
    x::NTuple{1,T},
    ::Val{1},
) where {T}
    return (x[1] * x[1],)
end


@inline function mfsqr(
    x::NTuple{2,T},
    ::Val{2},
) where {T}
    p00, e00 = two_prod(x[1], x[1])
    e00 = fma(x[1], twice(x[2]), e00)
    p00, e00 = fast_two_sum(p00, e00)
    return (p00, e00)
end


@inline function mfsqr(
    x::NTuple{3,T},
    ::Val{3},
) where {T}
    p00, e00 = two_prod(x[1], x[1])
    p01, e01 = two_prod(x[1], twice(x[2]))
    e00, p01 = two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    p01 += e01 + fma(x[1], twice(x[3]), one_prod(x[2], x[2]))
    e00, p01 = two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    e00, p01 = fast_two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    return (p00, e00, p01)
end


@inline function mfsqr(
    x::NTuple{4,T},
    ::Val{4},
) where {T}
    p00, e00 = two_prod(x[1], x[1])
    p01, e01 = two_prod(x[1], twice(x[2]))
    p02, e02 = two_prod(x[1], twice(x[3]))
    p11, e11 = two_prod(x[2], x[2])
    e00, p01 = two_sum(e00, p01)
    e01, p11 = two_sum(e01, p11)
    p00, e00 = fast_two_sum(p00, e00)
    e01, p02 = two_sum(e01, p02)
    p11 += e11
    p01, e01 = two_sum(p01, e01)
    p01, p10 = two_sum(p01, p11 + e01)
    e00, p01 = two_sum(e00, p01)
    p10 += (e02 + fma(x[1], twice(x[4]), one_prod(x[2], twice(x[3])))) + p02
    p00, e00 = fast_two_sum(p00, e00)
    p01, p10 = two_sum(p01, p10)
    e00, p01 = two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    p01, p10 = fast_two_sum(p01, p10)
    e00, p01 = fast_two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    p01, p10 = fast_two_sum(p01, p10)
    e00, p01 = fast_two_sum(e00, p01)
    p01, p10 = fast_two_sum(p01, p10)
    return (p00, e00, p01, p10)
end


################################################## LEVEL 1 ARITHMETIC OPERATIONS


@inline Base.:+(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _MF{T,N}(mfadd(x._limbs, y._limbs, Val{N}()))
@inline Base.:+(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(mfadd(x._limbs, y._limbs, Val{N}()))


@inline Base.:-(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = x + (-y)
@inline Base.:-(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = x + (-y)


@inline Base.:*(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _MF{T,N}(mfmul(x._limbs, y._limbs, Val{N}()))
@inline Base.:*(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(mfmul(x._limbs, y._limbs, Val{N}()))


@inline Base.abs2(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(mfsqr(x._limbs, Val{N}()))
@inline Base.abs2(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(mfsqr(x._limbs, Val{N}()))


@inline Base.sum(x::_MFV{M,T,N}) where {M,T,N} =
    +(ntuple(i -> x[i], Val{M}())...)


############################################################### POWER OPERATIONS


@inline Base.:^(x::_MF{T,N}, p::Integer) where {T,N} =
    signbit(p) ?
    power_by_squaring(inv(x), -p) :
    power_by_squaring(x, p)
@inline Base.:^(x::_MFV{M,T,N}, p::Integer) where {M,T,N} =
    signbit(p) ?
    power_by_squaring(inv(x), -p) :
    power_by_squaring(x, p)


function power_by_squaring(x, p::Integer)
    if p == 1
        return x
    elseif p == 0
        return one(x)
    elseif p == 2
        return abs2(x)
    elseif p < 0
        isone(x) && return x
        isone(-x) && return iseven(p) ? one(x) : x
        Base.throw_domerr_powbysq(x, p)
    end
    t = trailing_zeros(p) + 1
    p >>= t
    while (t -= 1) > 0
        x = abs2(x)
    end
    y = x
    while p > 0
        t = trailing_zeros(p) + 1
        p >>= t
        while (t -= 1) >= 0
            x = abs2(x)
        end
        y *= x
    end
    return y
end


######################################################### SQUARE ROOT OPERATIONS


# In Julia, Base.sqrt throws a DomainError when given a negative real argument.
# This is, in my opinion, a very unfortunate design choice. It forces otherwise
# non-throwing programs to unnecessarily include exception handling code paths.
# To facilitate the writing of branch-free code, MultiFloats.jl provides
# unsafe_sqrt and rsqrt functions that return NaN instead of throwing.


# NOTE: MultiFloats.unsafe_sqrt is not exported to avoid name conflicts.
# Users are expected to call it as MultiFloats.unsafe_sqrt(x).

@inline unsafe_sqrt(x::Any) = sqrt(x)

@inline unsafe_sqrt(x::Union{Float16,Float32,Float64}) = Base.sqrt_llvm(x)

function unsafe_sqrt(x::BigFloat)
    result = BigFloat()
    ccall((:mpfr_sqrt, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        result, x, MPFRRoundNearest)
    return result
end


# NOTE: MultiFloats.rsqrt is not exported to avoid name conflicts.
# Users are expected to call it as MultiFloats.rsqrt(x).

@inline rsqrt(x::Any) = inv(unsafe_sqrt(x))

function rsqrt(x::BigFloat)
    result = BigFloat()
    ccall((:mpfr_rec_sqrt, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        result, x, MPFRRoundNearest)
    return result
end


###################################################### KARP-MARKSTEIN ALGORITHMS


@inline _resize(x::NTuple{N,T}, ::Val{M}) where {T,N,M} =
    ntuple(i -> ((i <= N) ? x[i] : zero(T)), Val{M}())


# TODO: Develop half-to-full-width multiplication algorithms.
# TODO: Develop residual multiplication algorithms with built-in subtraction.


@inline function _mfinv_impl(
    x::NTuple{X,T},
    u::NTuple{U,T},
    ::Val{Z},
) where {T,X,U,Z}
    @assert 0 < U < Z
    _zero = zero(T)
    _one = one(T)
    if U + U >= Z
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{Z}())
        rx = _resize(x, Val{Z}())
        ru = _resize(u, Val{Z}())
        residual = mfadd(mfmul(rx, ru, Val{Z}()), _neg_one, Val{Z}())
        correction = mfmul(residual, ru, Val{Z}())
        return mfadd(ru, (-).(correction), Val{Z}())
    else
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{U + U}())
        rx = _resize(x, Val{U + U}())
        ru = _resize(u, Val{U + U}())
        residual = mfadd(mfmul(rx, ru, Val{U + U}()), _neg_one, Val{U + U}())
        correction = mfmul(residual, ru, Val{U + U}())
        next_u = mfadd(ru, (-).(correction), Val{U + U}())
        return _mfinv_impl(x, next_u, Val{Z}())
    end
end


@inline function _mfdiv_impl(
    x::NTuple{X,T},
    y::NTuple{Y,T},
    u::NTuple{U,T},
    ::Val{Z},
) where {T,X,Y,U,Z}
    @assert 0 < U < Z
    _zero = zero(T)
    _one = one(T)
    if U + U >= Z
        rx = _resize(x, Val{Z}())
        ry = _resize(y, Val{Z}())
        ru = _resize(u, Val{Z}())
        quotient = mfmul(rx, ru, Val{Z}())
        residual = mfadd(mfmul(quotient, ry, Val{Z}()), (-).(rx), Val{Z}())
        correction = mfmul(residual, ru, Val{Z}())
        return mfadd(quotient, (-).(correction), Val{Z}())
    else
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{U + U}())
        ry = _resize(y, Val{U + U}())
        ru = _resize(u, Val{U + U}())
        residual = mfadd(mfmul(ry, ru, Val{U + U}()), _neg_one, Val{U + U}())
        correction = mfmul(residual, ru, Val{U + U}())
        next_u = mfadd(ru, (-).(correction), Val{U + U}())
        return _mfdiv_impl(x, y, next_u, Val{Z}())
    end
end


@inline function _mfrsqrt_impl(
    x::NTuple{X,T},
    u::NTuple{U,T},
    ::Val{Z},
) where {T,X,U,Z}
    @assert 0 < U < Z
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _half = inv(_two)
    if U + U >= Z
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{Z}())
        rx = _resize(x, Val{Z}())
        ru = _resize(u, Val{Z}())
        u2 = mfsqr(ru, Val{Z}())
        residual = mfadd(mfmul(rx, u2, Val{Z}()), _neg_one, Val{Z}())
        correction = mfmul(residual, scale(_half, ru), Val{Z}())
        return mfadd(ru, (-).(correction), Val{Z}())
    else
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{U + U}())
        rx = _resize(x, Val{U + U}())
        ru = _resize(u, Val{U + U}())
        u2 = mfsqr(ru, Val{U + U}())
        residual = mfadd(mfmul(rx, u2, Val{U + U}()), _neg_one, Val{U + U}())
        correction = mfmul(residual, scale(_half, ru), Val{U + U}())
        next_u = mfadd(ru, (-).(correction), Val{U + U}())
        return _mfrsqrt_impl(x, next_u, Val{Z}())
    end
end


@inline function _mfsqrt_impl(
    x::NTuple{X,T},
    u::NTuple{U,T},
    ::Val{Z},
) where {T,X,U,Z}
    @assert 0 < U < Z
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _half = inv(_two)
    if U + U >= Z
        rx = _resize(x, Val{Z}())
        ru = _resize(u, Val{Z}())
        root = mfmul(rx, ru, Val{Z}())
        residual = mfadd(mfsqr(root, Val{Z}()), (-).(rx), Val{Z}())
        correction = mfmul(residual, scale(_half, ru), Val{Z}())
        return mfadd(root, (-).(correction), Val{Z}())
    else
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{U + U}())
        rx = _resize(x, Val{U + U}())
        ru = _resize(u, Val{U + U}())
        u2 = mfsqr(ru, Val{U + U}())
        residual = mfadd(mfmul(rx, u2, Val{U + U}()), _neg_one, Val{U + U}())
        correction = mfmul(residual, scale(_half, ru), Val{U + U}())
        next_u = mfadd(ru, (-).(correction), Val{U + U}())
        return _mfsqrt_impl(x, next_u, Val{Z}())
    end
end


@inline mfinv(x::NTuple{X,T}, ::Val{1}) where {T,X} =
    (inv(first(x)),)
@inline mfinv(x::NTuple{X,T}, ::Val{Z}) where {T,X,Z} =
    _mfinv_impl(x, (inv(first(x)),), Val{Z}())


@inline mfdiv(x::NTuple{X,T}, y::NTuple{Y,T}, ::Val{1}) where {T,X,Y} =
    (first(x) / first(y),)
@inline mfdiv(x::NTuple{X,T}, y::NTuple{Y,T}, ::Val{Z}) where {T,X,Y,Z} =
    _mfdiv_impl(x, y, (inv(first(y)),), Val{Z}())


@inline mfrsqrt(x::NTuple{X,T}, ::Val{1}) where {T,X} =
    (rsqrt(first(x)),)
@inline mfrsqrt(x::NTuple{X,T}, ::Val{Z}) where {T,X,Z} =
    _mfrsqrt_impl(x, (rsqrt(first(x)),), Val{Z}())


@inline mfsqrt(x::NTuple{X,T}, ::Val{1}) where {T,X} =
    (unsafe_sqrt(first(x)),)
@inline mfsqrt(x::NTuple{X,T}, ::Val{Z}) where {T,X,Z} =
    _mfsqrt_impl(x, (rsqrt(first(x)),), Val{Z}())


################################################## LEVEL 2 ARITHMETIC OPERATIONS


@inline Base.inv(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(mfinv(x._limbs, Val{N}()))
@inline Base.inv(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(mfinv(x._limbs, Val{N}()))


@inline Base.:/(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _MF{T,N}(mfdiv(x._limbs, y._limbs, Val{N}()))
@inline Base.:/(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(mfdiv(x._limbs, y._limbs, Val{N}()))


@inline rsqrt(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(mfrsqrt(x._limbs, Val{N}()))
@inline rsqrt(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(mfrsqrt(x._limbs, Val{N}()))


@inline unsafe_sqrt(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(mfsqrt(x._limbs, Val{N}()))
@inline unsafe_sqrt(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(mfsqrt(x._limbs, Val{N}()))


@inline Base.sqrt(x::_MF{T,N}) where {T,N} =
    ifelse(iszero(x), zero(x), unsafe_sqrt(x))
@inline Base.sqrt(x::_MFV{M,T,N}) where {M,T,N} =
    vifelse(iszero(x), zero(x), unsafe_sqrt(x))


####################################################################### PRINTING


using Printf: @sprintf


function hexfloat(x::T) where {T<:Base.IEEEFloat}
    _num_mantissa_bits = Base.significand_bits(T)
    _exponent_bias = Base.exponent_bias(T)
    _num_hex_digits = cld(_num_mantissa_bits, 4)
    _hex_shift = 4 * _num_hex_digits - _num_mantissa_bits
    _num_exponent_digits = ndigits(_exponent_bias)
    _string_length = 7 + _num_hex_digits + _num_exponent_digits

    U = Base.uinttype(T)
    bits = reinterpret(U, x)
    mantissa = bits & Base.significand_mask(T)
    biased_exponent = (bits & Base.exponent_mask(T)) >> _num_mantissa_bits
    unbiased_exponent = max(Int(biased_exponent), 1) - _exponent_bias
    if iszero(mantissa) & iszero(biased_exponent)
        unbiased_exponent = 0
    end

    buffer = Base.StringVector(_string_length)
    @inbounds begin
        buffer[1] = ifelse(signbit(x), UInt8('-'), UInt8('+'))
        buffer[2] = UInt8('0')
        buffer[3] = UInt8('x')
        buffer[4] = ifelse(iszero(biased_exponent), UInt8('0'), UInt8('1'))
        buffer[5] = UInt8('.')
        mantissa <<= _hex_shift
        for i = 0:_num_hex_digits-1
            shift = 4 * (_num_hex_digits - (i + 1))
            nibble = ((mantissa >> shift) % UInt8) & UInt8(0x0F)
            buffer[6+i] = nibble + ifelse(
                nibble < UInt8(10), UInt8('0'), UInt8('A') - UInt8(10))
        end
        buffer[6+_num_hex_digits] = UInt8('p')
        buffer[7+_num_hex_digits] = ifelse(
            signbit(unbiased_exponent), UInt8('-'), UInt8('+'))
        unbiased_exponent = abs(unbiased_exponent)
        for i = 0:_num_exponent_digits-1
            unbiased_exponent, digit = divrem(unbiased_exponent, 10)
            buffer[_string_length-i] = (digit % UInt8) + UInt8('0')
        end
    end
    return String(buffer)
end


function _format_digits(sign_str::String, digit_array::Vector{Int8}, e::Int)
    if 0 <= e < 6
        while length(digit_array) < e + 2
            push!(digit_array, zero(Int8))
        end
        pre_str = String('0' .+ digit_array[1:e+1])
        post_str = String('0' .+ digit_array[e+2:end])
        return @sprintf("%s%s.%s", sign_str, pre_str, post_str)
    elseif -5 < e < 0
        prepend!(digit_array, [zero(Int8) for _ = e:-2])
        post_str = String('0' .+ digit_array)
        return @sprintf("%s0.%s", sign_str, post_str)
    else
        while length(digit_array) < 2
            push!(digit_array, zero(Int8))
        end
        pre_char = '0' + digit_array[1]
        post_str = String('0' .+ digit_array[2:end])
        return @sprintf("%s%c.%se%d", sign_str, pre_char, post_str, e)
    end
end


function _half_past_floatmax(::Type{_MF{T,N}}) where {T,N}
    x = floatmax(_MF{T,N})
    return Rational{BigInt}(x) + Rational{BigInt}(eps(last(x._limbs))) // 2
end


function _to_string(x::_MF{T,N}) where {T,N}
    _floatmax = floatmax(T)
    _zero = zero(Int8)
    _one = one(Int8)
    _ten = Int8(10)

    if iszero(x)
        return signbit(x) ? "-0.0" : "0.0"
    end

    has_pos_inf = _has_pos_inf(x)
    has_neg_inf = _has_neg_inf(x)
    if _has_nan(x) | (has_pos_inf & has_neg_inf)
        return "NaN"
    elseif has_pos_inf
        return "Inf"
    elseif has_neg_inf
        return "-Inf"
    end

    px = prevfloat(x)
    nx = nextfloat(x)
    rx = Rational{BigInt}(x)
    prev = isfinite(px) ? Rational{BigInt}(px) : -_half_past_floatmax(_MF{T,N})
    next = isfinite(nx) ? Rational{BigInt}(nx) : +_half_past_floatmax(_MF{T,N})
    a = rx - (rx - prev) // 2
    b = rx
    c = rx + (next - rx) // 2
    @assert isfinite(a)
    @assert isfinite(b)
    @assert isfinite(c)
    @assert a < b < c

    if iszero(b) || ((a < 0) && (c > 0))
        return "0.0"
    end

    sign_str = ""
    if c <= 0
        a, b, c = -c, -b, -a
        sign_str = "-"
    end

    @assert 0 <= a < b < c

    e = 0
    while b < 1
        a *= 10
        b *= 10
        c *= 10
        e -= 1
    end
    while b >= 10
        a //= 10
        b //= 10
        c //= 10
        e += 1
    end

    digit_array = Vector{Int8}()
    while floor(a + 1) > ceil(c - 1)
        next_digit = floor(Int8, b)
        push!(digit_array, next_digit)
        a = 10 * (a - next_digit)
        b = 10 * (b - next_digit)
        c = 10 * (c - next_digit)
    end

    last_digit = clamp(round(Int8, b), floor(Int8, a + 1), ceil(Int8, c - 1))
    push!(digit_array, last_digit)
    i = lastindex(digit_array)
    while digit_array[i] >= _ten
        digit_array[i] -= _ten
        i -= 1
        if i < firstindex(digit_array)
            pushfirst!(digit_array, _zero)
            e += 1
            i = firstindex(digit_array)
        end
        digit_array[i] += _one
    end
    while iszero(last(digit_array))
        pop!(digit_array)
    end

    return _format_digits(sign_str, digit_array, e)
end


function Base.print(io::IO, x::_MF{T,N}) where {T,N}
    print(io, _to_string(x))
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", x::_MF{T,N}) where {T,N} = print(io, x)


function Base.print(io::IO, x::_MFV{M,T,N}) where {M,T,N}
    print(io, '<')
    print(io, M)
    print(io, " x ")
    print(io, T)
    print(io, " x ")
    print(io, N)
    print(io, ">[")
    for i = 1:M
        if i > 1
            print(io, ", ")
        end
        print(io, _to_string(x[i]))
    end
    print(io, ']')
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", x::_MFV{M,T,N}) where {M,T,N} =
    print(io, x)


function Base.show(io::IO, x::_MFV{M,T,N}) where {M,T,N}
    show(io, _MFV{M,T,N})
    print(io, "((")
    for i = 1:N
        if i > 1
            print(io, ", ")
        end
        show(io, Vec{M,T})
        print(io, "((")
        for j = 1:M
            if j > 1
                print(io, ", ")
            end
            show(io, x._limbs[i][j])
        end
        print(io, "))")
    end
    print(io, "))")
    return nothing
end


################################################# STANDARD LIBRARY COMPATIBILITY


_MF{T,N}(z::Complex) where {T,N} =
    isreal(z) ? _MF{T,N}(real(z)) :
    throw(InexactError(nameof(_MF{T,N}), _MF{T,N}, z))
# NOTE: SIMD.jl does not support complex vectors.


function Base.print(io::IO, z::Complex{_MF{T,N}}) where {T,N}
    x, y = reim(z)
    print(io, _to_string(x))
    print(io, ifelse(signbit(y), " - ", " + "))
    print(io, _to_string(abs(y)))
    print(io, "im")
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", z::Complex{_MF{T,N}}) where {T,N} =
    print(io, z)


import LinearAlgebra: floatmin2
@inline floatmin2(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(floatmin2(T))


import Printf: tofloat
tofloat(x::_MF{T,N}) where {T,N} = BigFloat(x;
    precision=(_full_precision(T) + ndigits(N; base=2)))


################################################################ PROMOTION RULES


# Promote MultiFloat scalar with scalar limb type.
Base.promote_rule(::Type{_MF{T,N}}, ::Type{T}) where {T,N} = _MF{T,N}


# Promote MultiFloat scalars with the same limb type.
Base.promote_rule(::Type{_MF{T,N1}}, ::Type{_MF{T,N2}}) where {T,N1,N2} =
    _MF{T,max(N1, N2)}


# Promote MultiFloat scalar with vector limb type.
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Vec{M,T}}) where {M,T,N} =
    _MFV{M,T,N}


# Promote MultiFloat vector with scalar limb type.
Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{T}) where {M,T,N} = _MFV{M,T,N}


# Promote MultiFloat vector with vector limb type.
Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{Vec{M,T}}) where {M,T,N} =
    _MFV{M,T,N}


# Promote MultiFloat vector with MultiFloat scalar.
Base.promote_rule(::Type{_MFV{M,T,N1}}, ::Type{_MF{T,N2}}) where {M,T,N1,N2} =
    _MFV{M,T,max(N1, N2)}


# Promote MultiFloat vectors with the same limb type.
Base.promote_rule(
    ::Type{_MFV{M,T,N1}},
    ::Type{_MFV{M,T,N2}},
) where {M,T,N1,N2} =
    _MFV{M,T,max(N1, N2)}


for S in [
    Bool,
    Int8, Int16, Int32, Int64, Int128,
    UInt8, UInt16, UInt32, UInt64, UInt128,
    Float16, Float32, Float64,
]
    # Promote MultiFloat scalar with fixed-precision real type.
    Base.promote_rule(::Type{_MF{T,N}}, ::Type{S}) where {T,N} =
        _MF{T,N}

    # Promote MultiFloat vector with fixed-precision real type.
    Base.promote_rule(::Type{_MFV{M,T,N}}, ::Type{S}) where {M,T,N} =
        _MFV{M,T,N}
end


# Promote MultiFloat scalar with arbitrary-precision real type.
Base.promote_rule(::Type{_MF{T,N}}, ::Type{BigInt}) where {T,N} = BigFloat
Base.promote_rule(::Type{_MF{T,N}}, ::Type{BigFloat}) where {T,N} = BigFloat


# Allow MultiFloat vector types to participate in the conversion system.
@inline Base.convert(::Type{_MFV{M,T,N}}, x::_MFV{M,T,N}) where {M,T,N} = x
@inline Base.convert(::Type{_MFV{M,T,N}}, x::Any) where {M,T,N} = _MFV{M,T,N}(x)


# Allow MultiFloat vector types to participate in the promotion system.
@inline Base.:(==)(x::_MFV{M,T,N}, y::Any) where {M,T,N} = ==(promote(x, y)...)
@inline Base.:(==)(x::Any, y::_MFV{M,T,N}) where {M,T,N} = ==(promote(x, y)...)
@inline Base.:(==)(x::_MFV{M,T,N1}, y::_MFV{M,T,N2}) where {M,T,N1,N2} =
    ==(promote(x, y)...)
@inline Base.:(!=)(x::_MFV{M,T,N}, y::Any) where {M,T,N} = !=(promote(x, y)...)
@inline Base.:(!=)(x::Any, y::_MFV{M,T,N}) where {M,T,N} = !=(promote(x, y)...)
@inline Base.:(!=)(x::_MFV{M,T,N1}, y::_MFV{M,T,N2}) where {M,T,N1,N2} =
    !=(promote(x, y)...)
@inline Base.:(<)(x::_MFV{M,T,N}, y::Any) where {M,T,N} = <(promote(x, y)...)
@inline Base.:(<)(x::Any, y::_MFV{M,T,N}) where {M,T,N} = <(promote(x, y)...)
@inline Base.:(<)(x::_MFV{M,T,N1}, y::_MFV{M,T,N2}) where {M,T,N1,N2} =
    <(promote(x, y)...)
@inline Base.:(>)(x::_MFV{M,T,N}, y::Any) where {M,T,N} = >(promote(x, y)...)
@inline Base.:(>)(x::Any, y::_MFV{M,T,N}) where {M,T,N} = >(promote(x, y)...)
@inline Base.:(>)(x::_MFV{M,T,N1}, y::_MFV{M,T,N2}) where {M,T,N1,N2} =
    >(promote(x, y)...)
@inline Base.:(<=)(x::_MFV{M,T,N}, y::Any) where {M,T,N} = <=(promote(x, y)...)
@inline Base.:(<=)(x::Any, y::_MFV{M,T,N}) where {M,T,N} = <=(promote(x, y)...)
@inline Base.:(<=)(x::_MFV{M,T,N1}, y::_MFV{M,T,N2}) where {M,T,N1,N2} =
    <=(promote(x, y)...)
@inline Base.:(>=)(x::_MFV{M,T,N}, y::Any) where {M,T,N} = >=(promote(x, y)...)
@inline Base.:(>=)(x::Any, y::_MFV{M,T,N}) where {M,T,N} = >=(promote(x, y)...)
@inline Base.:(>=)(x::_MFV{M,T,N1}, y::_MFV{M,T,N2}) where {M,T,N1,N2} =
    >=(promote(x, y)...)
@inline Base.:+(x::_MFV{M,T,N}, y::Any) where {M,T,N} = +(promote(x, y)...)
@inline Base.:+(x::Any, y::_MFV{M,T,N}) where {M,T,N} = +(promote(x, y)...)
@inline Base.:+(x::_MFV{M,T,N1}, y::_MFV{M,T,N2}) where {M,T,N1,N2} =
    +(promote(x, y)...)
@inline Base.:-(x::_MFV{M,T,N}, y::Any) where {M,T,N} = -(promote(x, y)...)
@inline Base.:-(x::Any, y::_MFV{M,T,N}) where {M,T,N} = -(promote(x, y)...)
@inline Base.:-(x::_MFV{M,T,N1}, y::_MFV{M,T,N2}) where {M,T,N1,N2} =
    -(promote(x, y)...)
@inline Base.:*(x::_MFV{M,T,N}, y::Any) where {M,T,N} = *(promote(x, y)...)
@inline Base.:*(x::Any, y::_MFV{M,T,N}) where {M,T,N} = *(promote(x, y)...)
@inline Base.:*(x::_MFV{M,T,N1}, y::_MFV{M,T,N2}) where {M,T,N1,N2} =
    *(promote(x, y)...)
@inline Base.:/(x::_MFV{M,T,N}, y::Any) where {M,T,N} = /(promote(x, y)...)
@inline Base.:/(x::Any, y::_MFV{M,T,N}) where {M,T,N} = /(promote(x, y)...)
@inline Base.:/(x::_MFV{M,T,N1}, y::_MFV{M,T,N2}) where {M,T,N1,N2} =
    /(promote(x, y)...)


####################################################### TRANSCENDENTAL FUNCTIONS


# TODO: Implement transcendental functions.
# TODO: frexp, modf, isqrt
const _BASE_TRANSCENDENTAL_FUNCTIONS = Symbol[
    :expm1, :log1p,
    :sin, :cos, :tan, :sec, :csc, :cot,
    :sind, :cosd, :tand, :secd, :cscd, :cotd,
    :asin, :acos, :atan, :asec, :acsc, :acot,
    :asind, :acosd, :atand, :asecd, :acscd, :acotd,
    :sinh, :cosh, :tanh, :sech, :csch, :coth,
    :asinh, :acosh, :atanh, :asech, :acsch, :acoth,
    :sinpi, :cospi, :sinc, :cosc, :deg2rad, :rad2deg,
]


const _BASE_TRANSCENDENTAL_TUPLE_FUNCTIONS = Symbol[
    :sincos, :sincosd, :sincospi,
]


for name in _BASE_TRANSCENDENTAL_FUNCTIONS
    eval(:(Base.$name(::MultiFloat{T,N}) where {T,N} = error($(
        "$name(::MultiFloat) is not yet implemented. For a workaround,\n" *
        "call MultiFloats.use_bigfloat_transcendentals() after importing\n" *
        "MultiFloats. This will use the BigFloat implementation of $name,\n" *
        "which will not be as fast as a pure-MultiFloat implementation.\n"
    ))))
end


for name in _BASE_TRANSCENDENTAL_TUPLE_FUNCTIONS
    eval(:(Base.$name(::MultiFloat{T,N}) where {T,N} = error($(
        "$name(::MultiFloat) is not yet implemented. For a workaround,\n" *
        "call MultiFloats.use_bigfloat_transcendentals() after importing\n" *
        "MultiFloats. This will use the BigFloat implementation of $name,\n" *
        "which will not be as fast as a pure-MultiFloat implementation.\n"
    ))))
end


function _eval_big(f::Function, x::MultiFloat{T,N}, p::Integer) where {T,N}
    return setprecision(BigFloat, p) do
        return f(BigFloat(x; precision=p))
    end
end


function use_bigfloat_transcendentals(num_extra_bits::Int=10)
    for name in _BASE_TRANSCENDENTAL_FUNCTIONS
        eval(:(Base.$name(x::MultiFloat{T,N}) where {T,N} = MultiFloat{T,N}(
            _eval_big($name, x, precision(MultiFloat{T,N}) + $num_extra_bits))))
    end
    for name in _BASE_TRANSCENDENTAL_TUPLE_FUNCTIONS
        eval(:(Base.$name(x::MultiFloat{T,N}) where {T,N} = MultiFloat{T,N}.(
            _eval_big($name, x, precision(MultiFloat{T,N}) + $num_extra_bits))))
    end
end


include("cbrt.jl")
include("exp.jl")
include("log.jl")


################################################################# RANDOM NUMBERS


using Random: AbstractRNG, CloseOpen01, SamplerTrivial, UInt23, UInt52
import Random: rand


@inline _rand_mantissa(rng::AbstractRNG, ::Type{Float32}) = rand(rng, UInt23())
@inline _rand_mantissa(rng::AbstractRNG, ::Type{Float64}) = rand(rng, UInt52())
@inline _rand_sign_mantissa(rng::AbstractRNG, ::Type{Float32}) =
    rand(rng, UInt32) & 0x807FFFFF
@inline _rand_sign_mantissa(rng::AbstractRNG, ::Type{Float64}) =
    rand(rng, UInt64) & 0x800FFFFFFFFFFFFF


@inline function _rand_e(rng::AbstractRNG, ::Type{T}, k::Int) where {T}
    # Subnormal numbers are intentionally not generated.
    if k < exponent(floatmin(T))
        return zero(T)
    end
    e = Base.uinttype(T)(exponent(floatmax(T)) + k) << (precision(T) - 1)
    return reinterpret(T, e | _rand_mantissa(rng, T))
end


@inline function _rand_se(rng::AbstractRNG, ::Type{T}, k::Int) where {T}
    # Subnormal numbers are intentionally not generated.
    if k < exponent(floatmin(T))
        return zero(T)
    end
    e = Base.uinttype(T)(exponent(floatmax(T)) + k) << (precision(T) - 1)
    return reinterpret(T, e | _rand_sign_mantissa(rng, T))
end


@inline function _rand_mf(
    rng::AbstractRNG,
    ::Type{T},
    offset::Int,
    padding::NTuple{N,Int},
) where {T,N}
    _iota = ntuple(identity, Val{N}())
    exponents = cumsum(padding) .+ (precision(T) + 1) .* _iota
    return _MF{T,N + 1}((_rand_e(rng, T, offset),
        _rand_se.(Ref(rng), T, offset .- exponents)...))
end


@inline function rand(
    rng::AbstractRNG,
    ::SamplerTrivial{CloseOpen01{_MF{T,N}}},
) where {T,N}
    offset = -leading_zeros(rand(rng, UInt64)) - 1
    padding = ntuple(_ -> leading_zeros(rand(rng, UInt64)), Val{N - 1}())
    return _rand_mf(rng, T, offset, padding)
end


end # module MultiFloats
