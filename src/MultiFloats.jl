module MultiFloats

using SIMD: Vec, vifelse, vgather
using SIMD.Intrinsics: extractelement


############################################################### TYPE DEFINITIONS


export MultiFloat, MultiFloatVec


struct MultiFloat{T,N} <: AbstractFloat
    _limbs::NTuple{N,T}
end


struct MultiFloatVec{M,T,N}
    _limbs::NTuple{N,Vec{M,T}}
end


# Private aliases for brevity.
const _MF = MultiFloat
const _MFV = MultiFloatVec


################################################################### TYPE ALIASES


export Float16x, Float32x, Float64x,
    Float64x1, Float64x2, Float64x3, Float64x4,
    Float64x5, Float64x6, Float64x7, Float64x8,
    v1Float64x1, v1Float64x2, v1Float64x3, v1Float64x4,
    v1Float64x5, v1Float64x6, v1Float64x7, v1Float64x8,
    v2Float64x1, v2Float64x2, v2Float64x3, v2Float64x4,
    v2Float64x5, v2Float64x6, v2Float64x7, v2Float64x8,
    v4Float64x1, v4Float64x2, v4Float64x3, v4Float64x4,
    v4Float64x5, v4Float64x6, v4Float64x7, v4Float64x8,
    v8Float64x1, v8Float64x2, v8Float64x3, v8Float64x4,
    v8Float64x5, v8Float64x6, v8Float64x7, v8Float64x8


const Float16x{N} = MultiFloat{Float16,N}
const Float32x{N} = MultiFloat{Float32,N}
const Float64x{N} = MultiFloat{Float64,N}


const Float64x1 = MultiFloat{Float64,1}
const Float64x2 = MultiFloat{Float64,2}
const Float64x3 = MultiFloat{Float64,3}
const Float64x4 = MultiFloat{Float64,4}
const Float64x5 = MultiFloat{Float64,5}
const Float64x6 = MultiFloat{Float64,6}
const Float64x7 = MultiFloat{Float64,7}
const Float64x8 = MultiFloat{Float64,8}


const v1Float64x1 = MultiFloatVec{1,Float64,1}
const v1Float64x2 = MultiFloatVec{1,Float64,2}
const v1Float64x3 = MultiFloatVec{1,Float64,3}
const v1Float64x4 = MultiFloatVec{1,Float64,4}
const v1Float64x5 = MultiFloatVec{1,Float64,5}
const v1Float64x6 = MultiFloatVec{1,Float64,6}
const v1Float64x7 = MultiFloatVec{1,Float64,7}
const v1Float64x8 = MultiFloatVec{1,Float64,8}


const v2Float64x1 = MultiFloatVec{2,Float64,1}
const v2Float64x2 = MultiFloatVec{2,Float64,2}
const v2Float64x3 = MultiFloatVec{2,Float64,3}
const v2Float64x4 = MultiFloatVec{2,Float64,4}
const v2Float64x5 = MultiFloatVec{2,Float64,5}
const v2Float64x6 = MultiFloatVec{2,Float64,6}
const v2Float64x7 = MultiFloatVec{2,Float64,7}
const v2Float64x8 = MultiFloatVec{2,Float64,8}


const v4Float64x1 = MultiFloatVec{4,Float64,1}
const v4Float64x2 = MultiFloatVec{4,Float64,2}
const v4Float64x3 = MultiFloatVec{4,Float64,3}
const v4Float64x4 = MultiFloatVec{4,Float64,4}
const v4Float64x5 = MultiFloatVec{4,Float64,5}
const v4Float64x6 = MultiFloatVec{4,Float64,6}
const v4Float64x7 = MultiFloatVec{4,Float64,7}
const v4Float64x8 = MultiFloatVec{4,Float64,8}


const v8Float64x1 = MultiFloatVec{8,Float64,1}
const v8Float64x2 = MultiFloatVec{8,Float64,2}
const v8Float64x3 = MultiFloatVec{8,Float64,3}
const v8Float64x4 = MultiFloatVec{8,Float64,4}
const v8Float64x5 = MultiFloatVec{8,Float64,5}
const v8Float64x6 = MultiFloatVec{8,Float64,6}
const v8Float64x7 = MultiFloatVec{8,Float64,7}
const v8Float64x8 = MultiFloatVec{8,Float64,8}


###################################################################### CONSTANTS


@inline Base.zero(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(ntuple(
    _ -> zero(T), Val{N}()))
@inline Base.zero(::Type{_MFV{M,T,N}}) where {M,T,N} = _MFV{M,T,N}(ntuple(
    _ -> zero(Vec{M,T}), Val{N}()))
@inline Base.one(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(ntuple(
    i -> ifelse(i == 1, one(T), zero(T)), Val{N}()))
@inline Base.one(::Type{_MFV{M,T,N}}) where {M,T,N} = _MFV{M,T,N}(ntuple(
    i -> ifelse(i == 1, one(Vec{M,T}), zero(Vec{M,T})), Val{N}()))


@inline Base.zero(::_MF{T,N}) where {T,N} = zero(_MF{T,N})
@inline Base.zero(::_MFV{M,T,N}) where {M,T,N} = zero(_MFV{M,T,N})
@inline Base.one(::_MF{T,N}) where {T,N} = one(_MF{T,N})
@inline Base.one(::_MFV{M,T,N}) where {M,T,N} = one(_MFV{M,T,N})


################################################################### CONSTRUCTORS


# Construct from a single limb: pad remaining limbs with zeroes.
@inline _MF{T,N}(x::T) where {T,N} = _MF{T,N}(ntuple(
    i -> ifelse(i == 1, x, zero(T)), Val{N}()))
@inline _MFV{M,T,N}(x::Vec{M,T}) where {M,T,N} = _MFV{M,T,N}(ntuple(
    i -> ifelse(i == 1, x, zero(Vec{M,T})), Val{N}()))
@inline _MFV{M,T,N}(x::NTuple{M,T}) where {M,T,N} = _MFV{M,T,N}(Vec{M,T}(x))


# Construct from multiple limbs: truncate or pad with zeroes.
@inline _MF{T,N1}(x::_MF{T,N2}) where {T,N1,N2} = _MF{T,N1}(tuple(
    ntuple(i -> x._limbs[i], Val{min(N1, N2)}())...,
    ntuple(_ -> zero(T), Val{max(N1 - N2, 0)}())...))
@inline _MFV{M,T,N1}(x::_MFV{M,T,N2}) where {M,T,N1,N2} = _MFV{M,T,N1}(tuple(
    ntuple(i -> x._limbs[i], Val{min(N1, N2)}())...,
    ntuple(_ -> zero(Vec{M,T}), Val{max(N1 - N2, 0)}())...))


# Construct vector from scalar: broadcast.
@inline _MFV{M,T,N}(x::T) where {M,T,N} = _MFV{M,T,N}(Vec{M,T}(x))
@inline _MFV{M,T,N}(x::_MF{T,N}) where {M,T,N} = _MFV{M,T,N}(ntuple(
    i -> Vec{M,T}(x._limbs[i]), Val{N}()))


# Construct vector from tuple of scalars: transpose.
@inline _MFV{M,T,N}(xs::NTuple{M,_MF{T,N}}) where {M,T,N} = _MFV{M,T,N}(ntuple(
    j -> Vec{M,T}(ntuple(i -> xs[i]._limbs[j], Val{M}())), Val{N}()))


################################################################ VECTOR INDEXING


export mfvgather


@inline Base.length(::_MFV{M,T,N}) where {M,T,N} = M


@inline Base.getindex(x::_MFV{M,T,N}, i::I) where {M,T,N,I} = _MF{T,N}(ntuple(
    j -> extractelement(x._limbs[j].data, i - one(I)), Val{N}()))


@inline function mfvgather(ptr::Ptr{_MF{T,N}}, idx::Vec{M,Int}) where {M,T,N}
    base = reinterpret(Ptr{T}, ptr) + N * sizeof(T) * idx
    return _MFV{M,T,N}(ntuple(
        i -> vgather(base + (i - 1) * sizeof(T)), Val{N}()))
end


################################################ CONVERSION FROM PRIMITIVE TYPES


# Bool, Int8, UInt8, Int16, UInt16, Float16, Int32, UInt32, and Float32
# can be directly converted to Float64 without losing precision.


@inline Float64x{N}(x::Bool) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Int8) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Int16) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Int32) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::UInt8) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::UInt16) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::UInt32) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Float16) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Float32) where {N} = Float64x{N}(Float64(x))


# Int64, UInt64, Int128, and UInt128 cannot be directly converted to Float64
# without losing precision, so they must be split into multiple components.


@inline Float64x1(x::Int64) = Float64x1(Float64(x))
@inline Float64x1(x::UInt64) = Float64x1(Float64(x))
@inline Float64x1(x::Int128) = Float64x1(Float64(x))
@inline Float64x1(x::UInt128) = Float64x1(Float64(x))


@inline function Float64x{N}(x::Int64) where {N}
    x0 = Float64(x)
    x1 = Float64(x - Int64(x0))
    return Float64x{N}((x0, x1, ntuple(_ -> 0.0, Val{N - 2}())...))
end


@inline function Float64x{N}(x::UInt64) where {N}
    x0 = Float64(x)
    x1 = Float64(reinterpret(Int64, x - UInt64(x0)))
    return Float64x{N}((x0, x1, ntuple(_ -> 0.0, Val{N - 2}())...))
end


@inline function Float64x2(x::Int128)
    x0 = Float64(x)
    x1 = Float64(x - Int128(x0))
    return Float64x2((x0, x1))
end


@inline function Float64x2(x::UInt128)
    x0 = Float64(x)
    x1 = Float64(reinterpret(Int128, x - UInt128(x0)))
    return Float64x2((x0, x1))
end


@inline function Float64x{N}(x::Int128) where {N}
    x0 = Float64(x)
    r1 = x - Int128(x0)
    x1 = Float64(r1)
    x2 = Float64(r1 - Int128(x1))
    return Float64x{N}((x0, x1, x2, ntuple(_ -> 0.0, Val{N - 3}())...))
end


@inline function Float64x{N}(x::UInt128) where {N}
    x0 = Float64(x)
    r1 = reinterpret(Int128, x - UInt128(x0))
    x1 = Float64(r1)
    x2 = Float64(r1 - Int128(x1))
    return Float64x{N}((x0, x1, x2, ntuple(_ -> 0.0, Val{N - 3}())...))
end


@inline _MFV{M,T,N}(x::Bool) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::Int8) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::Int16) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::Int32) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::Int64) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::Int128) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::UInt8) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::UInt16) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::UInt32) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::UInt64) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::UInt128) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::Float16) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::Float32) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))
@inline _MFV{M,T,N}(x::Float64) where {M,T,N} = _MFV{M,T,N}(_MF{T,N}(x))


######################################################################## SCALING


export scale


@inline scale(a, x) = a * x
@inline scale(a::T, x::_MF{T,N}) where {T,N} =
    _MF{T,N}(ntuple(i -> a * x._limbs[i], Val{N}()))
@inline scale(a::T, x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(ntuple(i -> a * x._limbs[i], Val{N}()))


########################################################## ERROR-FREE ARITHMETIC


@inline function fast_two_sum(a::T, b::T) where {T}
    sum = a + b
    b_prime = sum - a
    b_err = b - b_prime
    return (sum, b_err)
end


@inline function fast_two_diff(a::T, b::T) where {T}
    diff = a - b
    b_prime = a - diff
    b_err = b_prime - b
    return (diff, b_err)
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


@inline function two_diff(a::T, b::T) where {T}
    diff = a - b
    a_prime = diff + b
    b_prime = a_prime - diff
    a_err = a - a_prime
    b_err = b - b_prime
    err = a_err - b_err
    return (diff, err)
end


@inline function two_prod(a::T, b::T) where {T}
    prod = a * b
    err = fma(a, b, -prod)
    return (prod, err)
end


###################################################### METAPROGRAMMING UTILITIES


_inline_block() = [Expr(:meta, :inline)]


_meta_tuple(xs...) = Expr(:tuple, xs...)


_meta_unpack(lhs::Vector{Symbol}, rhs::Union{Symbol,Expr}) =
    Expr(:(=), _meta_tuple(lhs...), rhs)


_meta_fast_two_sum(s::Symbol, e::Symbol, a::Symbol, b::Symbol) =
    Expr(:(=), _meta_tuple(s, e), Expr(:call, :fast_two_sum, a, b))


_meta_two_sum(s::Symbol, e::Symbol, a::Symbol, b::Symbol) =
    Expr(:(=), _meta_tuple(s, e), Expr(:call, :two_sum, a, b))


_meta_two_diff(d::Symbol, e::Symbol, a::Symbol, b::Symbol) =
    Expr(:(=), _meta_tuple(d, e), Expr(:call, :two_diff, a, b))


_meta_prod(p::Symbol, a::Symbol, b::Symbol) =
    Expr(:(=), p, Expr(:call, :*, a, b))


_meta_two_prod(p::Symbol, e::Symbol, a::Symbol, b::Symbol) =
    Expr(:(=), _meta_tuple(p, e), Expr(:call, :two_prod, a, b))


function _meta_sum(T::DataType, xs::Vector{Symbol})
    if isempty(xs)
        return zero(T)
    elseif length(xs) == 1
        return only(xs)
    else
        xs = Vector{Union{Symbol,Expr}}(xs)
        while length(xs) > 1
            push!(xs, Expr(:call, :+, xs[1], xs[2]))
            deleteat!(xs, 1:2)
        end
        return only(xs)
    end
end


################################################################ RENORMALIZATION


export renormalize


function _one_pass_renorm_expr(T::DataType, num_inputs::Int, num_outputs::Int)
    @assert num_outputs > 0
    @assert ((num_inputs == num_outputs) ||
             (num_inputs == num_outputs + 1))
    code = _inline_block()

    # Unpack argument tuple.
    args = [Symbol('x', i) for i = 1:num_inputs]
    push!(code, _meta_unpack(args, :xs))

    # Generate one-pass renormalization code.
    for i = 1:num_outputs-1
        push!(code, _meta_fast_two_sum(args[i], args[i+1], args[i], args[i+1]))
    end

    # Return a tuple of renormalized terms.
    return Expr(:block, code..., Expr(:return, _meta_tuple(
        args[1:num_outputs-1]..., _meta_sum(T, args[num_outputs:end]))))
end


@generated _one_pass_renorm(::Val{N}, xs::T...) where {T,N} =
    _one_pass_renorm_expr(T, length(xs), N)


function _two_pass_renorm_expr(T::DataType, num_inputs::Int, num_outputs::Int)
    @assert num_outputs > 0
    @assert ((num_inputs == num_outputs) ||
             (num_inputs == num_outputs + 1))
    code = _inline_block()

    # Unpack argument tuple.
    args = [Symbol('x', i) for i = 1:num_inputs]
    push!(code, _meta_unpack(args, :xs))

    # Generate two-pass renormalization code.
    for i = num_inputs-1:-1:2
        push!(code, _meta_fast_two_sum(args[i], args[i+1], args[i], args[i+1]))
    end
    for i = 1:num_outputs-1
        push!(code, _meta_fast_two_sum(args[i], args[i+1], args[i], args[i+1]))
    end

    # Return a tuple of renormalized terms.
    return Expr(:block, code..., Expr(:return, _meta_tuple(
        args[1:num_outputs-1]..., _meta_sum(T, args[num_outputs:end]))))
end


@generated _two_pass_renorm(::Val{N}, xs::T...) where {T,N} =
    _two_pass_renorm_expr(T, length(xs), N)


# This function is needed to work around the following SIMD bug:
# https://github.com/eschnett/SIMD.jl/issues/115
@inline _ntuple_equal(x::NTuple{N,T}, y::NTuple{N,T}
) where {N,T} = all(x .== y)
@inline _ntuple_equal(x::NTuple{N,Vec{M,T}}, y::NTuple{N,Vec{M,T}}
) where {N,M,T} = all(all.(x .== y))


@inline function renormalize(xs::NTuple{N,T}) where {T,N}
    total = sum(xs)
    if !isfinite(total)
        return ntuple(_ -> total, Val{N}())
    end
    while true
        xs_new = _two_pass_renorm(Val{N}(), xs...)
        if _ntuple_equal(xs, xs_new)
            return xs
        else
            xs = xs_new
        end
    end
end


@inline _mask_each(
    mask::Vec{M,Bool}, x::NTuple{N,Vec{M,T}}, y::Vec{M,T}
) where {M,T,N} = ntuple(i -> vifelse(mask, x[i], y), Val{N}())
@inline _mask_each(
    mask::Vec{M,Bool}, x::NTuple{N,Vec{M,T}}, y::NTuple{N,Vec{M,T}}
) where {M,T,N} = ntuple(i -> vifelse(mask, x[i], y[i]), Val{N}())


@inline function renormalize(xs::NTuple{N,Vec{M,T}}) where {M,T,N}
    total = sum(xs)
    mask = isfinite(total)
    xs = _mask_each(mask, xs, zero(Vec{M,T}))
    while true
        xs_new = _two_pass_renorm(Val{N}(), xs...)
        if _ntuple_equal(xs, xs_new)
            return _mask_each(mask, xs, total)
        else
            xs = xs_new
        end
    end
end


@inline renormalize(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(renormalize(x._limbs))
@inline renormalize(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(renormalize(x._limbs))


################################################### FLOATING-POINT INTROSPECTION


@inline _and(::Tuple{}) = true
@inline _and(x::NTuple{1,Bool}) = x[1]
@inline _and(x::NTuple{N,Bool}) where {N} = (&)(x...)


@inline _vand(::Val{M}, ::Tuple{}) where {M} = one(Vec{M,Bool})
@inline _vand(::Val{M}, x::NTuple{1,Vec{M,Bool}}) where {M} = x[1]
@inline _vand(::Val{M}, x::NTuple{N,Vec{M,Bool}}) where {M,N} = (&)(x...)


@inline _iszero(x::_MF{T,N}) where {T,N} = _and(
    ntuple(i -> iszero(x._limbs[i]), Val{N}()))
@inline _iszero(x::_MFV{M,T,N}) where {M,T,N} = _vand(
    Val{M}(), ntuple(i -> iszero(x._limbs[i]), Val{N}()))
@inline _isone(x::_MF{T,N}) where {T,N} = isone(x._limbs[1]) & _and(
    ntuple(i -> iszero(x._limbs[i+1]), Val{N - 1}()))
@inline _isone(x::_MFV{M,T,N}) where {M,T,N} = isone(x._limbs[1]) & _vand(
    Val{M}(), ntuple(i -> iszero(x._limbs[i+1]), Val{N - 1}()))


@inline Base.iszero(x::_MF{T,N}) where {T,N} = _iszero(renormalize(x))
@inline Base.iszero(x::_MFV{M,T,N}) where {M,T,N} = _iszero(renormalize(x))
@inline Base.isone(x::_MF{T,N}) where {T,N} = _isone(renormalize(x))
@inline Base.isone(x::_MFV{M,T,N}) where {M,T,N} = _isone(renormalize(x))


@inline _head(x::_MF{T,N}) where {T,N} = renormalize(x)._limbs[1]
@inline _head(x::_MFV{M,T,N}) where {M,T,N} = renormalize(x)._limbs[1]


@inline Base.issubnormal(x::_MF{T,N}) where {T,N} = issubnormal(_head(x))
@inline Base.issubnormal(x::_MFV{M,T,N}) where {M,T,N} = issubnormal(_head(x))
@inline Base.isfinite(x::_MF{T,N}) where {T,N} = isfinite(_head(x))
@inline Base.isfinite(x::_MFV{M,T,N}) where {M,T,N} = isfinite(_head(x))
@inline Base.isinf(x::_MF{T,N}) where {T,N} = isinf(_head(x))
@inline Base.isinf(x::_MFV{M,T,N}) where {M,T,N} = isinf(_head(x))
@inline Base.isnan(x::_MF{T,N}) where {T,N} = isnan(_head(x))
@inline Base.isnan(x::_MFV{M,T,N}) where {M,T,N} = isnan(_head(x))
@inline Base.signbit(x::_MF{T,N}) where {T,N} = signbit(_head(x))
@inline Base.signbit(x::_MFV{M,T,N}) where {M,T,N} = signbit(_head(x))


# Note: SIMD.jl does not define Base.exponent or Base.isinteger for vectors.
@inline Base.exponent(x::_MF{T,N}) where {T,N} = exponent(_head(x))
@inline Base.isinteger(x::_MF{T,N}) where {T,N} =
    all(isinteger.(renormalize(x)._limbs))


# Note: SIMD.jl does not define Base.ldexp for vectors.
@inline function Base.ldexp(x::_MF{T,N}, n::I) where {T,N,I}
    x = renormalize(x)
    return _MF{T,N}(ntuple(i -> ldexp(x._limbs[i], n), Val{N}()))
end


# Note: SIMD.jl does not define Base.prevfloat or Base.nextfloat for vectors.
_prevfloat(x::_MF{T,N}) where {T,N} = renormalize(_MF{T,N}((ntuple(
        i -> x._limbs[i], Val{N - 1}())..., prevfloat(x._limbs[N]))))
_nextfloat(x::_MF{T,N}) where {T,N} = renormalize(_MF{T,N}((ntuple(
        i -> x._limbs[i], Val{N - 1}())..., nextfloat(x._limbs[N]))))
@inline Base.prevfloat(x::_MF{T,N}) where {T,N} = _prevfloat(renormalize(x))
@inline Base.nextfloat(x::_MF{T,N}) where {T,N} = _nextfloat(renormalize(x))


# Note: SIMD.jl does not define Base.precision for vectors.
if isdefined(Base, :_precision)
    @inline Base._precision(::Type{_MF{T,N}}) where {T,N} =
        N * precision(T) + (N - 1) # implicit bits of precision between limbs
else
    @inline Base.precision(::Type{_MF{T,N}}) where {T,N} =
        N * precision(T) + (N - 1) # implicit bits of precision between limbs
end


# Note: SIMD.jl does not define Base.eps for vectors.
@inline Base.eps(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(eps(T)^N)


# Note: SIMD.jl does not define Base.floatmin or Base.floatmax for vectors.
@inline Base.floatmin(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(floatmin(T))
@inline Base.floatmax(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(floatmax(T))
# TODO: This is technically not the maximum/minimum representable MultiFloat.


# Note: SIMD.jl does not define Base.typemin or Base.typemax for vectors.
@inline Base.typemin(::Type{_MF{T,N}}) where {T,N} =
    _MF{T,N}(ntuple(_ -> typemin(T), Val{N}()))
@inline Base.typemax(::Type{_MF{T,N}}) where {T,N} =
    _MF{T,N}(ntuple(_ -> typemax(T), Val{N}()))


################################################## CONVERSION TO PRIMITIVE TYPES


@inline Base.Float16(x::Float16x{N}) where {N} = _head(x)
@inline Base.Float32(x::Float32x{N}) where {N} = _head(x)
@inline Base.Float64(x::Float64x{N}) where {N} = _head(x)


@inline Base.Float16(x::Float32x{N}) where {N} = Float16(_head(x))
@inline Base.Float16(x::Float64x{N}) where {N} = Float16(_head(x))
@inline Base.Float32(x::Float64x{N}) where {N} = Float32(_head(x))


# TODO: Conversion from Float32x{N} to Float64.
# TODO: Conversion from Float16x{N} to Float32.
# TODO: Conversion from Float16x{N} to Float64.


######################################################## CONVERSION TO BIG TYPES


Base.BigFloat(x::_MF{T,N}) where {T,N} =
    +(BigFloat.(reverse(renormalize(x)._limbs))...)


Base.Rational{BigInt}(x::_MF{T,N}) where {T,N} =
    +(Rational{BigInt}.(reverse(renormalize(x)._limbs))...)


####################################################################### PRINTING


function _call_big(callback, x::_MF{T,N}) where {T,N}
    x = renormalize(x)
    total = +(reverse(x._limbs)...)
    if !isfinite(total)
        return setprecision(() -> callback(BigFloat(total)), precision(T))
    end
    i = N
    while (i > 0) && iszero(x._limbs[i])
        i -= 1
    end
    if iszero(i)
        return setprecision(() -> callback(zero(BigFloat)), precision(T))
    else
        return setprecision(
            () -> callback(BigFloat(x)),
            precision(T) + exponent(x._limbs[1]) - exponent(x._limbs[i])
        )
    end
end


function Base.show(io::IO, x::_MF{T,N}) where {T,N}
    _call_big(y -> show(io, y), x)
    return nothing
end


function Base.show(io::IO, x::_MFV{M,T,N}) where {M,T,N}
    write(io, '<')
    show(io, M)
    write(io, " x ")
    show(io, T)
    write(io, " x ")
    show(io, N)
    write(io, ">[")
    for i = 1:M
        if i > 1
            write(io, ", ")
        end
        _call_big(y -> show(io, y), x[i])
    end
    write(io, ']')
    return nothing
end


# import Printf: tofloat
# tofloat(x::_MF{T,N}) where {T,N} = _call_big(BigFloat, x)


##################################################################### COMPARISON


# TODO: MultiFloat-to-Float comparison.


_eq_expr(n::Int) = (n == 1) ? :(x._limbs[$n] == y._limbs[$n]) : :(
    $(_eq_expr(n - 1)) & (x._limbs[$n] == y._limbs[$n]))
_ne_expr(n::Int) = (n == 1) ? :(x._limbs[$n] != y._limbs[$n]) : :(
    $(_ne_expr(n - 1)) | (x._limbs[$n] != y._limbs[$n]))
_lt_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$i] < y._limbs[$i]) : :(
    (x._limbs[$i] < y._limbs[$i]) |
    ((x._limbs[$i] == y._limbs[$i]) & $(_lt_expr(i + 1, n))))
_gt_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$i] > y._limbs[$i]) : :(
    (x._limbs[$i] > y._limbs[$i]) |
    ((x._limbs[$i] == y._limbs[$i]) & $(_gt_expr(i + 1, n))))
_le_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$i] <= y._limbs[$i]) : :(
    (x._limbs[$i] < y._limbs[$i]) |
    ((x._limbs[$i] == y._limbs[$i]) & $(_le_expr(i + 1, n))))
_ge_expr(i::Int, n::Int) = (i == n) ? :(x._limbs[$i] >= y._limbs[$i]) : :(
    (x._limbs[$i] > y._limbs[$i]) |
    ((x._limbs[$i] == y._limbs[$i]) & $(_ge_expr(i + 1, n))))


@generated _eq(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _eq_expr(N)
@generated _eq(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _eq_expr(N)
@generated _ne(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _ne_expr(N)
@generated _ne(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _ne_expr(N)
@generated _lt(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _lt_expr(1, N)
@generated _lt(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _lt_expr(1, N)
@generated _gt(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _gt_expr(1, N)
@generated _gt(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _gt_expr(1, N)
@generated _le(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _le_expr(1, N)
@generated _le(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _le_expr(1, N)
@generated _ge(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = _ge_expr(1, N)
@generated _ge(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} = _ge_expr(1, N)


@inline Base.:(==)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _eq(renormalize(x), renormalize(y))
@inline Base.:(==)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _eq(renormalize(x), renormalize(y))
@inline Base.:(!=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _ne(renormalize(x), renormalize(y))
@inline Base.:(!=)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _ne(renormalize(x), renormalize(y))
@inline Base.:(<)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _lt(renormalize(x), renormalize(y))
@inline Base.:(<)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _lt(renormalize(x), renormalize(y))
@inline Base.:(>)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _gt(renormalize(x), renormalize(y))
@inline Base.:(>)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _gt(renormalize(x), renormalize(y))
@inline Base.:(<=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _le(renormalize(x), renormalize(y))
@inline Base.:(<=)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _le(renormalize(x), renormalize(y))
@inline Base.:(>=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    _ge(renormalize(x), renormalize(y))
@inline Base.:(>=)(x::_MFV{M,T,N}, y::_MFV{M,T,N}) where {M,T,N} =
    _ge(renormalize(x), renormalize(y))


########################################################### ARITHMETIC UTILITIES


function _accurate_sum_expr(T::DataType, num_inputs::Int, num_outputs::Int)
    @assert num_outputs > 0
    code = _inline_block()

    # Unpack argument tuple.
    args = [Symbol('x', i) for i = 1:num_inputs]
    push!(code, _meta_unpack(args, :xs))

    # Instantiate lists of terms of order 1, epsilon, epsilon^2, ...
    terms = [Symbol[] for _ = 1:num_outputs]

    # All input arguments are assumed to be on the same order of magnitude.
    # (If they are not, the error terms will automatically bubble down.)
    append!(terms[1], args)

    # Repeatedly call two_sum until only one term of each order remains.
    count = 0
    for j = 1:num_outputs-1
        curr_terms = terms[j]
        next_terms = terms[j+1]
        while length(curr_terms) > 1
            count += 1
            sum_term = Symbol('s', count)
            err_term = Symbol('e', count)
            push!(code, _meta_two_sum(sum_term, err_term,
                curr_terms[1], curr_terms[2]))
            deleteat!(curr_terms, 1:2)
            push!(curr_terms, sum_term)
            push!(next_terms, err_term)
        end
    end

    # Return a tuple containing the final term of each order.
    push!(code, Expr(:return, _meta_tuple(_meta_sum.(T, terms)...)))
    return Expr(:block, code...)
end


@generated _accurate_sum(::Val{N}, xs::T...) where {T,N} =
    _accurate_sum_expr(T, length(xs), N)


function _push_accumulation_code!(
    code::Vector{Expr},
    results::Vector{Symbol},
    terms::Vector{Vector{Symbol}}
)
    @assert length(terms) == length(results)
    count = 0
    for i = 1:length(terms)
        num_remain = length(terms) - i + 1
        num_spill = min(length(terms[i]), num_remain)
        lhs = [results[i]]
        for j = 2:num_spill
            count += 1
            push!(lhs, Symbol('t', count))
        end
        push!(code, _meta_unpack(lhs, Expr(:call, :_accurate_sum,
            Val(num_spill), terms[i]...)))
        for j = 2:num_spill
            push!(terms[i+j-1], lhs[j])
        end
    end
    return code
end


function _meta_type(vec_width::Int, T::DataType, num_limbs::Int)
    if vec_width == -1
        return Expr(:curly, :MultiFloat, T, num_limbs)
    else
        return Expr(:curly, :MultiFloatVec, vec_width, T, num_limbs)
    end
end


function _meta_mf(
    vec_width::Int, T::DataType, num_limbs::Int,
    code::Vector{Expr}, terms::Vector{Vector{Symbol}}, renorm_func::Symbol
)
    results = [Symbol('x', i) for i = 1:length(terms)]
    _push_accumulation_code!(code, results, terms)
    push!(code, Expr(:return, Expr(:call,
        _meta_type(vec_width, T, num_limbs),
        Expr(:call, renorm_func, Val(num_limbs), results...))))
    return Expr(:block, code...)
end


##################################################################### ARITHMETIC


function _add_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()
    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))
    b_limbs = [Symbol('b', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(b_limbs, :(b._limbs)))
    terms = [Symbol[] for _ = 1:num_limbs+1]
    for i = 1:num_limbs
        sum_term = Symbol('s', i)
        err_term = Symbol('e', i)
        push!(code, _meta_two_sum(sum_term, err_term,
            a_limbs[i], b_limbs[i]))
        push!(terms[i], sum_term)
        push!(terms[i+1], err_term)
    end
    return _meta_mf(vec_width, T, num_limbs, code, terms, :_two_pass_renorm)
end


function _addf_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()
    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))
    terms = [Symbol[] for _ = 1:num_limbs+1]
    last_term = :b
    for i = 1:num_limbs
        sum_term = Symbol('s', i)
        err_term = Symbol('e', i)
        push!(code, _meta_two_sum(sum_term, err_term,
            a_limbs[i], last_term))
        push!(terms[i], sum_term)
        last_term = err_term
    end
    push!(terms[num_limbs+1], last_term)
    return _meta_mf(vec_width, T, num_limbs, code, terms, :_two_pass_renorm)
end


function _sub_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()
    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))
    b_limbs = [Symbol('b', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(b_limbs, :(b._limbs)))
    terms = [Symbol[] for _ = 1:num_limbs+1]
    for i = 1:num_limbs
        diff_term = Symbol('d', i)
        err_term = Symbol('e', i)
        push!(code, _meta_two_diff(diff_term, err_term,
            a_limbs[i], b_limbs[i]))
        push!(terms[i], diff_term)
        push!(terms[i+1], err_term)
    end
    return _meta_mf(vec_width, T, num_limbs, code, terms, :_two_pass_renorm)
end


function _subf_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()
    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))
    terms = [Symbol[] for _ = 1:num_limbs+1]
    if num_limbs > 0
        diff_term = Symbol('d', 1)
        last_term = Symbol('e', 1)
        push!(code, _meta_two_diff(diff_term, last_term,
            a_limbs[1], :b))
        push!(terms[1], diff_term)
        for i = 2:num_limbs
            diff_term = Symbol('d', i)
            err_term = Symbol('e', i)
            push!(code, _meta_two_sum(diff_term, err_term,
                a_limbs[i], last_term))
            push!(terms[i], diff_term)
            last_term = err_term
        end
        push!(terms[num_limbs+1], last_term)
    else
        push!(code, :(d1 = -b))
        push!(terms[1], :d1)
    end
    return _meta_mf(vec_width, T, num_limbs, code, terms, :_two_pass_renorm)
end


function _fsub_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()
    b_limbs = [Symbol('b', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(b_limbs, :(b._limbs)))
    terms = [Symbol[] for _ = 1:num_limbs+1]
    last_term = :a
    for i = 1:num_limbs
        diff_term = Symbol('d', i)
        err_term = Symbol('e', i)
        push!(code, _meta_two_diff(diff_term, err_term,
            last_term, b_limbs[i]))
        push!(terms[i], diff_term)
        last_term = err_term
    end
    push!(terms[num_limbs+1], last_term)
    return _meta_mf(vec_width, T, num_limbs, code, terms, :_two_pass_renorm)
end


function _mul_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()
    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))
    b_limbs = [Symbol('b', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(b_limbs, :(b._limbs)))
    terms = [Symbol[] for _ = 1:num_limbs]
    count = 0
    for i = 1:num_limbs-1
        for j = 1:i
            count += 1
            prod_term = Symbol('p', count)
            err_term = Symbol('e', count)
            push!(code, _meta_two_prod(prod_term, err_term,
                a_limbs[j], b_limbs[i-j+1]))
            push!(terms[i], prod_term)
            push!(terms[i+1], err_term)
        end
    end
    for j = 1:num_limbs
        count += 1
        prod_term = Symbol('p', count)
        push!(code, _meta_prod(prod_term,
            a_limbs[j], b_limbs[num_limbs-j+1]))
        push!(terms[num_limbs], prod_term)
    end
    return _meta_mf(vec_width, T, num_limbs, code, terms, :_two_pass_renorm)
end


function _mulf_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()
    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))
    terms = [Symbol[] for _ = 1:num_limbs]
    for i = 1:num_limbs-1
        prod_term = Symbol('p', i)
        err_term = Symbol('e', i)
        push!(code, _meta_two_prod(prod_term, err_term,
            a_limbs[i], :b))
        push!(terms[i], prod_term)
        push!(terms[i+1], err_term)
    end
    if num_limbs > 0
        prod_term = Symbol('p', num_limbs)
        push!(code, _meta_prod(prod_term,
            a_limbs[num_limbs], :b))
        push!(terms[num_limbs], prod_term)
    end
    return _meta_mf(vec_width, T, num_limbs, code, terms, :_two_pass_renorm)
end


function _div_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()
    terms = [[Symbol('q', i)] for i = 1:num_limbs]
    for i = 1:num_limbs
        if i == 1
            push!(code, :(r = a))
        else
            push!(code, :(r = r - b * $(only(terms[i-1]))))
        end
        push!(code, :($(only(terms[i])) = r._limbs[1] / b._limbs[1]))
    end
    return _meta_mf(vec_width, T, num_limbs, code, terms, :_two_pass_renorm)
end


@generated _add(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = _add_expr(-1, T, N)
@generated _add(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = _add_expr(M, T, N)
@generated _addf(a::_MF{T,N}, b::T) where {T,N} = _addf_expr(-1, T, N)
@generated _addf(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} = _addf_expr(M, T, N)
@generated _sub(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = _sub_expr(-1, T, N)
@generated _sub(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = _sub_expr(M, T, N)
@generated _subf(a::_MF{T,N}, b::T) where {T,N} = _subf_expr(-1, T, N)
@generated _subf(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} = _subf_expr(M, T, N)
@generated _fsub(a::T, b::_MF{T,N}) where {T,N} = _fsub_expr(-1, T, N)
@generated _fsub(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T,N} = _fsub_expr(M, T, N)
@generated _mul(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = _mul_expr(-1, T, N)
@generated _mul(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = _mul_expr(M, T, N)
@generated _mulf(a::MultiFloat{T,N}, b::T) where {T,N} = _mulf_expr(-1, T, N)
@generated _mulf(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} = _mulf_expr(M, T, N)
@generated _div(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = _div_expr(-1, T, N)
@generated _div(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = _div_expr(M, T, N)


@inline Base.:+(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = _add(a, b)
@inline Base.:+(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = _add(a, b)
@inline Base.:+(a::_MF{T,N}, b::T) where {T,N} = _addf(a, b)
@inline Base.:+(a::_MF{T,N}, b::T) where {T<:Number,N} = _addf(a, b)
@inline Base.:+(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} = _addf(a, b)
@inline Base.:+(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T<:Number,N} = _addf(a, b)
@inline Base.:+(a::T, b::_MF{T,N}) where {T,N} = _addf(b, a)
@inline Base.:+(a::T, b::_MF{T,N}) where {T<:Number,N} = _addf(b, a)
@inline Base.:+(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T,N} = _addf(b, a)
@inline Base.:+(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T<:Number,N} = _addf(b, a)
@inline Base.:-(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = _sub(a, b)
@inline Base.:-(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = _sub(a, b)
@inline Base.:-(a::_MF{T,N}, b::T) where {T,N} = _subf(a, b)
@inline Base.:-(a::_MF{T,N}, b::T) where {T<:Number,N} = _subf(a, b)
@inline Base.:-(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} = _subf(a, b)
@inline Base.:-(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T<:Number,N} = _subf(a, b)
@inline Base.:-(a::T, b::_MF{T,N}) where {T,N} = _fsub(a, b)
@inline Base.:-(a::T, b::_MF{T,N}) where {T<:Number,N} = _fsub(a, b)
@inline Base.:-(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T,N} = _fsub(a, b)
@inline Base.:-(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T<:Number,N} = _fsub(a, b)
@inline Base.:*(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = _mul(a, b)
@inline Base.:*(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = _mul(a, b)
@inline Base.:*(a::_MF{T,N}, b::T) where {T,N} = _mulf(a, b)
@inline Base.:*(a::_MF{T,N}, b::T) where {T<:Number,N} = _mulf(a, b)
@inline Base.:*(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} = _mulf(a, b)
@inline Base.:*(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T<:Number,N} = _mulf(a, b)
@inline Base.:*(a::T, b::_MF{T,N}) where {T,N} = _mulf(b, a)
@inline Base.:*(a::T, b::_MF{T,N}) where {T<:Number,N} = _mulf(b, a)
@inline Base.:*(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T,N} = _mulf(b, a)
@inline Base.:*(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T<:Number,N} = _mulf(b, a)
@inline Base.:/(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = _div(a, b)
@inline Base.:/(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = _div(a, b)


########################################################### ARITHMETIC OVERLOADS


@inline Base.:-(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(ntuple(i -> -x._limbs[i], Val{N}()))
@inline Base.:-(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(ntuple(i -> -x._limbs[i], Val{N}()))


@inline _abs(x::_MF{T,N}) where {T,N} =
    ifelse(signbit(x._limbs[1]), -x, x)
@inline _abs(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(_mask_each(signbit(x._limbs[1]), (-x)._limbs, x._limbs))


@inline Base.abs(x::_MF{T,N}) where {T,N} = _abs(renormalize(x))
@inline Base.abs(x::_MFV{M,T,N}) where {M,T,N} = _abs(renormalize(x))


@inline Base.abs2(x::_MF{T,N}) where {T,N} = x * x
@inline Base.abs2(x::_MFV{M,T,N}) where {M,T,N} = x * x


@inline Base.inv(x::_MF{T,N}) where {T,N} = one(_MF{T,N}) / x
@inline Base.inv(x::_MFV{M,T,N}) where {M,T,N} = one(_MFV{M,T,N}) / x


@inline Base.sum(x::_MFV{M,T,N}) where {M,T,N} =
    +(ntuple(i -> x[i], Val{M}())...)


#################################################################### SQUARE ROOT


@inline unsafe_sqrt(x::Float16) = Base.sqrt_llvm(x)
@inline unsafe_sqrt(x::Float32) = Base.sqrt_llvm(x)
@inline unsafe_sqrt(x::Float64) = Base.sqrt_llvm(x)
@inline unsafe_sqrt(x::BigFloat) = sqrt(x)


@inline rsqrt(x::Float16) = inv(unsafe_sqrt(x))
@inline rsqrt(x::Float32) = inv(unsafe_sqrt(x))
@inline rsqrt(x::Float64) = inv(unsafe_sqrt(x))
@inline rsqrt(x::BigFloat) = inv(sqrt(x))


@inline function _rsqrt(x::_MF{T,N}, ::Val{I}) where {T,N,I}
    _one = one(T)
    _half = inv(_one + _one)
    r = _MF{T,N}(inv(unsafe_sqrt(x._limbs[1])))
    h = scale(_half, x)
    for _ = 1:I
        r += r * (_half - h * (r * r))
    end
    return r
end


@inline function _rsqrt(x::_MFV{M,T,N}, ::Val{I}) where {M,T,N,I}
    _one = one(T)
    _half = inv(_one + _one)
    _half_vec = Vec{M,T}(ntuple(_ -> _half, Val{M}()))
    r = _MFV{M,T,N}(inv(sqrt(x._limbs[1])))
    h = scale(_half, x)
    for _ = 1:I
        r += r * (_half_vec - h * (r * r))
    end
    return r
end


@inline rsqrt(x::_MF{Float64,1}) = _rsqrt(x, Val{0}())
@inline rsqrt(x::_MF{Float64,2}) = _rsqrt(x, Val{1}())
@inline rsqrt(x::_MF{Float64,3}) = _rsqrt(x, Val{2}())
@inline rsqrt(x::_MF{Float64,4}) = _rsqrt(x, Val{2}())
@inline rsqrt(x::_MF{Float64,5}) = _rsqrt(x, Val{3}())
@inline rsqrt(x::_MF{Float64,6}) = _rsqrt(x, Val{3}())
@inline rsqrt(x::_MF{Float64,7}) = _rsqrt(x, Val{3}())
@inline rsqrt(x::_MF{Float64,8}) = _rsqrt(x, Val{4}())


@inline rsqrt(x::_MFV{M,Float64,1}) where {M} = _rsqrt(x, Val{0}())
@inline rsqrt(x::_MFV{M,Float64,2}) where {M} = _rsqrt(x, Val{1}())
@inline rsqrt(x::_MFV{M,Float64,3}) where {M} = _rsqrt(x, Val{2}())
@inline rsqrt(x::_MFV{M,Float64,4}) where {M} = _rsqrt(x, Val{2}())
@inline rsqrt(x::_MFV{M,Float64,5}) where {M} = _rsqrt(x, Val{3}())
@inline rsqrt(x::_MFV{M,Float64,6}) where {M} = _rsqrt(x, Val{3}())
@inline rsqrt(x::_MFV{M,Float64,7}) where {M} = _rsqrt(x, Val{3}())
@inline rsqrt(x::_MFV{M,Float64,8}) where {M} = _rsqrt(x, Val{4}())


@inline unsafe_sqrt(x::_MF{T,N}) where {T,N} = inv(rsqrt(x))
@inline unsafe_sqrt(x::_MFV{M,T,N}) where {M,T,N} = inv(rsqrt(x))


@inline Base.sqrt(x::_MF{T,N}) where {T,N} =
    iszero(x) ? zero(_MF{T,N}) : unsafe_sqrt(x)
@inline Base.sqrt(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(_mask_each(!iszero(x), unsafe_sqrt(x)._limbs, zero(Vec{M,T})))


################################################################ PROMOTION RULES


Base.promote_rule(::Type{_MF{T,N}}, ::Type{Bool}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int8}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int16}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int32}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int64}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int128}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt8}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt16}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt32}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt64}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt128}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{BigFloat}) where {T,N} = BigFloat


#=
#################################################### CONSTRUCTION FROM BIG TYPES

tuple_from_collection(collection, ::Val{n}) where {n} =
    ntuple(
        let c = collection
            i -> c[begin-1+i]
        end,
        Val{n}())

function constructed_from_big(::Type{T}, ::Val{N}, x::Src) where
{T<:Real,N,Src<:Union{BigInt,BigFloat}}
    y = Vector{T}(undef, N)
    y[1] = T(x)
    for i = 2:N
        x -= y[i-1]
        y[i] = T(x)
    end
    MultiFloat{T,N}(tuple_from_collection(y, Val{N}()))
end

function MultiFloat{T,N}(x::BigFloat) where {T,N}
    if x > floatmax(T)
        return typemax(MultiFloat{T,N})
    elseif x < -floatmax(T)
        return typemin(MultiFloat{T,N})
    elseif -floatmin(T) < x < floatmin(T)
        return zero(MultiFloat{T,N})
    elseif isnan(x)
        return _MF{T,N}(ntuple(_ -> T(NaN), Val{N}()))
    end
    setrounding(
        let x = x
            () -> setprecision(
                let x = x
                    () -> constructed_from_big(T, Val{N}(), x)
                end,
                BigFloat,
                precision(x))
        end,
        BigFloat,
        RoundNearest)
end

MultiFloat{T,N}(x::BigInt) where {T,N} =
    constructed_from_big(T, Val{N}(), x)

MultiFloat{T,N}(x::Rational{U}) where {T,N,U} =
    MultiFloat{T,N}(numerator(x)) / MultiFloat{T,N}(denominator(x))

MultiFloat{T,N}(x::AbstractString) where {T,N} =
    MultiFloat{T,N}(BigFloat(x, precision=(
        precision(T) + exponent(floatmax(T)) - exponent(floatmin(T))
    )))

Base.promote_rule(::Type{Float32x{N}}, ::Type{Float16}) where {N} = Float32x{N}
Base.promote_rule(::Type{Float64x{N}}, ::Type{Float16}) where {N} = Float64x{N}
Base.promote_rule(::Type{Float64x{N}}, ::Type{Float32}) where {N} = Float64x{N}

############################################## BIGFLOAT TRANSCENDENTAL STOP-GAPS

BASE_TRANSCENDENTAL_FUNCTIONS = [
    :exp2, :exp10, :log2, :log10,
    :sin, :cos, :tan, :sec, :csc, :cot,
    :sinpi, :cospi,
    :sinh, :cosh, :tanh, :sech, :csch, :coth,
    :sind, :cosd, :tand, :secd, :cscd, :cotd,
    :asin, :acos, :atan, :asec, :acsc, :acot,
    :asinh, :acosh, :atanh, :asech, :acsch, :acoth,
    :asind, :acosd, :atand, :asecd, :acscd, :acotd
]

for name in BASE_TRANSCENDENTAL_FUNCTIONS
    eval(:(Base.$name(x::MultiFloat{T,N}) where {T,N} = error($(
        "$name(MultiFloat) is not yet implemented. For a temporary workaround,\n" *
        "call MultiFloats.use_bigfloat_transcendentals() immediately after\n" *
        "importing MultiFloats. This will use the BigFloat implementation of\n" *
        "$name, which will not be as fast as a pure-MultiFloat implementation.\n"
    ))))
end

eval_bigfloat(f::Function, x::MultiFloat{T,N}, k::Int) where {T,N} =
    setprecision(
        () -> MultiFloat{T,N}(f(BigFloat(x))),
        BigFloat, precision(MultiFloat{T,N}) + k
    )

function use_bigfloat_transcendentals(k::Int=20)
    for name in BASE_TRANSCENDENTAL_FUNCTIONS
        eval(:(
            Base.$name(x::MultiFloat{T,N}) where {T,N} =
                eval_bigfloat($name, x, $k)
        ))
    end
end

####################################################### RANDOM NUMBER GENERATION

import Random
using Random: AbstractRNG, SamplerTrivial, CloseOpen01

@inline function _rand_f64(rng::AbstractRNG, k::Int)
    expnt = reinterpret(UInt64,
        exponent(floatmax(Float64)) + k) << (precision(Float64) - 1)
    mntsa = rand(rng, Random.UInt52())
    return reinterpret(Float64, expnt | mntsa)
end

@inline function _rand_sf64(rng::AbstractRNG, k::Int)
    expnt = reinterpret(UInt64,
        exponent(floatmax(Float64)) + k) << (precision(Float64) - 1)
    mntsa = rand(rng, UInt64) & 0x800FFFFFFFFFFFFF
    return reinterpret(Float64, expnt | mntsa)
end

@inline function _rand_mf64(
    rng::AbstractRNG, offset::Int, padding::NTuple{N,Int}
) where {N}
    exponents = (
        cumsum(padding) .+
        (precision(Float64) + 1) .* ntuple(identity, Val{N}())
    )
    return Float64x{N + 1}((
        _rand_f64(rng, offset),
        _rand_sf64.(rng, offset .- exponents)...
    ))
end

function multifloat_rand_func(n::Int)
    return :(
        Random.rand(
            rng::AbstractRNG,
            ::SamplerTrivial{CloseOpen01{Float64x{$n}}}
        ) = _rand_mf64(
            rng,
            -leading_zeros(rand(rng, UInt64)) - 1,
            $(Expr(:tuple, ntuple(
                _ -> :(leading_zeros(rand(rng, UInt64))),
                Val{n - 1}()
            )...))
        )
    )
end

=#

end # module MultiFloats
