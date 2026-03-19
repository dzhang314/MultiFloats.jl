using Random: AbstractRNG, CloseOpen01, SamplerTrivial, SamplerType
import Random: rand, randn, randexp


@inline _rand_mantissa(rng::AbstractRNG, ::Type{T}) where {T} =
    rand(rng, Base.uinttype(T)) & Base.significand_mask(T)
@inline _rand_sign_mantissa(rng::AbstractRNG, ::Type{T}) where {T} =
    rand(rng, Base.uinttype(T)) & ~Base.exponent_mask(T)


@inline function _rand_limb(
    ::Type{T},
    e::S,
    mantissa_bits::U,
) where {T,S<:Signed,U<:Unsigned}
    _exponent_bias = S(Base.exponent_bias(T))
    _exponent_shift = S(Base.significand_bits(T))
    exponent_bits = reinterpret(U, _exponent_bias + e) << _exponent_shift
    result = reinterpret(T, exponent_bits | mantissa_bits)
    # Subnormal numbers are intentionally not generated.
    _min_exponent = S(exponent(floatmin(T)))
    return ifelse(e < _min_exponent, zero(T), result)
end

@inline function _rand_limb(
    ::Type{T},
    e::Vec{M,S},
    mantissa_bits::Vec{M,U},
) where {T,M,S<:Signed,U<:Unsigned}
    _exponent_bias = S(Base.exponent_bias(T))
    _exponent_shift = S(Base.significand_bits(T))
    exponent_bits = reinterpret(Vec{M,U}, _exponent_bias + e) << _exponent_shift
    result = reinterpret(Vec{M,T}, exponent_bits | mantissa_bits)
    # Subnormal numbers are intentionally not generated.
    _min_exponent = S(exponent(floatmin(T)))
    return vifelse(e < _min_exponent, zero(Vec{M,T}), result)
end


@inline _rand_trailing_limb(
    rng::AbstractRNG,
    ::Type{T},
    e::S,
) where {T,S<:Signed} = _rand_limb(T, e, _rand_sign_mantissa(rng, T))

@inline _rand_trailing_limb(
    rng::AbstractRNG,
    ::Type{T},
    e::Vec{M,S},
) where {T,M,S<:Signed} = _rand_limb(T, e, Vec(ntuple(
    @inline(_ -> _rand_sign_mantissa(rng, T)), Val{M}())))


@inline function _rand_mf(
    rng::AbstractRNG,
    leading_limb::T,
    ::Val{N},
) where {T,N}
    U = Base.uinttype(T)
    S = signed(U)
    _iota = ntuple(S, Val{N - 1}())
    offset = unsafe_exponent(leading_limb) % S
    padding = ntuple(_ -> leading_zeros(rand(rng, UInt64)) % S, Val{N - 1}())
    exponents = (cumsum(padding) .% S) .+ S(precision(T) + 1) .* _iota
    trailing_limbs = _rand_trailing_limb.(rng, T, offset .- exponents)
    return _MF{T,N}((leading_limb, trailing_limbs...))
end

@inline function _rand_mf(
    rng::AbstractRNG,
    leading_limb::Vec{M,T},
    ::Val{N},
) where {M,T,N}
    U = Base.uinttype(T)
    S = signed(U)
    _iota = ntuple(S, Val{N - 1}())
    offset = unsafe_exponent(leading_limb)
    padding = ntuple(@inline(_ -> Vec(ntuple(
            @inline(_ -> leading_zeros(rand(rng, UInt64)) % S),
            Val{M}()))), Val{N - 1}())
    exponents = cumsum(padding) .+ S(precision(T) + 1) .* _iota
    trailing_limbs = map(
        @inline(e -> _rand_trailing_limb(rng, T, offset - e)), exponents)
    return _MFV{M,T,N}((leading_limb, trailing_limbs...))
end


@inline function rand(
    rng::AbstractRNG,
    ::SamplerTrivial{CloseOpen01{_MF{T,N}}},
) where {T,N}
    S = signed(Base.uinttype(T))
    e = (-leading_zeros(rand(rng, UInt64)) - 1) % S
    mantissa_bits = _rand_mantissa(rng, T)
    return _rand_mf(rng, _rand_limb(T, e, mantissa_bits), Val{N}())
end

@inline function rand(
    rng::AbstractRNG,
    ::SamplerType{_MFV{M,T,N}},
) where {M,T,N}
    S = signed(Base.uinttype(T))
    e = Vec(ntuple(@inline(_ -> (-leading_zeros(rand(rng, UInt64)) - 1) % S),
        Val{M}()))
    mantissa_bits = Vec(ntuple(@inline(_ -> _rand_mantissa(rng, T)), Val{M}()))
    return _rand_mf(rng, _rand_limb(T, e, mantissa_bits), Val{N}())
end


@inline function randn(
    rng::AbstractRNG,
    ::Type{_MF{T,N}},
) where {T,N}
    leading_limb = randn(rng, T)
    return _rand_mf(rng, leading_limb, Val{N}())
end

@inline function randn(
    rng::AbstractRNG,
    ::Type{_MFV{M,T,N}},
) where {M,T,N}
    leading_limb = Vec{M,T}(ntuple(@inline(_ -> randn(rng, T)), Val{M}()))
    return _rand_mf(rng, leading_limb, Val{N}())
end


@inline function randexp(
    rng::AbstractRNG,
    ::Type{_MF{T,N}},
) where {T,N}
    leading_limb = randexp(rng, T)
    return _rand_mf(rng, leading_limb, Val{N}())
end

@inline function randexp(
    rng::AbstractRNG,
    ::Type{_MFV{M,T,N}},
) where {M,T,N}
    leading_limb = Vec{M,T}(ntuple(@inline(_ -> randexp(rng, T)), Val{M}()))
    return _rand_mf(rng, leading_limb, Val{N}())
end
