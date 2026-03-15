using Random: AbstractRNG, CloseOpen01, SamplerTrivial, UInt23, UInt52
import Random: rand, randn, randexp


@inline _rand_mantissa(rng::AbstractRNG, ::Type{Float32}) = rand(rng, UInt23())
@inline _rand_mantissa(rng::AbstractRNG, ::Type{Float64}) = rand(rng, UInt52())
@inline _rand_sign_mantissa(rng::AbstractRNG, ::Type{Float32}) =
    rand(rng, UInt32) & 0x807FFFFF
@inline _rand_sign_mantissa(rng::AbstractRNG, ::Type{Float64}) =
    rand(rng, UInt64) & 0x800FFFFFFFFFFFFF


@inline function _rand_leading_limb(
    rng::AbstractRNG,
    ::Type{T},
    e::Int,
) where {T}
    U = Base.uinttype(T)
    e_bits = ((Base.exponent_bias(T) + e) % U) << Base.significand_bits(T)
    result = reinterpret(T, e_bits | _rand_mantissa(rng, T))
    # Subnormal numbers are intentionally not generated.
    return ifelse(e < exponent(floatmin(T)), zero(T), result)
end


@inline function _rand_trailing_limb(
    rng::AbstractRNG,
    ::Type{T},
    e::Int,
) where {T}
    U = Base.uinttype(T)
    e_bits = ((Base.exponent_bias(T) + e) % U) << Base.significand_bits(T)
    result = reinterpret(T, e_bits | _rand_sign_mantissa(rng, T))
    # Subnormal numbers are intentionally not generated.
    return ifelse(e < exponent(floatmin(T)), zero(T), result)
end


@inline function _rand_mf(
    rng::AbstractRNG,
    leading_limb::T,
    ::Val{N},
) where {T,N}
    _iota = ntuple(identity, Val{N - 1}())
    padding = ntuple(_ -> leading_zeros(rand(rng, UInt64)), Val{N - 1}())
    exponents = cumsum(padding) .+ (precision(T) + 1) .* _iota
    return _MF{T,N}((leading_limb, _rand_trailing_limb.(
        Ref(rng), T, unsafe_exponent(leading_limb) .- exponents)...))
end


@inline function rand(
    rng::AbstractRNG,
    ::SamplerTrivial{CloseOpen01{_MF{T,N}}},
) where {T,N}
    leading_limb = _rand_leading_limb(
        rng, T, -leading_zeros(rand(rng, UInt64)) - 1)
    return _rand_mf(rng, leading_limb, Val{N}())
end


@inline function randn(
    rng::AbstractRNG,
    ::Type{_MF{T,N}},
) where {T,N}
    leading_limb = randn(rng, T)
    iszero(leading_limb) && return zero(_MF{T,N})
    return _rand_mf(rng, leading_limb, Val{N}())
end


@inline function randexp(
    rng::AbstractRNG,
    ::Type{_MF{T,N}},
) where {T,N}
    leading_limb = randexp(rng, T)
    return _rand_mf(rng, leading_limb, Val{N}())
end
