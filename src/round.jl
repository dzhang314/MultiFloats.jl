@inline _clear_signed_zeros(x::NTuple{N,T}) where {N,T} =
    map(limb -> limb + zero(T), x)
@inline _clear_signed_zeros(x::NTuple{N,Vec{M,T}}) where {N,M,T} =
    map(limb -> limb + zero(T), x)
@inline _clear_signed_zeros(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(_clear_signed_zeros(x._limbs))
@inline _clear_signed_zeros(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(_clear_signed_zeros(x._limbs))


@inline _fast_sweep_down(x::NTuple{1,T}, y::T) where {T} = (x[1] + y,)
@inline function _fast_sweep_down(x::NTuple{N,T}, y::T) where {N,T}
    s, e = fast_two_sum(x[1], y)
    return (s, _fast_sweep_down(Base.tail(x), e)...)
end


@inline _dec_int(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(_fast_sweep_up(_fast_sweep_down(x._limbs, -one(T))))
@inline _dec_int(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(_fast_sweep_up(_fast_sweep_down(x._limbs, -one(Vec{M,T}))))
@inline _inc_int(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(_fast_sweep_up(_fast_sweep_down(x._limbs, +one(T))))
@inline _inc_int(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(_fast_sweep_up(_fast_sweep_down(x._limbs, +one(Vec{M,T}))))


@inline function Base.round(
    x::_MF{T,N}, ::RoundingMode{:ToZero},
) where {T,N}
    s = signbit(x)
    t = _MF{T,N}(_clear_signed_zeros(trunc.(x._limbs)))
    u = _dec_int(t)
    v = _inc_int(t)
    return ifelse((t > x) & !s, u, ifelse((t < x) & s, v, t))
end

@inline function Base.round(
    x::_MFV{M,T,N}, ::RoundingMode{:ToZero},
) where {M,T,N}
    s = signbit(x)
    t = _MFV{M,T,N}(_clear_signed_zeros(trunc.(x._limbs)))
    u = _dec_int(t)
    v = _inc_int(t)
    return vifelse((t > x) & !s, u, vifelse((t < x) & s, v, t))
end


@inline function Base.round(x::_MF{T,N}, ::RoundingMode{:Down}) where {T,N}
    t = round(x, RoundToZero)
    return ifelse((t != x) & signbit(x), _dec_int(t), t)
end

@inline function Base.round(x::_MFV{M,T,N}, ::RoundingMode{:Down}) where {M,T,N}
    t = round(x, RoundToZero)
    return vifelse((t != x) & signbit(x), _dec_int(t), t)
end


@inline function Base.round(x::_MF{T,N}, ::RoundingMode{:Up}) where {T,N}
    t = round(x, RoundToZero)
    return ifelse((t != x) & !signbit(x), _inc_int(t), t)
end

@inline function Base.round(x::_MFV{M,T,N}, ::RoundingMode{:Up}) where {M,T,N}
    t = round(x, RoundToZero)
    return vifelse((t != x) & !signbit(x), _inc_int(t), t)
end


@inline Base.round(x::_MF{T,N}, ::RoundingMode{:FromZero}) where {T,N} =
    ifelse(signbit(x), round(x, RoundDown), round(x, RoundUp))
@inline Base.round(x::_MFV{M,T,N}, ::RoundingMode{:FromZero}) where {M,T,N} =
    vifelse(signbit(x), round(x, RoundDown), round(x, RoundUp))


@inline function Base.round(
    x::_MF{T,N}, ::RoundingMode{:NearestTiesAway},
) where {T,N}
    _one = one(T)
    _two = _one + _one
    _half = inv(_two)
    s = signbit(x)
    t = trunc(x)
    mid = _MF{T,N}(_fast_sweep_up(_fast_sweep_down(
        t._limbs, ifelse(s, -_half, +_half))))
    away = ifelse(s, _dec_int(t), _inc_int(t))
    far = ((x > mid) & !s) | ((x < mid) & s)
    tie = (x == mid) & (mid != t)
    return ifelse(far | tie, away, t)
end

@inline function Base.round(
    x::_MFV{M,T,N}, ::RoundingMode{:NearestTiesAway},
) where {M,T,N}
    _one = one(Vec{M,T})
    _two = _one + _one
    _half = inv(_two)
    s = signbit(x)
    t = trunc(x)
    mid = _MFV{M,T,N}(_fast_sweep_up(_fast_sweep_down(
        t._limbs, vifelse(s, -_half, +_half))))
    away = vifelse(s, _dec_int(t), _inc_int(t))
    far = ((x > mid) & !s) | ((x < mid) & s)
    tie = (x == mid) & (mid != t)
    return vifelse(far | tie, away, t)
end


@inline function Base.round(
    x::_MF{T,N}, ::RoundingMode{:NearestTiesUp},
) where {T,N}
    _one = one(T)
    _two = _one + _one
    _half = inv(_two)
    s = signbit(x)
    t = trunc(x)
    mid = _MF{T,N}(_fast_sweep_up(_fast_sweep_down(
        t._limbs, ifelse(s, -_half, +_half))))
    away = ifelse(s, _dec_int(t), _inc_int(t))
    far = ((x > mid) & !s) | ((x < mid) & s)
    tie = (x == mid) & (mid != t)
    return ifelse(far | (tie & !s), away, t)
end

@inline function Base.round(
    x::_MFV{M,T,N}, ::RoundingMode{:NearestTiesUp},
) where {M,T,N}
    _one = one(Vec{M,T})
    _two = _one + _one
    _half = inv(_two)
    s = signbit(x)
    t = trunc(x)
    mid = _MFV{M,T,N}(_fast_sweep_up(_fast_sweep_down(
        t._limbs, vifelse(s, -_half, +_half))))
    away = vifelse(s, _dec_int(t), _inc_int(t))
    far = ((x > mid) & !s) | ((x < mid) & s)
    tie = (x == mid) & (mid != t)
    return vifelse(far | (tie & !s), away, t)
end


@inline function _isodd_float(x::T) where {T}
    _one = one(T)
    _two = _one + _one
    _half = inv(_two)
    half_x = _half * x
    return (trunc(half_x) != half_x)
end


@inline _isodd_mf(x::NTuple{1,T}) where {T} = _isodd_float(x[1])
@inline _isodd_mf(x::NTuple{N,T}) where {N,T} =
    xor(_isodd_mf(Base.front(x)), _isodd_float(x[N]))


@inline function Base.round(
    x::_MF{T,N}, ::RoundingMode{:Nearest},
) where {T,N}
    _one = one(T)
    _two = _one + _one
    _half = inv(_two)
    s = signbit(x)
    t = trunc(x)
    mid = _MF{T,N}(_fast_sweep_up(_fast_sweep_down(
        t._limbs, ifelse(s, -_half, +_half))))
    away = ifelse(s, _dec_int(t), _inc_int(t))
    far = ((x > mid) & !s) | ((x < mid) & s)
    tie = (x == mid) & (mid != t)
    return ifelse(far | (tie & _isodd_mf(t._limbs)), away, t)
end

@inline function Base.round(
    x::_MFV{M,T,N}, ::RoundingMode{:Nearest},
) where {M,T,N}
    _one = one(Vec{M,T})
    _two = _one + _one
    _half = inv(_two)
    s = signbit(x)
    t = trunc(x)
    mid = _MFV{M,T,N}(_fast_sweep_up(_fast_sweep_down(
        t._limbs, vifelse(s, -_half, +_half))))
    away = vifelse(s, _dec_int(t), _inc_int(t))
    far = ((x > mid) & !s) | ((x < mid) & s)
    tie = (x == mid) & (mid != t)
    return vifelse(far | (tie & _isodd_mf(t._limbs)), away, t)
end
