# NOTE: MultiFloats.rcbrt is not exported to avoid name conflicts.
# Users are expected to call it as MultiFloats.rcbrt(x).

@inline rcbrt(x::Any) = inv(cbrt(x))


@inline function _mfrcbrt_impl(
    x::NTuple{X,T},
    u::NTuple{U,T},
    ::Val{Z},
) where {T,X,U,Z}
    @assert 0 < U < Z
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _three = _two + _one
    _third = inv(_three)
    if U + U >= Z
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{Z}())
        rx = _resize(x, Val{Z}())
        ru = _resize(u, Val{Z}())
        u2 = mfsqr(ru, Val{Z}())
        u3 = mfmul(u2, ru, Val{Z}())
        residual = mfadd(mfmul(rx, u3, Val{Z}()), _neg_one, Val{Z}())
        correction = mfmul(residual, scale(_third, ru), Val{Z}())
        return mfadd(ru, (-).(correction), Val{Z}())
    else
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{U + U}())
        rx = _resize(x, Val{U + U}())
        ru = _resize(u, Val{U + U}())
        u2 = mfsqr(ru, Val{U + U}())
        u3 = mfmul(u2, ru, Val{U + U}())
        residual = mfadd(mfmul(rx, u3, Val{U + U}()), _neg_one, Val{U + U}())
        correction = mfmul(residual, scale(_third, ru), Val{U + U}())
        next_u = mfadd(ru, (-).(correction), Val{U + U}())
        return _mfrcbrt_impl(x, next_u, Val{Z}())
    end
end


@inline function _mfcbrt_impl(
    x::NTuple{X,T},
    u::NTuple{U,T},
    ::Val{Z},
) where {T,X,U,Z}
    @assert 0 < U < Z
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _three = _two + _one
    _third = inv(_three)
    if U + U >= Z
        rx = _resize(x, Val{Z}())
        ru = _resize(u, Val{Z}())
        u2 = mfsqr(ru, Val{Z}())
        root = mfmul(rx, u2, Val{Z}())
        root3 = mfmul(mfsqr(root, Val{Z}()), root, Val{Z}())
        residual = mfadd(root3, (-).(rx), Val{Z}())
        correction = mfmul(residual, scale(_third, u2), Val{Z}())
        return mfadd(root, (-).(correction), Val{Z}())
    else
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{U + U}())
        rx = _resize(x, Val{U + U}())
        ru = _resize(u, Val{U + U}())
        u2 = mfsqr(ru, Val{U + U}())
        u3 = mfmul(u2, ru, Val{U + U}())
        residual = mfadd(mfmul(rx, u3, Val{U + U}()), _neg_one, Val{U + U}())
        correction = mfmul(residual, scale(_third, ru), Val{U + U}())
        next_u = mfadd(ru, (-).(correction), Val{U + U}())
        return _mfcbrt_impl(x, next_u, Val{Z}())
    end
end


@inline mfrcbrt(x::NTuple{X,T}, ::Val{1}) where {T,X} =
    (inv(cbrt(first(x))),)
@inline mfrcbrt(x::NTuple{X,T}, ::Val{Z}) where {T,X,Z} =
    _mfrcbrt_impl(x, (rcbrt(first(x)),), Val{Z}())


@inline mfcbrt(x::NTuple{X,T}, ::Val{1}) where {T,X} =
    (cbrt(first(x)),)
@inline mfcbrt(x::NTuple{X,T}, ::Val{Z}) where {T,X,Z} =
    _mfcbrt_impl(x, (rcbrt(first(x)),), Val{Z}())


@inline rcbrt(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(mfrcbrt(x._limbs, Val{N}()))
@inline rcbrt(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(mfrcbrt(x._limbs, Val{N}()))


# TODO: Handle zero with ifelse/mask
@inline Base.cbrt(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(mfcbrt(x._limbs, Val{N}()))
@inline Base.cbrt(x::_MFV{M,T,N}) where {M,T,N} =
    _MFV{M,T,N}(mfcbrt(x._limbs, Val{N}()))
