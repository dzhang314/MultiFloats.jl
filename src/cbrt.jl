# NOTE: MultiFloats.rcbrt is not exported to avoid name conflicts.
# Users are expected to call it as MultiFloats.rcbrt(x).

@inline rcbrt(x::Any) = inv(cbrt(x))


const _ONE_THIRD_FULL_F32 = (
    Float32(+0x1.555556p-002), Float32(-0x1.555556p-027),
    Float32(+0x1.555556p-052), Float32(-0x1.555556p-077),
    Float32(+0x1.555556p-102), Float32(-0x0.AAAAAAp-126),
)

const _ONE_THIRD_FULL_F64 = (
    +0x1.5555555555555p-0002, +0x1.5555555555555p-0056,
    +0x1.5555555555555p-0110, +0x1.5555555555555p-0164,
    +0x1.5555555555555p-0218, +0x1.5555555555555p-0272,
    +0x1.5555555555555p-0326, +0x1.5555555555555p-0380,
    +0x1.5555555555555p-0434, +0x1.5555555555555p-0488,
    +0x1.5555555555555p-0542, +0x1.5555555555555p-0596,
    +0x1.5555555555555p-0650, +0x1.5555555555555p-0704,
    +0x1.5555555555555p-0758, +0x1.5555555555555p-0812,
    +0x1.5555555555555p-0866, +0x1.5555555555555p-0920,
    +0x1.5555555555555p-0974, +0x0.0555555555555p-1022,
)

@inline _one_third_full(::Type{Float32}) = _ONE_THIRD_FULL_F32
@inline _one_third_full(::Type{Float64}) = _ONE_THIRD_FULL_F64


@inline function _mfrcbrt_impl(
    x::NTuple{X,T},
    u::NTuple{U,T},
    ::Val{Z},
) where {T,X,U,Z}
    @assert 0 < U < Z
    _zero = zero(T)
    _one = one(T)
    if U + U >= Z
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{Z}())
        _one_third = _resize(_one_third_full(T), Val{Z}())
        rx = _resize(x, Val{Z}())
        ru = _resize(u, Val{Z}())
        u2 = mfsqr(ru, Val{Z}())
        u3 = mfmul(u2, ru, Val{Z}())
        residual = mfadd(mfmul(rx, u3, Val{Z}()), _neg_one, Val{Z}())
        u_over_3 = mfmul(_one_third, ru, Val{Z}())
        correction = mfmul(residual, u_over_3, Val{Z}())
        return mfadd(ru, (-).(correction), Val{Z}())
    else
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{U + U}())
        _one_third = _resize(_one_third_full(T), Val{U + U}())
        rx = _resize(x, Val{U + U}())
        ru = _resize(u, Val{U + U}())
        u2 = mfsqr(ru, Val{U + U}())
        u3 = mfmul(u2, ru, Val{U + U}())
        residual = mfadd(mfmul(rx, u3, Val{U + U}()), _neg_one, Val{U + U}())
        u_over_3 = mfmul(_one_third, ru, Val{U + U}())
        correction = mfmul(residual, u_over_3, Val{U + U}())
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
    if U + U >= Z
        _one_third = _resize(_one_third_full(T), Val{Z}())
        rx = _resize(x, Val{Z}())
        ru = _resize(u, Val{Z}())
        u2 = mfsqr(ru, Val{Z}())
        root = mfmul(rx, u2, Val{Z}())
        root3 = mfmul(mfsqr(root, Val{Z}()), root, Val{Z}())
        residual = mfadd(root3, (-).(rx), Val{Z}())
        u2_over_3 = mfmul(_one_third, u2, Val{Z}())
        correction = mfmul(residual, u2_over_3, Val{Z}())
        return mfadd(root, (-).(correction), Val{Z}())
    else
        _neg_one = ntuple(i -> (isone(i) ? -_one : _zero), Val{U + U}())
        _one_third = _resize(_one_third_full(T), Val{U + U}())
        rx = _resize(x, Val{U + U}())
        ru = _resize(u, Val{U + U}())
        u2 = mfsqr(ru, Val{U + U}())
        u3 = mfmul(u2, ru, Val{U + U}())
        residual = mfadd(mfmul(rx, u3, Val{U + U}()), _neg_one, Val{U + U}())
        u_over_3 = mfmul(_one_third, ru, Val{U + U}())
        correction = mfmul(residual, u_over_3, Val{U + U}())
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


@inline Base.cbrt(x::_MF{T,N}) where {T,N} =
    ifelse(iszero(x), x, _MF{T,N}(mfcbrt(x._limbs, Val{N}())))
@inline Base.cbrt(x::_MFV{M,T,N}) where {M,T,N} =
    vifelse(iszero(x), x, _MFV{M,T,N}(mfcbrt(x._limbs, Val{N}())))
