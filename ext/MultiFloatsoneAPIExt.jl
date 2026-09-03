module MultiFloatsoneAPIExt

using oneAPI: @device_override, method_table
import MultiFloats: _MF, Float64x3, Float64x4,
    _mfinv_impl, inv_r, div_r, sqrt_r, mfsqrt


@inline function _quotient_is_subnormal(x::Float32, y::Float32)
    x_bits = reinterpret(UInt32, x) & 0x7FFFFFFF
    y_bits = reinterpret(UInt32, y) & 0x7FFFFFFF
    ex = x_bits >> 23
    ey = y_bits >> 23
    shifted_ex = ex + 0x0000007E
    return (!iszero(ex)) & (!iszero(ey)) & ((shifted_ex < ey) | (
        (shifted_ex == ey) & ((x_bits & 0x007FFFFF) <= (y_bits & 0x007FFFFF))))
end


@inline function _rounded_quotient(num::UInt64, den::UInt64)
    q, r = divrem(num, den) .% UInt32
    two_r = r + r
    d32 = den % UInt32
    return q + UInt32((two_r > d32) | ((two_r == d32) & isodd(q)))
end


@noinline function _div_subnormal(x::Float32, y::Float32)
    x_bits = reinterpret(UInt32, x) & 0x7FFFFFFF
    y_bits = reinterpret(UInt32, y) & 0x7FFFFFFF
    ex = x_bits >> 23
    ey = y_bits >> 23
    mx = UInt64(0x00800000 | (x_bits & 0x007FFFFF))
    my = UInt64(0x00800000 | (y_bits & 0x007FFFFF))
    shifted_ex = ex + 0x00000096
    q = zero(UInt32)
    if shifted_ex > ey
        q = _rounded_quotient(mx << (shifted_ex - (ey + 0x00000001)), my)
    elseif shifted_ex == ey
        q = _rounded_quotient(mx, my << 1)
    end
    return reinterpret(Float32, (UInt32(xor(signbit(x), signbit(y))) << 31) | q)
end


const DIV_SCALE_F32 = reinterpret(Float32, 0x5F800000)


@noinline function _div_slow_path(
    x::Float32,
    y::Float32,
    q::Float32,
    x_tiny::Bool,
    q_tiny::Bool,
)
    if q_tiny && _quotient_is_subnormal(x, y)
        return _div_subnormal(x, y)
    elseif x_tiny && (abs(y) < DIV_SCALE_F32)
        scaled_x = DIV_SCALE_F32 * x
        scaled_y = DIV_SCALE_F32 * y
        return fma(inv(scaled_y), fma(-q, scaled_y, scaled_x), q)
    end
    residual = fma(-q, y, x)
    return fma(inv(y), residual, q)
end


@device_override @inline function inv_r(x::Float32)
    _one = one(Float32)
    q = inv(x)
    return ((abs(x) >= reinterpret(Float32, 0x7E800000)) ?
            _div_subnormal(_one, x) : fma(q, fma(-x, q, _one), q))
end


@device_override @inline function div_r(x::Float32, y::Float32)
    q = x / y
    x_tiny = abs(x) < reinterpret(Float32, 0x0C800000)
    q_tiny = abs(q) < reinterpret(Float32, 0x01000000)
    return ((x_tiny | q_tiny) ? _div_slow_path(x, y, q, x_tiny, q_tiny) :
            fma(inv(y), fma(-q, y, x), q))
end


# The following definitions work around a context-dependent compiler bug.
# If the first call to `inv` is inlined, oneAPI.jl generates incorrect code.
@noinline _inv_seed(x::Float64) = inv(x)
@device_override @inline Base.inv(x::Float64x3) = Float64x3(
    _mfinv_impl(x._limbs, (_inv_seed(first(x._limbs)),), Val{3}()))
@device_override @inline Base.inv(x::Float64x4) = Float64x4(
    _mfinv_impl(x._limbs, (_inv_seed(first(x._limbs)),), Val{4}()))


const SQRT_THRESHOLD_F32 = reinterpret(Float32, 0x0C800000)
const SQRT_SCALE_UP_F32 = reinterpret(Float32, 0x4D800000)
const SQRT_SCALE_DOWN_F32 = reinterpret(Float32, 0x38800000)


@device_override @inline function sqrt_r(x::Float32)
    small = (x < SQRT_THRESHOLD_F32)
    x = ifelse(small, SQRT_SCALE_UP_F32 * x, x)
    s = Base.sqrt_llvm(x)
    h = 0.5f0 * inv_r(s)
    e = fma(-s, s, x)
    r = fma(h, e, s)
    return ifelse(iszero(x), x, ifelse(small, SQRT_SCALE_DOWN_F32 * r, r))
end


@device_override @inline sqrt_r(x::_MF{T,N}) where {T,N} =
    iszero(x) ? x : _MF{T,N}(mfsqrt(x._limbs, Val{N}()))


end # module MultiFloatsoneAPIExt
