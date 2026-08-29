module MultiFloatsoneAPIExt

using oneAPI: @device_override, method_table
import MultiFloats: _MF, inv_r, div_r, sqrt_r, mfsqrt


@device_override @inline function inv_r(x::Float32)
    y = inv(x)
    e = fma(-x, y, 1.0f0)
    return fma(y, e, y)
end


@device_override @inline function inv_r(x::Float64)
    y = inv(x)
    e = fma(-x, y, 1.0)
    return fma(y, e, y)
end


@device_override @inline function div_r(x::Float32, y::Float32)
    q = x / y
    r = fma(-q, y, x)
    return fma(inv_r(y), r, q)
end


@device_override @inline function div_r(x::Float64, y::Float64)
    q = x / y
    r = fma(-q, y, x)
    return fma(inv_r(y), r, q)
end


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


@device_override @inline function sqrt_r(x::_MF{T,N}) where {T,N}
    return iszero(x) ? x : _MF{T,N}(mfsqrt(x._limbs, Val{N}()))
end


end # module MultiFloatsoneAPIExt
