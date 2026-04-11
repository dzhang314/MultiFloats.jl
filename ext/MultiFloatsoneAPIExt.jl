module MultiFloatsoneAPIExt

using oneAPI: @device_override, method_table
import MultiFloats: inv_r, div_r, sqrt_r


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


@device_override @inline function sqrt_r(x::Float32)
    s = Base.sqrt_llvm(x)
    h = 0.5f0 * inv_r(s)
    e = fma(-s, s, x)
    return fma(h, e, s)
end


@device_override @inline function sqrt_r(x::Float64)
    s = Base.sqrt_llvm(x)
    h = 0.5 * inv_r(s)
    e = fma(-s, s, x)
    return fma(h, e, s)
end


end # module MultiFloatsoneAPIExt
