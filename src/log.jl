_log(x, ::Val{ℯ}) = log(x)
_log(x, ::Val{2}) = log2(x)
_log(x, ::Val{10}) = log10(x)

# 1 newton iteration
function log_impl(x::_MF{T,2}, base) where {T}
    y0 = _MF{T,2}(_log(first(x._limbs), base))
    t = exp_impl(-y0, base)
    return y0 + (x*t - 1)
end

# 1 halley iteration
function log_impl(x::_MF{T,3}, base) where {T}
    y0 = _MF{T,3}(_log(first(x._limbs), base))
    t = exp_impl(y0, base)
    return y0 + scale(2, (x-t)/(x+t))
end

# 2 newton iterations
function log_impl(x::_MF{T,4}, base) where {T}
    y0 = _MF{T,2}(_log(first(x._limbs), base))
    t0 = exp_impl(-y0, base)
    y1 = _MF{T,4}(y0 + (x*t0 - 1))
    t1 = exp_impl(-y1, base)
    return y1 + (x*t1 - 1)
end

Base.log(x::_MF) = log_impl(x, Val(ℯ))
Base.log2(x::_MF) = log_impl(x, Val(2))
Base.log10(x::_MF) = log_impl(x, Val(10))
