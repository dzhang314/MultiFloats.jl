@generated function _sinpi_coefs(::_MF{T,N}) where {T, N}
    N_TERMS = T == Float64 ? 5+5N : 2+3N
    setprecision(BigFloat, precision(T)*N*2) do
        res = ntuple(n->_MF{T,N}(-(-1)^n*big(pi)^(2n-1)/factorial(big(2n-1))), N_TERMS)
        return :($res)
    end
end

@generated function _cospi_coefs(::_MF{T,N}) where {T, N}
    N_TERMS = T == Float64 ? 5+5N : 2+3N
    setprecision(BigFloat, precision(T)*N*2) do
        res = ntuple(n->_MF{T,N}(-(-1)^n*big(pi)^(2n-2)/factorial(big(2n-2))), N_TERMS)
        return :($res)
    end
end

function sinpi_kernel(x::_MF{T,N}) where {T, N}
    x*evalpoly(x*x, _sinpi_coefs(x))
end
function cospi_kernel(x::_MF{T,N}) where {T, N}
    evalpoly(x*x, _cospi_coefs(x))
end

function _shuffle_down(x::_MF{T,N}) where {T, N}
    return _MF{T,N}((x._limbs[2:end]..., zero(T)))
end

function Base.cospi(x::_MF{T,N}) where {T, N}
    !isfinite(x) && return _MF{T,N}(NaN)
    # For large x, we know the first limb is even and can thus be ignored
    x >= maxintfloat(T) && return cospi(_shuffle_down(x))

    x = abs(x)
    # reduce to interval [0, 0.5]
    first_limb = first(x._limbs)
    n = round(2*first_limb)
    rx = T(-.5)*n + x
    n = unsafe_trunc(Int64, n) & 3
    if n==0
        return cospi_kernel(rx)
    elseif n==1
        return -sinpi_kernel(rx)
    elseif n==2
        return -cospi_kernel(rx)
    else
        return sinpi_kernel(rx)
    end
end

function Base.sinpi(_x::_MF{T,N}) where {T, N}
    !isfinite(_x) && return _MF{T,N}(NaN)
    # For large x, we know the first limb is even and can thus be ignored
    _x >= maxintfloat() && return sinpi(_shuffle_down(_x))

    x = abs(_x)
    # reduce to interval [0, 0.5]
    first_limb = first(x._limbs)
    n = round(2*first_limb)
    rx = T(-.5)*n + x
    n = unsafe_trunc(Int64, n) & 3
    if n==0
        res = sinpi_kernel(rx)
    elseif n==1
        res = cospi_kernel(rx)
    elseif n==2
        res = -sinpi_kernel(rx)
    else
        res = -cospi_kernel(rx)
    end
    return ifelse(signbit(_x), -res, res)
end

function Base.sincospi(_x::_MF{T,N}) where {T, N}
    !isfinite(_x) && return _MF{T,N}(NaN)
    # For large x, we know the first limb is even and can thus be ignored
    _x >= maxintfloat(T) && return sincospi(_shuffle_down(_x))

    x = abs(_x)
    # reduce to interval [0, 0.5]
    first_limb = first(x._limbs)
    n = round(2*first_limb)
    rx = T(-.5)*n + x
    n = unsafe_trunc(Int64, n) & 3
    si, co = sinpi_kernel(rx),cospi_kernel(rx)
    if n==0
        si, co = si, co
    elseif n==1
        si, co  = co, -si
    elseif n==2
        si, co  = -si, -co
    else
        si, co  = -co, si
    end
    si = ifelse(signbit(_x), -si, si)
    return si, co
end
