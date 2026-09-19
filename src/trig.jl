include("trig_tables.jl")

# Minimax kernels on the reduced interval x in [-1/4, 1/4] (t = x^2 in [0, 1/16]),
# evaluated with the branch-free multi-limb Horner scheme `_horner_expr_mf`. See
# scripts/gen_trig_tables.jl for the coefficient generation.
#   sinpi(x) = x * P(t),      P(0) = pi          (pi kept inside the polynomial)
#   cospi(x) = 1 + t * R(t),  R(0) = -pi^2/2     (leading 1 reconstructed exactly)
@generated _sinpi_polynomial(x::NTuple{N,T}) where {N,T} =
    _horner_expr_mf(_sinpi_coefficients(T, Val{N}()))
@generated _cospi_polynomial(x::NTuple{N,T}) where {N,T} =
    _horner_expr_mf(_cospi_coefficients(T, Val{N}()))

function sinpi_kernel(x::_MF{T,N}) where {T, N}
    t = abs2(x)
    return x * _MF{T,N}(_sinpi_polynomial(t._limbs))
end
function cospi_kernel(x::_MF{T,N}) where {T, N}
    t = abs2(x)
    return one(_MF{T,N}) + t * _MF{T,N}(_cospi_polynomial(t._limbs))
end

function _shuffle_down(x::_MF{T,N}) where {T, N}
    return _MF{T,N}((x._limbs[2:end]..., zero(T)))
end

function Base.cospi(x::_MF{T,N}) where {T, N}
    !isfinite(x) && return _MF{T,N}(NaN)
    x = abs(x)
    # For large x, we know the first limb is even and can thus be ignored
    x >= maxintfloat(T) && return cospi(_shuffle_down(x))

    # reduce to interval [-0.25, 0.25].
    # We do this in 2 passes because a lower limb can individually exceed 0.25 and push
    # the reduced argument out of range.
    first_limb = first(x._limbs)
    n = round(2*first_limb)
    rx = T(-.5)*n + x
    m = round(2*first(rx._limbs))
    rx = T(-.5)*m + rx
    n = (unsafe_trunc(Int64, n) + unsafe_trunc(Int64, m)) & 3
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
    x = abs(_x)
    
    # For large x, we know the first limb is even and can thus be ignored
    x >= maxintfloat(T) && return sinpi(_shuffle_down(_x))

    # reduce to interval [-0.25, 0.25].
    # We do this in 2 passes because a lower limb can individually exceed 0.25 and push
    # the reduced argument out of range.
    first_limb = first(x._limbs)
    n = round(2*first_limb)
    rx = T(-.5)*n + x
    m = round(2*first(rx._limbs))
    rx = T(-.5)*m + rx
    n = (unsafe_trunc(Int64, n) + unsafe_trunc(Int64, m)) & 3
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
    x = abs(_x)

    # For large x, we know the first limb is even and can thus be ignored
    x >= maxintfloat(T) && return sincospi(_shuffle_down(_x))
    
    # reduce to interval [-0.25, 0.25].
    # We do this in 2 passes because a lower limb can individually exceed 0.25 and push
    # the reduced argument out of range.
    first_limb = first(x._limbs)
    n = round(2*first_limb)
    rx = T(-.5)*n + x
    m = round(2*first(rx._limbs))
    rx = T(-.5)*m + rx
    n = (unsafe_trunc(Int64, n) + unsafe_trunc(Int64, m)) & 3
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
