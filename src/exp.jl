# max, min, and subnormal arguments
@generated function MAX_EXP(::Val{base}, ::T) where {base,T}
    res = T(Base.exponent_bias(T)*log(base, big(2)) + log(base, 2 - exp2(big(-Base.significand_bits(T)))))
    return :($res)
end
@generated function MIN_EXP(::Val{base}, ::T) where {base,T}
    res = T(-(Base.exponent_bias(T)+Base.significand_bits(T)) * log(base, big(2)))
    return :($res)
end
SUBNORM_EXP(::Val{base}, T) where base = log(base, floatmin(T))

function Log2B(::Val{base}, ::Type{T}) where {base, T}
    log2(base)
end
@generated function EXP_REDUCTION_COEFS(::Val{base}, ::_MF{T,N}) where {base, T,N}
    setprecision(BigFloat, precision(T)*N+1) do
        res = (_MF{T,N+1}(log(big(base))), _MF{T,N+1}(log(big(2))))
        return :($res)
    end
end

function exp_kernel(r::_MF{T,N}, ::Val{TERMS}) where {T,N,TERMS}
    return evalpoly(r, ntuple(i->one(_MF{T,N})/factorial(i-1), Val(M)))
end

# b^x = 2^(x*log(b)/log(2)) = 2^n*exp(y)
# where n::Int32 = round(x*log(b)/log(2))
# y = x*log(b) - n*log(2), |y| <= log(2)/2
function exp_impl(x::_MF{T,N}, base) where {T,N}
    max_exp = Int32(Base.exponent_bias(T))
    head = first(x._limbs)
    head > MAX_EXP(base, head) && return _MF{T,N}(Inf)
    head < MIN_EXP(base, head) && return zero(_MF{T,N})
    n = round(Int32, head*Log2B(base, T))
    logb, log2 = EXP_REDUCTION_COEFS(base, x)
    r = _MF{T,N}(_MF{T,N+1}(x)*logb - n*log2)
    N_REDUCTIONS = 4+2N
    N_TERMS = 4+2N
    # ERROR = (log(2)*exp2(-N_REDUCTIONS))^N_TERMS/factorial(N_TERMS)
    r = scale(T(exp2(-N_REDUCTIONS)), r)
    small_part = exp_kernel(r, Val(N_TERMS))._limbs
    for _ in 1:N_REDUCTIONS
        small_part = mfsqr(small_part, Val(N))
    end
    return ldexp(_MF{T,N}(small_part), n)
end

Base.exp(x::_MF) = exp_impl(x, Val(ℯ))
Base.exp2(x::_MF) = exp_impl(x, Val(2))
Base.exp10(x::_MF) = exp_impl(x, Val(10))
