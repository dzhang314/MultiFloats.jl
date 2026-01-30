push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using MultiFloats


@generated function _renorm_pass(x::NTuple{N,T}) where {N,T}
    xs = [Symbol('x', i) for i in Base.OneTo(N)]
    body = Expr[]
    push!(body, Expr(:meta, :inline))
    push!(body, Expr(:(=), Expr(:tuple, xs...), :x))
    for i = 1:2:N-1
        push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[i+1]),
            Expr(:call, MultiFloats.two_sum, xs[i], xs[i+1])))
    end
    for i = 2:2:N-1
        push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[i+1]),
            Expr(:call, MultiFloats.two_sum, xs[i], xs[i+1])))
    end
    push!(body, Expr(:return, Expr(:tuple, xs...)))
    return Expr(:block, body...)
end


@inline function _renormalize(x::NTuple{N,T}) where {N,T}
    while true
        x_next = _renorm_pass(x)
        if x_next === x
            return x
        end
        x = x_next
    end
end


@inline function _bit_rand(::Type{T}) where {T}
    while true
        x = reinterpret(T, rand(Base.uinttype(T)))
        if isfinite(x)
            return x
        end
    end
end


@inline function _bit_rand(::Type{MultiFloat{T,N}}) where {T,N}
    while true
        x = _renormalize(ntuple(_ -> _bit_rand(T), Val{N}()))
        if all(isfinite.(x))
            return MultiFloat{T,N}(x)
        end
    end
end


@inline _bit_rand(::Type{MultiFloatVec{M,T,N}}) where {M,T,N} =
    MultiFloatVec{M,T,N}(ntuple(_ -> _bit_rand(MultiFloat{T,N}), Val{M}()))


@inline _cmp_zero(a::T, b::T) where {T<:Base.IEEEFloat} =
    (a === b) | (iszero(a) & iszero(b))
@inline _cmp_zero(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T} =
    all(ntuple(i -> _cmp_zero(a[i], b[i]), Val{N}()))


function run_experiment(::Type{T}, ::Val{N}, op::F, ulps::Int) where {T,N,F}
    setprecision(BigFloat, MultiFloats._full_precision(T))
    threshold = nextfloat(zero(T), ulps)
    max_err = zero(BigFloat)
    while true
        a = _bit_rand(MultiFloat{T,N})
        b = _bit_rand(MultiFloat{T,N})
        exact = op(BigFloat(a), BigFloat(b))
        if abs(exact) <= floatmax(T)
            c = op(a, b)
            if !_cmp_zero(c._limbs, _renorm_pass(c._limbs))
                println("RENORM ERROR:")
                println("    a = ", a)
                println("    b = ", b)
                println("    c = ", c)
                println("    r = ", MultiFloat{T,N}(_renormalize(c._limbs)))
                exit(1)
            end
            abs_err = BigFloat(c) - exact
            if abs(abs_err) >= threshold
                rel_err = abs(abs_err / exact)
                if rel_err > max_err
                    println("New best: ", -log2(Float64(rel_err)))
                    println("    a = ", a)
                    println("    b = ", b)
                    flush(stdout)
                    max_err = rel_err
                end
            end
        end
    end
end


run_experiment(
    eval(Symbol(ARGS[1])),
    Val{parse(Int, ARGS[2])}(),
    eval(Symbol(ARGS[3])),
    parse(Int, ARGS[4]),
)