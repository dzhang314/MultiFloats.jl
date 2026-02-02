push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using MultiFloats


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
        x = MultiFloats.renormalize(ntuple(_ -> _bit_rand(T), Val{N}()))
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
            if !_cmp_zero(c._limbs, MultiFloats._renorm_pass(c._limbs))
                r = MultiFloat{T,N}(MultiFloats.renormalize(c._limbs))
                println("RENORM ERROR:")
                println("    a = ", a)
                println("    b = ", b)
                println("    c = ", c)
                println("    r = ", r)
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
