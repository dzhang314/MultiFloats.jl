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


@inline isrenormalized(x) = (x === MultiFloats.renormalize(x))


for _ = 1:10000
    x = _bit_rand(Float32x4)
    s = MultiFloats._to_string(x)
    r = Float32x4(s)
    if r !== x
        println(x, " : ", isrenormalized(x))
        println(r, " : ", isrenormalized(r))
        println(s)
    end
end
