baremodule Arithmetic

#=

num_sqrt_iters(N::Int, sloppy::Bool) =
    (2 <= N <= 3) ? 2 :
    (4 <= N <= 5) ? 3 :
    sloppy ? div(N + 1, 2) : div(N + 2, 2)

function multifloat_sqrt_func(N::Int; sloppy::Bool=false)
    code = inline_block()
    push!(code, :(r = MultiFloat{T,$N}(inv(unsafe_sqrt(x._limbs[1])))))
    push!(code, :(h = scale(T(0.5), x)))
    for _ = 1:num_sqrt_iters(N, sloppy)
        push!(code, :(r += r * (T(0.5) - h * (r * r))))
    end
    push!(code, :(r * x))
    function_def_typed(:multifloat_sqrt, meta_multifloat(N), [:x], code)
end

=#

end # baremodule Arithmetic
