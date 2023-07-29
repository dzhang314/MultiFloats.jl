baremodule Arithmetic

#=

export one_pass_renorm_func, two_pass_renorm_func,
    multifloat_add_func, multifloat_float_add_func,
    multifloat_mul_func, multifloat_float_mul_func,
    multifloat_div_func, multifloat_sqrt_func,
    multifloat_eq_func, multifloat_ne_func, multifloat_lt_func,
    multifloat_gt_func, multifloat_le_func, multifloat_ge_func

using Base: +, -, *, div, !, (:), ==, !=, <, <=, >, >=, min, max,
    Vector, isempty, length, push!, deleteat!, reverse!, Dict, haskey, @assert

###################################################### METAPROGRAMMING UTILITIES

function_def(name, args, body) =
    Expr(:function,
        Expr(:where, Expr(:call, name, args...), :T),
        Expr(:block, body...))

function_def_typed(name, arg_type, args, body) =
    function_def(name, [Expr(:(::), arg, arg_type) for arg in args], body)

meta_multifloat(N::Int) = :(MultiFloat{T,$N})

########################################################### ARITHMETIC FUNCTIONS

function multifloat_div_func(N::Int; sloppy::Bool=false)
    code = inline_block()
    quots = [Symbol('q', i) for i = 0:N-sloppy]
    push!(code, :($(quots[1]) = a._limbs[1] / b._limbs[1]))
    push!(code, :(r = a - b * $(quots[1])))
    for i = 2:N-sloppy
        push!(code, :($(quots[i]) = r._limbs[1] / b._limbs[1]))
        push!(code, :(r -= b * $(quots[i])))
    end
    push!(code, :($(quots[N+1-sloppy]) = r._limbs[1] / b._limbs[1]))
    push!(code, Expr(:call, meta_multifloat(N),
        Expr(:call, renorm_name(N), quots...)))
    function_def_typed(:multifloat_div, meta_multifloat(N), [:a, :b], code)
end

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

########################################################### RECURSIVE COMPARISON

eq_expr(n::Int) = (
    n == 1
    ? :(x._limbs[$n] == y._limbs[$n])
    : :($(eq_expr(n - 1)) & (x._limbs[$n] == y._limbs[$n]))
)

ne_expr(n::Int) = (
    n == 1
    ? :(x._limbs[$n] != y._limbs[$n])
    : :($(ne_expr(n - 1)) | (x._limbs[$n] != y._limbs[$n]))
)

lt_expr(m::Int, n::Int) = (
    m == n
    ? :(x._limbs[$m] < y._limbs[$m])
    : :(
        (x._limbs[$m] < y._limbs[$m]) |
        ((x._limbs[$m] == y._limbs[$m]) & $(lt_expr(m + 1, n)))
    )
)

gt_expr(m::Int, n::Int) = (
    m == n
    ? :(x._limbs[$m] > y._limbs[$m])
    : :(
        (x._limbs[$m] > y._limbs[$m]) |
        ((x._limbs[$m] == y._limbs[$m]) & $(gt_expr(m + 1, n)))
    )
)

le_expr(m::Int, n::Int) = (
    m == n
    ? :(x._limbs[$m] <= y._limbs[$m])
    : :(
        (x._limbs[$m] < y._limbs[$m]) |
        ((x._limbs[$m] == y._limbs[$m]) & $(le_expr(m + 1, n)))
    )
)

ge_expr(m::Int, n::Int) = (
    m == n
    ? :(x._limbs[$m] >= y._limbs[$m])
    : :(
        (x._limbs[$m] > y._limbs[$m]) |
        ((x._limbs[$m] == y._limbs[$m]) & $(ge_expr(m + 1, n)))
    )
)

########################################################### COMPARISON FUNCTIONS

cmp_func(name, N, expr) = function_def_typed(name,
    meta_multifloat(N), [:x, :y], push!(inline_block(), expr))

multifloat_eq_func(N::Int) = cmp_func(:multifloat_eq, N, eq_expr(N))
multifloat_ne_func(N::Int) = cmp_func(:multifloat_ne, N, ne_expr(N))
multifloat_lt_func(N::Int) = cmp_func(:multifloat_lt, N, lt_expr(1, N))
multifloat_gt_func(N::Int) = cmp_func(:multifloat_gt, N, gt_expr(1, N))
multifloat_le_func(N::Int) = cmp_func(:multifloat_le, N, le_expr(1, N))
multifloat_ge_func(N::Int) = cmp_func(:multifloat_ge, N, ge_expr(1, N))

=#

end # baremodule Arithmetic
