baremodule Arithmetic

export one_pass_renorm_func, two_pass_renorm_func,
    multifloat_add_func, multifloat_float_add_func,
    multifloat_mul_func, multifloat_float_mul_func,
    multifloat_div_func, multifloat_sqrt_func,
    multifloat_eq_func, multifloat_ne_func, multifloat_lt_func,
    multifloat_gt_func, multifloat_le_func, multifloat_ge_func

using Base: +, -, *, div, !, (:), ==, !=, <, <=, >, >=, min, max,
    Vector, isempty, length, push!, deleteat!, reverse!, Dict, haskey, @assert

###################################################### METAPROGRAMMING UTILITIES

const SymExpr = Union{Symbol,Expr}

meta_tuple(x...) = Expr(:tuple, x...)

inline_block() = [Expr(:meta, :inline)] # same as @inline

function_def(name, args, body) =
    Expr(:function,
        Expr(:where, Expr(:call, name, args...), :T),
        Expr(:block, body...))

function_def_typed(name, arg_type, args, body) =
    function_def(name, [Expr(:(::), arg, arg_type) for arg in args], body)

meta_two_sum(s::Symbol, e::Symbol, a::SymExpr, b::SymExpr) =
    Expr(:(=), meta_tuple(s, e), Expr(:call, :two_sum, a, b))

meta_two_prod(p::Symbol, e::Symbol, a::SymExpr, b::SymExpr) =
    Expr(:(=), meta_tuple(p, e), Expr(:call, :two_prod, a, b))

meta_quick_two_sum(s::Symbol, e::Symbol, a::Symbol, b::Symbol) =
    Expr(:(=), meta_tuple(s, e), Expr(:call, :quick_two_sum, a, b))

meta_sum(addends::Vector{Symbol}) =
    length(addends) == 1 ? addends[1] : Expr(:call, :+, addends...)

meta_sum(result::Symbol, addends::Vector{Symbol}) =
    Expr(:(=), result, meta_sum(addends))

meta_multifloat(N::Int) = :(MultiFloat{T,$N})

renorm_name(n::Int) = Symbol("renorm_", n)

mpadd_name(src_len::Int, dst_len::Int) = Symbol("mpadd_", src_len, '_', dst_len)

################################################################ RENORMALIZATION

function one_pass_renorm_func(n::Int; sloppy::Bool=false)
    args = [Symbol('a', i) for i = 0:n-sloppy]
    sums = [Symbol('s', i) for i = 0:n-1]
    if (n == 2) && sloppy
        return function_def_typed(renorm_name(2), :T, args,
            [Expr(:call, :quick_two_sum, args...)])
    end
    code = inline_block()
    push!(code, meta_quick_two_sum(sums[1], sums[2], args[1], args[2]))
    for i = 1:n-2
        push!(code, meta_quick_two_sum(
            sums[i+1], sums[i+2], sums[i+1], args[i+2]))
    end
    push!(code, sloppy ? meta_tuple(sums...) :
        meta_tuple(sums[1:n-1]..., meta_sum([sums[n], args[n+1]])))
    function_def_typed(renorm_name(n), :T, args, code)
end

function two_pass_renorm_func(n::Int; sloppy::Bool=false)
    args = [Symbol('a', i) for i = 0:n-sloppy]
    sums = [Symbol('s', i) for i = 0:n-1]
    if (n == 2) && sloppy
        return function_def_typed(renorm_name(2), :T, args,
            [Expr(:call, :quick_two_sum, args...)])
    end
    code = inline_block()
    push!(code, meta_quick_two_sum(:t, args[end], args[end-1], args[end]))
    for i = n-2-sloppy:-1:1
        push!(code, meta_quick_two_sum(:t, args[i+2], args[i+1], :t))
    end
    push!(code, meta_quick_two_sum(sums[1], sums[2], args[1], :t))
    for i = 1:n-2
        push!(code, meta_quick_two_sum(
            sums[i+1], sums[i+2], sums[i+1], args[i+2]))
    end
    push!(code, sloppy ? meta_tuple(sums...) :
        meta_tuple(sums[1:n-1]..., meta_sum([sums[n], args[n+1]])))
    function_def_typed(renorm_name(n), :T, args, code)
end

################################################################### ACCUMULATION

function add_var!(vars::Vector{Tuple{Symbol,Int}}, prefix::Char, i::Int)
    var = Symbol(prefix, i)
    push!(vars, (var, i))
    var
end

function add_var!(vars::Vector{Tuple{Symbol,Int}}, prefix::Char, i::Int, j::Int)
    var = Symbol(prefix, i, '_', j)
    push!(vars, (var, j))
    var
end

function mpadd_func(src_len::Int, dst_len::Int)
    @assert src_len >= dst_len
    args = [Symbol('a', i) for i = 1:src_len]
    sums = [Symbol('s', i) for i = 0:dst_len-1]
    vars = [(a, 0) for a in args]
    code = inline_block()
    k = 0
    for i = 0:dst_len-2
        while true
            addends = [v[1] for v in vars if v[2] == i]
            if length(addends) == 1
                push!(code, Expr(:(=), sums[i+1], addends[1]))
                deleteat!(vars, [v[1] == addends[1] for v in vars])
                break
            end
            for j = 1:div(length(addends), 2)
                a, b = addends[2*j-1], addends[2*j]
                push!(code, meta_two_sum(add_var!(vars, 'm', k, i),
                    add_var!(vars, 'm', k + 1, i + 1), a, b))
                k += 2
                deleteat!(vars, [v[1] == a || v[1] == b for v in vars])
            end
        end
    end
    push!(code, meta_sum(sums[dst_len], [v[1] for v in vars]))
    push!(code, meta_tuple(sums...))
    function_def_typed(mpadd_name(src_len, dst_len), :T, args, code)
end

const MPADD_CACHE = Dict{Symbol,Expr}()

function _mpsum(results::Vector{Symbol}, addends::Vector{Symbol})
    src_len = length(addends)
    dst_len = length(results)
    @assert src_len >= dst_len
    if dst_len == 1
        meta_sum(results[1], addends)
    elseif src_len == dst_len == 2
        meta_two_sum(results[1], results[2], addends[1], addends[2])
    else
        func_name = mpadd_name(src_len, dst_len)
        if !haskey(MPADD_CACHE, func_name)
            MPADD_CACHE[func_name] = mpadd_func(src_len, dst_len)
        end
        Expr(:(=), meta_tuple(results...), Expr(:call, func_name, addends...))
    end
end

function generate_accumulation_code!(code::Vector{Expr},
        vars::Vector{Tuple{Symbol,Int}}, N::Int; sloppy::Bool=false)
    sums = [Symbol('s', i) for i = 0 : N - sloppy]
    for i = 0 : N - sloppy
        addends = [v[1] for v in vars if v[2] == i]
        results = [sums[i+1]]
        for j = i+1:min(i + length(addends) - 1, N - sloppy)
            push!(results, add_var!(vars, 'm', i, j))
        end
        push!(code, _mpsum(results, addends))
        deleteat!(vars, [v[2] <= i for v in vars])
    end
    push!(code, Expr(:call, meta_multifloat(N),
        Expr(:call, renorm_name(N), sums...)))
end

########################################################### ARITHMETIC FUNCTIONS

function multifloat_add_func(N::Int; sloppy::Bool=false)
    code = inline_block()
    vars = Tuple{Symbol,Int}[]
    for i = 1:N-sloppy
        tmp = add_var!(vars, 't', i - 1)
        err = add_var!(vars, 'e', i)
        push!(code, meta_two_sum(tmp, err, :(a._limbs[$i]), :(b._limbs[$i])))
    end
    if sloppy
        tmp = add_var!(vars, 't', N - 1)
        push!(code, :($tmp = a._limbs[$N] + b._limbs[$N]))
    end
    reverse!(vars)
    generate_accumulation_code!(code, vars, N, sloppy=sloppy)
    function_def_typed(:multifloat_add, meta_multifloat(N), [:a, :b], code)
end

function multifloat_float_add_func(N::Int; sloppy::Bool=false)
    code = inline_block()
    vars = Tuple{Symbol,Int}[]
    push!(code, meta_two_sum(add_var!(vars, 't', 0), add_var!(vars, 'e', 1),
        :(a._limbs[1]), :b))
    for i = 2:N
        push!(code, Expr(:(=), add_var!(vars, 't', i - 1), :(a._limbs[$i])))
    end
    reverse!(vars)
    generate_accumulation_code!(code, vars, N, sloppy=sloppy)
    function_def(:multifloat_float_add,
        [Expr(:(::), :a, meta_multifloat(N)), Expr(:(::), :b, :T)], code)
end

function multifloat_mul_func(N::Int; sloppy::Bool=false)
    code = inline_block()
    for i = 0:N-1-sloppy, j = 0:i
        push!(code, meta_two_prod(
            Symbol('t', j, '_', i), Symbol('e', j, '_', i + 1),
            :(a._limbs[$(j + 1)]), :(b._limbs[$(i - j + 1)])))
    end
    for i = 0:N-2+sloppy
        push!(code, Expr(:(=), Symbol('t', i, '_', N - sloppy),
            :(a._limbs[$(i + 2 - sloppy)] * b._limbs[$(N - i)])))
    end
    vars = Tuple{Symbol,Int}[]
    for i = 0:N-1-sloppy, j = 0:i
        add_var!(vars, 't', j, i)
    end
    for i = 0:N-2+sloppy
        add_var!(vars, 't', i, N - sloppy)
    end
    for i = 0:N-1-sloppy, j = 0:i
        add_var!(vars, 'e', j, i + 1)
    end
    generate_accumulation_code!(code, vars, N, sloppy=sloppy)
    function_def_typed(:multifloat_mul, meta_multifloat(N), [:a, :b], code)
end

function multifloat_float_mul_func(N::Int; sloppy::Bool=false)
    code = inline_block()
    for i = 0 : N-1-sloppy
        push!(code, meta_two_prod(Symbol('t', i), Symbol('e', i+1),
            :(a._limbs[$(i+1)]), :b))
    end
    if sloppy
        push!(code, Expr(:(=), Symbol('t', N - 1), :(a._limbs[$N] * b)))
    end
    vars = Tuple{Symbol,Int}[]
    for i = 0:N-1
        add_var!(vars, 't', i)
    end
    for i = 1:N-sloppy
        add_var!(vars, 'e', i)
    end
    generate_accumulation_code!(code, vars, N, sloppy=sloppy)
    function_def(:multifloat_float_mul,
        [Expr(:(::), :a, meta_multifloat(N)), Expr(:(::), :b, :T)], code)
end

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

eq_expr(n::Int) = (n == 1
    ? :(x._limbs[$n] == y._limbs[$n])
    : :($(eq_expr(n-1)) & (x._limbs[$n] == y._limbs[$n])))

ne_expr(n::Int) = (n == 1
    ? :(x._limbs[$n] != y._limbs[$n])
    : :($(ne_expr(n-1)) | (x._limbs[$n] != y._limbs[$n])))

lt_expr(m::Int, n::Int) = (m == n
    ? :(x._limbs[$m] < y._limbs[$m])
    : :((x._limbs[$m] < y._limbs[$m]) |
        ((x._limbs[$m] == y._limbs[$m]) & $(lt_expr(m+1, n)))))

gt_expr(m::Int, n::Int) = (m == n
    ? :(x._limbs[$m] > y._limbs[$m])
    : :((x._limbs[$m] > y._limbs[$m]) |
        ((x._limbs[$m] == y._limbs[$m]) & $(gt_expr(m+1, n)))))

le_expr(m::Int, n::Int) = (m == n
    ? :(x._limbs[$m] <= y._limbs[$m])
    : :((x._limbs[$m] < y._limbs[$m]) |
        ((x._limbs[$m] == y._limbs[$m]) & $(le_expr(m+1, n)))))

ge_expr(m::Int, n::Int) = (m == n
    ? :(x._limbs[$m] >= y._limbs[$m])
    : :((x._limbs[$m] > y._limbs[$m]) |
        ((x._limbs[$m] == y._limbs[$m]) & $(ge_expr(m+1, n)))))

########################################################### COMPARISON FUNCTIONS

cmp_func(name, N, expr) = function_def_typed(name,
    meta_multifloat(N), [:x, :y], push!(inline_block(), expr))

multifloat_eq_func(N::Int) = cmp_func(:multifloat_eq, N, eq_expr(N))
multifloat_ne_func(N::Int) = cmp_func(:multifloat_ne, N, ne_expr(N))
multifloat_lt_func(N::Int) = cmp_func(:multifloat_lt, N, lt_expr(1, N))
multifloat_gt_func(N::Int) = cmp_func(:multifloat_gt, N, gt_expr(1, N))
multifloat_le_func(N::Int) = cmp_func(:multifloat_le, N, le_expr(1, N))
multifloat_ge_func(N::Int) = cmp_func(:multifloat_ge, N, ge_expr(1, N))

end # baremodule Arithmetic
