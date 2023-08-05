module MultiFloats

using SIMD: Vec
using SIMD.Intrinsics: extractelement


########################################################## ERROR-FREE ARITHMETIC


@inline function fast_two_sum(a::T, b::T) where {T}
    sum = a + b
    b_prime = sum - a
    b_err = b - b_prime
    return (sum, b_err)
end


@inline function fast_two_diff(a::T, b::T) where {T}
    diff = a - b
    b_prime = a - diff
    b_err = b_prime - b
    return (diff, b_err)
end


@inline function two_sum(a::T, b::T) where {T}
    sum = a + b
    a_prime = sum - b
    b_prime = sum - a_prime
    a_err = a - a_prime
    b_err = b - b_prime
    err = a_err + b_err
    return (sum, err)
end


@inline function two_diff(a::T, b::T) where {T}
    diff = a - b
    a_prime = diff + b
    b_prime = a_prime - diff
    a_err = a - a_prime
    b_err = b - b_prime
    err = a_err - b_err
    return (diff, err)
end


@inline function two_prod(a::T, b::T) where {T}
    prod = a * b
    err = fma(a, b, -prod)
    return (prod, err)
end


###################################################### METAPROGRAMMING UTILITIES


_inline_block() = [Expr(:meta, :inline)]


_meta_tuple(xs...) = Expr(:tuple, xs...)


_meta_unpack(lhs::Vector{Symbol}, rhs::Union{Symbol,Expr}) =
    Expr(:(=), _meta_tuple(lhs...), rhs)


_meta_fast_two_sum(s::Symbol, e::Symbol, a::Symbol, b::Symbol) =
    Expr(:(=), _meta_tuple(s, e), Expr(:call, :fast_two_sum, a, b))


_meta_two_sum(s::Symbol, e::Symbol, a::Symbol, b::Symbol) =
    Expr(:(=), _meta_tuple(s, e), Expr(:call, :two_sum, a, b))


_meta_two_diff(d::Symbol, e::Symbol, a::Symbol, b::Symbol) =
    Expr(:(=), _meta_tuple(d, e), Expr(:call, :two_diff, a, b))


_meta_prod(p::Symbol, a::Symbol, b::Symbol) =
    Expr(:(=), p, Expr(:call, :*, a, b))


_meta_two_prod(p::Symbol, e::Symbol, a::Symbol, b::Symbol) =
    Expr(:(=), _meta_tuple(p, e), Expr(:call, :two_prod, a, b))


function _meta_sum(T::DataType, xs::Vector{Symbol})
    if isempty(xs)
        return zero(T)
    elseif length(xs) == 1
        return only(xs)
    else
        xs = Vector{Union{Symbol,Expr}}(xs)
        while length(xs) > 1
            push!(xs, Expr(:call, :+, xs[1], xs[2]))
            deleteat!(xs, 1:2)
        end
        return only(xs)
    end
end


################################################# EXTENDED ERROR-FREE ARITHMETIC


function accurate_sum_expr(
    T::DataType,
    num_inputs::Int,
    num_outputs::Int
)
    @assert num_outputs > 0
    code = _inline_block()

    # Unpack argument tuple.
    args = [Symbol('x', i) for i = 1:num_inputs]
    push!(code, _meta_unpack(args, :xs))

    # Instantiate lists of terms of order 1, epsilon, epsilon^2, ...
    terms = [Symbol[] for _ = 1:num_outputs]

    # All input arguments are assumed to be on the same order of magnitude.
    # (If they are not, the error terms will automatically bubble down.)
    append!(terms[1], args)

    # Repeatedly call two_sum until only one term of each order remains.
    count = 0
    for j = 1:num_outputs-1
        curr_terms = terms[j]
        next_terms = terms[j+1]
        while length(curr_terms) > 1
            count += 1
            sum_term = Symbol('s', count)
            err_term = Symbol('e', count)
            push!(code, _meta_two_sum(
                sum_term, err_term, curr_terms[1], curr_terms[2]
            ))
            deleteat!(curr_terms, 1:2)
            push!(curr_terms, sum_term)
            push!(next_terms, err_term)
        end
    end

    # Return a tuple containing the final term of each order.
    push!(code, Expr(:return, _meta_tuple(_meta_sum.(T, terms)...)))
    return Expr(:block, code...)
end


@generated function accurate_sum(::Val{N}, xs::T...) where {T,N}
    return accurate_sum_expr(T, length(xs), N)
end


function one_pass_renorm_expr(
    T::DataType,
    num_inputs::Int,
    num_outputs::Int
)
    @assert num_outputs > 0
    @assert ((num_inputs == num_outputs) ||
             (num_inputs == num_outputs + 1))
    code = _inline_block()

    # Unpack argument tuple.
    args = [Symbol('x', i) for i = 1:num_inputs]
    push!(code, _meta_unpack(args, :xs))

    # Generate one-pass renormalization code.
    for i = 1:num_outputs-1
        push!(code, _meta_fast_two_sum(args[i], args[i+1], args[i], args[i+1]))
    end

    # Return a tuple of renormalized terms.
    push!(code, Expr(:return, _meta_tuple(
        args[1:num_outputs-1]...,
        _meta_sum(T, args[num_outputs:end])
    )))
    return Expr(:block, code...)
end


@generated function one_pass_renorm(::Val{N}, xs::T...) where {T,N}
    return one_pass_renorm_expr(T, length(xs), N)
end


function two_pass_renorm_expr(
    T::DataType,
    num_inputs::Int,
    num_outputs::Int
)
    @assert num_outputs > 0
    @assert ((num_inputs == num_outputs) ||
             (num_inputs == num_outputs + 1))
    code = _inline_block()

    # Unpack argument tuple.
    args = [Symbol('x', i) for i = 1:num_inputs]
    push!(code, _meta_unpack(args, :xs))

    # Generate two-pass renormalization code.
    for i = num_inputs-1:-1:2
        push!(code, _meta_fast_two_sum(args[i], args[i+1], args[i], args[i+1]))
    end
    for i = 1:num_outputs-1
        push!(code, _meta_fast_two_sum(args[i], args[i+1], args[i], args[i+1]))
    end

    # Return a tuple of renormalized terms.
    push!(code, Expr(:return, _meta_tuple(
        args[1:num_outputs-1]...,
        _meta_sum(T, args[num_outputs:end])
    )))
    return Expr(:block, code...)
end


@generated function two_pass_renorm(::Val{N}, xs::T...) where {T,N}
    return two_pass_renorm_expr(T, length(xs), N)
end


############################################################### TYPE DEFINITIONS


export MultiFloat, MultiFloatVec


struct MultiFloat{T,N} <: AbstractFloat
    _limbs::NTuple{N,T}
end


struct MultiFloatVec{M,T,N}
    _limbs::NTuple{N,Vec{M,T}}
end


################################################################### TYPE ALIASES


export Float16x, Float32x, Float64x,
    Float64x1, Float64x2, Float64x3, Float64x4,
    Float64x5, Float64x6, Float64x7, Float64x8,
    v1Float64x1, v1Float64x2, v1Float64x3, v1Float64x4,
    v1Float64x5, v1Float64x6, v1Float64x7, v1Float64x8,
    v2Float64x1, v2Float64x2, v2Float64x3, v2Float64x4,
    v2Float64x5, v2Float64x6, v2Float64x7, v2Float64x8,
    v4Float64x1, v4Float64x2, v4Float64x3, v4Float64x4,
    v4Float64x5, v4Float64x6, v4Float64x7, v4Float64x8,
    v8Float64x1, v8Float64x2, v8Float64x3, v8Float64x4,
    v8Float64x5, v8Float64x6, v8Float64x7, v8Float64x8


const Float16x{N} = MultiFloat{Float16,N}
const Float32x{N} = MultiFloat{Float32,N}
const Float64x{N} = MultiFloat{Float64,N}


const Float64x1 = MultiFloat{Float64,1}
const Float64x2 = MultiFloat{Float64,2}
const Float64x3 = MultiFloat{Float64,3}
const Float64x4 = MultiFloat{Float64,4}
const Float64x5 = MultiFloat{Float64,5}
const Float64x6 = MultiFloat{Float64,6}
const Float64x7 = MultiFloat{Float64,7}
const Float64x8 = MultiFloat{Float64,8}


const v1Float64x1 = MultiFloatVec{1,Float64,1}
const v1Float64x2 = MultiFloatVec{1,Float64,2}
const v1Float64x3 = MultiFloatVec{1,Float64,3}
const v1Float64x4 = MultiFloatVec{1,Float64,4}
const v1Float64x5 = MultiFloatVec{1,Float64,5}
const v1Float64x6 = MultiFloatVec{1,Float64,6}
const v1Float64x7 = MultiFloatVec{1,Float64,7}
const v1Float64x8 = MultiFloatVec{1,Float64,8}


const v2Float64x1 = MultiFloatVec{2,Float64,1}
const v2Float64x2 = MultiFloatVec{2,Float64,2}
const v2Float64x3 = MultiFloatVec{2,Float64,3}
const v2Float64x4 = MultiFloatVec{2,Float64,4}
const v2Float64x5 = MultiFloatVec{2,Float64,5}
const v2Float64x6 = MultiFloatVec{2,Float64,6}
const v2Float64x7 = MultiFloatVec{2,Float64,7}
const v2Float64x8 = MultiFloatVec{2,Float64,8}


const v4Float64x1 = MultiFloatVec{4,Float64,1}
const v4Float64x2 = MultiFloatVec{4,Float64,2}
const v4Float64x3 = MultiFloatVec{4,Float64,3}
const v4Float64x4 = MultiFloatVec{4,Float64,4}
const v4Float64x5 = MultiFloatVec{4,Float64,5}
const v4Float64x6 = MultiFloatVec{4,Float64,6}
const v4Float64x7 = MultiFloatVec{4,Float64,7}
const v4Float64x8 = MultiFloatVec{4,Float64,8}


const v8Float64x1 = MultiFloatVec{8,Float64,1}
const v8Float64x2 = MultiFloatVec{8,Float64,2}
const v8Float64x3 = MultiFloatVec{8,Float64,3}
const v8Float64x4 = MultiFloatVec{8,Float64,4}
const v8Float64x5 = MultiFloatVec{8,Float64,5}
const v8Float64x6 = MultiFloatVec{8,Float64,6}
const v8Float64x7 = MultiFloatVec{8,Float64,7}
const v8Float64x8 = MultiFloatVec{8,Float64,8}


############################################## CONSTRUCTION FROM PRIMITIVE TYPES


@inline function MultiFloat{T,N}(x::T) where {T,N}
    return MultiFloat{T,N}(
        ntuple(i -> ifelse(i == 1, x, zero(T)), Val{N}())
    )
end


@inline function MultiFloatVec{M,T,N}(x::Vec{M,T}) where {M,T,N}
    return MultiFloatVec{M,T,N}(
        ntuple(i -> ifelse(i == 1, x, zero(Vec{M,T})), Val{N}())
    )
end


@inline function MultiFloatVec{M,T,N}(x::T) where {M,T,N}
    return MultiFloatVec{M,T,N}(Vec{M,T}(x))
end


@inline function MultiFloatVec{M,T,N}(
    xs::NTuple{M,MultiFloat{T,N}}
) where {M,T,N}
    return MultiFloatVec{M,T,N}(ntuple(
        j -> Vec{M,T}(ntuple(i -> xs[i]._limbs[j], Val{M}())),
        Val{N}()
    ))
end


################################################################ VECTOR INDEXING


@inline function Base.getindex(x::MultiFloatVec{M,T,N}, i::Int) where {M,T,N}
    return MultiFloat{T,N}(ntuple(
        j -> extractelement(x._limbs[j].data, i - 1),
        Val{N}()
    ))
end


################################################################################


const _MF = MultiFloat
const _MFV = MultiFloatVec


@inline Base.zero(::Type{_MF{T,N}}) where {T,N} =
    _MF{T,N}(ntuple(_ -> zero(T), Val{N}()))
@inline Base.zero(::Type{_MFV{M,T,N}}) where {M,T,N} =
    _MFV{M,T,N}(ntuple(_ -> zero(Vec{M,T}), Val{N}()))
@inline Base.zero(::_MF{T,N}) where {T,N} = zero(_MF{T,N})
@inline Base.zero(::_MFV{M,T,N}) where {M,T,N} = zero(_MFV{M,T,N})
@inline Base.one(::Type{_MF{T,N}}) where {T,N} =
    _MF{T,N}(ntuple(i -> ifelse(i == 1, one(T), zero(T)), Val{N}()))
@inline Base.one(::Type{_MFV{M,T,N}}) where {M,T,N} =
    _MFV{M,T,N}(ntuple(i -> ifelse(i == 1, one(Vec{M,T}), zero(Vec{M,T})), Val{N}()))
@inline Base.one(::_MF{T,N}) where {T,N} = one(_MF{T,N})
@inline Base.one(::_MFV{M,T,N}) where {M,T,N} = one(_MFV{M,T,N})


################################################# MULTIFLOAT-SPECIFIC ARITHMETIC


export renormalize, scale


@inline function _ntuple_equal(x::NTuple{N,T}, y::NTuple{N,T}) where {N,T}
    return all(x .== y)
end


@inline function _ntuple_equal(
    x::NTuple{N,Vec{M,T}}, y::NTuple{N,Vec{M,T}}
) where {N,M,T}
    return all(all.(x .== y))
end


@inline function renormalize(xs::NTuple{N,T}) where {N,T}
    # total = +(xs...)
    # if !isfinite(total)
    #     return ntuple(_ -> total, Val{N}())
    # end
    while true
        xs_new = two_pass_renorm(Val{N}(), xs...)
        if _ntuple_equal(xs, xs_new)
            return xs
        else
            xs = xs_new
        end
    end
end


@inline renormalize(x::_MF{T,N}) where {T,N} =
    MultiFloat{T,N}(renormalize(x._limbs))


@inline renormalize(x::_MFV{M,T,N}) where {M,T,N} =
    MultiFloatVec{M,T,N}(renormalize(x._limbs))


@inline scale(a::T, x::_MF{T,N}) where {T,N} =
    MultiFloat{T,N}(ntuple(i -> a * x._limbs[i], Val{N}()))


@inline scale(a::T, x::_MFV{M,T,N}) where {M,T,N} =
    MultiFloatVec{M,T,N}(ntuple(i -> a * x._limbs[i], Val{N}()))


@inline scale(a::Vec{M,T}, x::_MF{T,N}) where {M,T,N} =
    MultiFloatVec{M,T,N}(ntuple(i -> a * x._limbs[i], Val{N}()))


@inline scale(a::Vec{M,T}, x::_MFV{M,T,N}) where {M,T,N} =
    MultiFloatVec{M,T,N}(ntuple(i -> a * x._limbs[i], Val{N}()))


########################################### ARITHMETIC METAPROGRAMMING UTILITIES


function _push_accumulation_code!(
    code::Vector{Expr},
    results::Vector{Symbol},
    terms::Vector{Vector{Symbol}}
)
    @assert length(terms) == length(results)
    count = 0
    for i = 1:length(terms)
        num_remain = length(terms) - i + 1
        num_spill = min(length(terms[i]), num_remain)
        lhs = [results[i]]
        for j = 2:num_spill
            count += 1
            push!(lhs, Symbol('t', count))
        end
        push!(code, _meta_unpack(lhs,
            Expr(:call, :accurate_sum, Val(num_spill), terms[i]...)
        ))
        for j = 2:num_spill
            push!(terms[i+j-1], lhs[j])
        end
    end
    return code
end


function _meta_multifloat(vec_width::Int, T::DataType, num_limbs::Int)
    if vec_width == -1
        return Expr(:curly, :MultiFloat, T, num_limbs)
    else
        return Expr(:curly, :MultiFloatVec, vec_width, T, num_limbs)
    end
end


function _meta_return_multifloat(
    vec_width::Int, T::DataType, num_limbs::Int,
    code::Vector{Expr}, terms::Vector{Vector{Symbol}}, renorm_func::Symbol
)
    results = [Symbol('x', i) for i = 1:length(terms)]
    _push_accumulation_code!(code, results, terms)
    push!(code, Expr(:return, Expr(:call,
        _meta_multifloat(vec_width, T, num_limbs),
        Expr(:call, renorm_func, Val(num_limbs), results...)
    )))
    return Expr(:block, code...)
end


####################################################################### ADDITION


function multifloat_add_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()

    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))

    b_limbs = [Symbol('b', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(b_limbs, :(b._limbs)))

    terms = [Symbol[] for _ = 1:num_limbs+1]
    for i = 1:num_limbs
        sum_term = Symbol('s', i)
        err_term = Symbol('e', i)
        push!(code, _meta_two_sum(sum_term, err_term, a_limbs[i], b_limbs[i]))
        push!(terms[i], sum_term)
        push!(terms[i+1], err_term)
    end

    return _meta_return_multifloat(
        vec_width, T, num_limbs, code, terms, :two_pass_renorm
    )
end


@generated multifloat_add(a::_MF{T,N}, b::_MF{T,N}) where {T,N} =
    multifloat_add_expr(-1, T, N)
@generated multifloat_add(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} =
    multifloat_add_expr(M, T, N)


function multifloat_float_add_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()

    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))

    terms = [Symbol[] for _ = 1:num_limbs+1]
    last_term = :b
    for i = 1:num_limbs
        sum_term = Symbol('s', i)
        err_term = Symbol('e', i)
        push!(code, _meta_two_sum(sum_term, err_term, a_limbs[i], last_term))
        push!(terms[i], sum_term)
        last_term = err_term
    end
    push!(terms[num_limbs+1], last_term)

    return _meta_return_multifloat(
        vec_width, T, num_limbs, code, terms, :two_pass_renorm
    )
end


@generated multifloat_float_add(a::_MF{T,N}, b::T) where {T,N} =
    multifloat_float_add_expr(-1, T, N)
@generated multifloat_float_add(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} =
    multifloat_float_add_expr(M, T, N)


#################################################################### SUBTRACTION


function multifloat_sub_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()

    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))

    b_limbs = [Symbol('b', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(b_limbs, :(b._limbs)))

    terms = [Symbol[] for _ = 1:num_limbs+1]
    for i = 1:num_limbs
        diff_term = Symbol('d', i)
        err_term = Symbol('e', i)
        push!(code, _meta_two_diff(diff_term, err_term, a_limbs[i], b_limbs[i]))
        push!(terms[i], diff_term)
        push!(terms[i+1], err_term)
    end

    return _meta_return_multifloat(
        vec_width, T, num_limbs, code, terms, :two_pass_renorm
    )
end


@generated multifloat_sub(a::_MF{T,N}, b::_MF{T,N}) where {T,N} =
    multifloat_sub_expr(-1, T, N)
@generated multifloat_sub(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} =
    multifloat_sub_expr(M, T, N)


function multifloat_float_sub_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()

    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))

    terms = [Symbol[] for _ = 1:num_limbs+1]
    if num_limbs > 0
        diff_term = Symbol('d', 1)
        last_term = Symbol('e', 1)
        push!(code, _meta_two_diff(diff_term, last_term, a_limbs[1], :b))
        push!(terms[1], diff_term)
        for i = 2:num_limbs
            diff_term = Symbol('d', i)
            err_term = Symbol('e', i)
            push!(code, _meta_two_sum(diff_term, err_term, a_limbs[i], last_term))
            push!(terms[i], diff_term)
            last_term = err_term
        end
        push!(terms[num_limbs+1], last_term)
    else
        push!(code, :(d1 = -b))
        push!(terms[1], :d1)
    end

    return _meta_return_multifloat(
        vec_width, T, num_limbs, code, terms, :two_pass_renorm
    )
end


@generated multifloat_float_sub(a::_MF{T,N}, b::T) where {T,N} =
    multifloat_float_sub_expr(-1, T, N)
@generated multifloat_float_sub(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} =
    multifloat_float_sub_expr(M, T, N)


function float_multifloat_sub_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()

    b_limbs = [Symbol('b', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(b_limbs, :(b._limbs)))

    terms = [Symbol[] for _ = 1:num_limbs+1]
    last_term = :a
    for i = 1:num_limbs
        diff_term = Symbol('d', i)
        err_term = Symbol('e', i)
        push!(code, _meta_two_diff(diff_term, err_term, last_term, b_limbs[i]))
        push!(terms[i], diff_term)
        last_term = err_term
    end
    push!(terms[num_limbs+1], last_term)

    return _meta_return_multifloat(
        vec_width, T, num_limbs, code, terms, :two_pass_renorm
    )
end


@generated float_multifloat_sub(a::T, b::_MF{T,N}) where {T,N} =
    float_multifloat_sub_expr(-1, T, N)
@generated float_multifloat_sub(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T,N} =
    float_multifloat_sub_expr(M, T, N)


################################################################# MULTIPLICATION


function multifloat_mul_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()

    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))

    b_limbs = [Symbol('b', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(b_limbs, :(b._limbs)))

    terms = [Symbol[] for _ = 1:num_limbs]
    count = 0
    for i = 1:num_limbs-1
        for j = 1:i
            count += 1
            prod_term = Symbol('p', count)
            err_term = Symbol('e', count)
            push!(code, _meta_two_prod(
                prod_term, err_term, a_limbs[j], b_limbs[i-j+1]
            ))
            push!(terms[i], prod_term)
            push!(terms[i+1], err_term)
        end
    end
    for j = 1:num_limbs
        count += 1
        prod_term = Symbol('p', count)
        push!(code, _meta_prod(prod_term, a_limbs[j], b_limbs[num_limbs-j+1]))
        push!(terms[num_limbs], prod_term)
    end

    return _meta_return_multifloat(
        vec_width, T, num_limbs, code, terms, :two_pass_renorm
    )
end


@generated multifloat_mul(a::_MF{T,N}, b::_MF{T,N}) where {T,N} =
    multifloat_mul_expr(-1, T, N)
@generated multifloat_mul(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} =
    multifloat_mul_expr(M, T, N)


function multifloat_float_mul_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()

    a_limbs = [Symbol('a', i) for i = 1:num_limbs]
    push!(code, _meta_unpack(a_limbs, :(a._limbs)))

    terms = [Symbol[] for _ = 1:num_limbs]
    for i = 1:num_limbs-1
        prod_term = Symbol('p', i)
        err_term = Symbol('e', i)
        push!(code, _meta_two_prod(prod_term, err_term, a_limbs[i], :b))
        push!(terms[i], prod_term)
        push!(terms[i+1], err_term)
    end
    if num_limbs > 0
        prod_term = Symbol('p', num_limbs)
        push!(code, _meta_prod(prod_term, a_limbs[num_limbs], :b))
        push!(terms[num_limbs], prod_term)
    end

    return _meta_return_multifloat(
        vec_width, T, num_limbs, code, terms, :two_pass_renorm
    )
end


@generated multifloat_float_mul(a::MultiFloat{T,N}, b::T) where {T,N} =
    multifloat_float_mul_expr(-1, T, N)
@generated multifloat_float_mul(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} =
    multifloat_float_mul_expr(M, T, N)


####################################################################### DIVISION


function multifloat_div_expr(vec_width::Int, T::DataType, num_limbs::Int)
    code = _inline_block()
    terms = [[Symbol('q', i)] for i = 1:num_limbs]
    for i = 1:num_limbs
        if i == 1
            push!(code, :(r = a))
        else
            push!(code, :(r = r - b * $(only(terms[i-1]))))
        end
        push!(code, :($(only(terms[i])) = r._limbs[1] / b._limbs[1]))
    end
    return _meta_return_multifloat(
        vec_width, T, num_limbs, code, terms, :two_pass_renorm
    )
end


@generated multifloat_div(a::_MF{T,N}, b::_MF{T,N}) where {T,N} =
    multifloat_div_expr(-1, T, N)
@generated multifloat_div(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} =
    multifloat_div_expr(M, T, N)


########################################################### OPERATOR OVERLOADING


@inline Base.:+(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = multifloat_add(a, b)
@inline Base.:+(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = multifloat_add(a, b)
@inline Base.:+(a::_MF{T,N}, b::T) where {T,N} = multifloat_float_add(a, b)
@inline Base.:+(a::_MF{T,N}, b::T) where {T<:Number,N} = multifloat_float_add(a, b)
@inline Base.:+(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} = multifloat_float_add(a, b)
@inline Base.:+(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T<:Number,N} = multifloat_float_add(a, b)
@inline Base.:+(a::T, b::_MF{T,N}) where {T,N} = multifloat_float_add(b, a)
@inline Base.:+(a::T, b::_MF{T,N}) where {T<:Number,N} = multifloat_float_add(b, a)
@inline Base.:+(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T,N} = multifloat_float_add(b, a)
@inline Base.:+(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T<:Number,N} = multifloat_float_add(b, a)
@inline Base.:-(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = multifloat_sub(a, b)
@inline Base.:-(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = multifloat_sub(a, b)
@inline Base.:-(a::_MF{T,N}, b::T) where {T,N} = multifloat_float_sub(a, b)
@inline Base.:-(a::_MF{T,N}, b::T) where {T<:Number,N} = multifloat_float_sub(a, b)
@inline Base.:-(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} = multifloat_float_sub(a, b)
@inline Base.:-(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T<:Number,N} = multifloat_float_sub(a, b)
@inline Base.:-(a::T, b::_MF{T,N}) where {T,N} = float_multifloat_sub(a, b)
@inline Base.:-(a::T, b::_MF{T,N}) where {T<:Number,N} = float_multifloat_sub(a, b)
@inline Base.:-(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T,N} = float_multifloat_sub(a, b)
@inline Base.:-(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T<:Number,N} = float_multifloat_sub(a, b)
@inline Base.:*(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = multifloat_mul(a, b)
@inline Base.:*(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = multifloat_mul(a, b)
@inline Base.:*(a::_MF{T,N}, b::T) where {T,N} = multifloat_float_mul(a, b)
@inline Base.:*(a::_MF{T,N}, b::T) where {T<:Number,N} = multifloat_float_mul(a, b)
@inline Base.:*(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T,N} = multifloat_float_mul(a, b)
@inline Base.:*(a::_MFV{M,T,N}, b::Vec{M,T}) where {M,T<:Number,N} = multifloat_float_mul(a, b)
@inline Base.:*(a::T, b::_MF{T,N}) where {T,N} = multifloat_float_mul(b, a)
@inline Base.:*(a::T, b::_MF{T,N}) where {T<:Number,N} = multifloat_float_mul(b, a)
@inline Base.:*(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T,N} = multifloat_float_mul(b, a)
@inline Base.:*(a::Vec{M,T}, b::_MFV{M,T,N}) where {M,T<:Number,N} = multifloat_float_mul(b, a)
@inline Base.:/(a::_MF{T,N}, b::_MF{T,N}) where {T,N} = multifloat_div(a, b)
@inline Base.:/(a::_MFV{M,T,N}, b::_MFV{M,T,N}) where {M,T,N} = multifloat_div(a, b)


#################################################################### LEGACY CODE

#=

export renormalize,
    use_clean_multifloat_arithmetic,
    use_standard_multifloat_arithmetic,
    use_sloppy_multifloat_arithmetic

############################################## CONSTRUCTION FROM PRIMITIVE TYPES

@inline MultiFloat{T,N}(x::MultiFloat{T,N}) where {T,N} = x

@inline MultiFloat{T,N}(x::MultiFloat{T,M}) where {T,M,N} =
    MultiFloat{T,N}((
        ntuple(i -> x._limbs[i], Val{min(M, N)}())...,
        ntuple(_ -> zero(T), Val{max(N - M, 0)}())...
    ))

# Values of the types Bool, Int8, UInt8, Int16, UInt16, Float16, Int32, UInt32,
# and Float32 can be converted losslessly to a single Float64, which has 53
# bits of integer precision.

@inline Float64x{N}(x::Bool) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Int8) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::UInt8) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Int16) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::UInt16) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Float16) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Int32) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::UInt32) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Float32) where {N} = Float64x{N}(Float64(x))

# Values of the types Int64, UInt64, Int128, and UInt128 cannot be converted
# losslessly to a single Float64 and must be split into multiple components.

@inline Float64x1(x::Int64) = Float64x1(Float64(x))
@inline Float64x1(x::UInt64) = Float64x1(Float64(x))
@inline Float64x1(x::Int128) = Float64x1(Float64(x))
@inline Float64x1(x::UInt128) = Float64x1(Float64(x))

@inline function Float64x2(x::Int128)
    x0 = Float64(x)
    x1 = Float64(x - Int128(x0))
    Float64x2((x0, x1))
end

@inline function Float64x2(x::UInt128)
    x0 = Float64(x)
    x1 = Float64(reinterpret(Int128, x - UInt128(x0)))
    Float64x2((x0, x1))
end

@inline function Float64x{N}(x::Int64) where {N}
    x0 = Float64(x)
    x1 = Float64(x - Int64(x0))
    Float64x{N}((x0, x1, ntuple(_ -> 0.0, Val{N - 2}())...))
end

@inline function Float64x{N}(x::UInt64) where {N}
    x0 = Float64(x)
    x1 = Float64(reinterpret(Int64, x - UInt64(x0)))
    Float64x{N}((x0, x1, ntuple(_ -> 0.0, Val{N - 2}())...))
end

@inline function Float64x{N}(x::Int128) where {N}
    x0 = Float64(x)
    r1 = x - Int128(x0)
    x1 = Float64(r1)
    x2 = Float64(r1 - Int128(x1))
    Float64x{N}((x0, x1, x2, ntuple(_ -> 0.0, Val{N - 3}())...))
end

@inline function Float64x{N}(x::UInt128) where {N}
    x0 = Float64(x)
    r1 = reinterpret(Int128, x - UInt128(x0))
    x1 = Float64(r1)
    x2 = Float64(r1 - Int128(x1))
    Float64x{N}((x0, x1, x2, ntuple(_ -> 0.0, Val{N - 3}())...))
end

################################################## CONVERSION TO PRIMITIVE TYPES

@inline Base.Float16(x::Float64x{N}) where {N} = Float16(x._limbs[1])
@inline Base.Float32(x::Float64x{N}) where {N} = Float32(x._limbs[1])

@inline Base.Float16(x::Float16x{N}) where {N} = x._limbs[1]
@inline Base.Float32(x::Float32x{N}) where {N} = x._limbs[1]
@inline Base.Float64(x::Float64x{N}) where {N} = x._limbs[1]

####################################################### FLOATING-POINT CONSTANTS

# overload Base._precision to support the base keyword in Julia 1.8
let precision = isdefined(Base, :_precision) ? (:_precision) : (:precision)
    @eval @inline Base.$precision(::Type{_MF{T,N}}) where {T,N} =
        N * precision(T) + (N - 1) # implicit bits of precision between limbs
end

@inline Base.zero(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(zero(T))
@inline Base.one(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(one(T))
@inline Base.eps(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(eps(T)^N)

# TODO: This is technically not the maximum/minimum representable MultiFloat.
@inline Base.floatmin(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(floatmin(T))
@inline Base.floatmax(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(floatmax(T))

@inline Base.typemin(::Type{_MF{T,N}}) where {T,N} =
    _MF{T,N}(ntuple(_ -> typemin(T), Val{N}()))
@inline Base.typemax(::Type{_MF{T,N}}) where {T,N} =
    _MF{T,N}(ntuple(_ -> typemax(T), Val{N}()))

#################################################### CONSTRUCTION FROM BIG TYPES

tuple_from_collection(collection, ::Val{n}) where {n} =
    ntuple(
        let c = collection
            i -> c[begin-1+i]
        end,
        Val{n}())

function constructed_from_big(::Type{T}, ::Val{N}, x::Src) where
{T<:Real,N,Src<:Union{BigInt,BigFloat}}
    y = Vector{T}(undef, N)
    y[1] = T(x)
    for i = 2:N
        x -= y[i-1]
        y[i] = T(x)
    end
    MultiFloat{T,N}(tuple_from_collection(y, Val{N}()))
end

function MultiFloat{T,N}(x::BigFloat) where {T,N}
    if x > floatmax(T)
        return typemax(MultiFloat{T,N})
    elseif x < -floatmax(T)
        return typemin(MultiFloat{T,N})
    elseif -floatmin(T) < x < floatmin(T)
        return zero(MultiFloat{T,N})
    elseif isnan(x)
        return _MF{T,N}(ntuple(_ -> T(NaN), Val{N}()))
    end
    setrounding(
        let x = x
            () -> setprecision(
                let x = x
                    () -> constructed_from_big(T, Val{N}(), x)
                end,
                BigFloat,
                precision(x))
        end,
        BigFloat,
        RoundNearest)
end

MultiFloat{T,N}(x::BigInt) where {T,N} =
    constructed_from_big(T, Val{N}(), x)

MultiFloat{T,N}(x::Rational{U}) where {T,N,U} =
    MultiFloat{T,N}(numerator(x)) / MultiFloat{T,N}(denominator(x))

MultiFloat{T,N}(x::AbstractString) where {T,N} =
    MultiFloat{T,N}(BigFloat(x, precision=(
        precision(T) + exponent(floatmax(T)) - exponent(floatmin(T))
    )))

######################################################## CONVERSION TO BIG TYPES

Base.BigFloat(x::MultiFloat{T,N}) where {T,N} =
    +(ntuple(i -> BigFloat(x._limbs[N-i+1]), Val{N}())...)

Base.Rational{BigInt}(x::MultiFloat{T,N}) where {T,N} =
    sum(Rational{BigInt}.(x._limbs))

################################################################ PROMOTION RULES

Base.promote_rule(::Type{_MF{T,N}}, ::Type{T}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int8}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int16}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int32}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int64}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Int128}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{Bool}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt8}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt16}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt32}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt64}) where {T,N} = _MF{T,N}
Base.promote_rule(::Type{_MF{T,N}}, ::Type{UInt128}) where {T,N} = _MF{T,N}

Base.promote_rule(::Type{_MF{T,N}}, ::Type{BigFloat}) where {T,N} = BigFloat

Base.promote_rule(::Type{Float32x{N}}, ::Type{Float16}) where {N} = Float32x{N}
Base.promote_rule(::Type{Float64x{N}}, ::Type{Float16}) where {N} = Float64x{N}
Base.promote_rule(::Type{Float64x{N}}, ::Type{Float32}) where {N} = Float64x{N}

################################################################ RENORMALIZATION

@inline function renormalize(x::_MF{T,N}) where {T,N}
    total = +(x._limbs...)
    if isfinite(total)
        while true
            x0::_MF{T,N} = x + zero(T)
            if !(x0._limbs != x._limbs)
                break
            end
            x = x0
        end
        return x
    else
        return MultiFloat{T,N}(ntuple(_ -> total, Val{N}()))
    end
end

@inline renormalize(x::T) where {T<:Number} = x

function call_normalized(callback, x::MultiFloat{T,N}) where {T,N}
    x = renormalize(x)
    if !isfinite(x._limbs[1])
        return callback(x._limbs[1])
    else
        i = N
        while (i > 0) && iszero(x._limbs[i])
            i -= 1
        end
        if iszero(i)
            return callback(zero(T))
        else
            return setprecision(() -> callback(BigFloat(x)),
                precision(T) + exponent(x._limbs[1]) - exponent(x._limbs[i]))
        end
    end
end

####################################################################### PRINTING

function Base.show(io::IO, x::MultiFloat{T,N}) where {T,N}
    return call_normalized(y -> show(io, y), x)
end

import Printf: tofloat
tofloat(x::MultiFloat{T,N}) where {T,N} = call_normalized(BigFloat, x)

################################################### FLOATING-POINT INTROSPECTION

@inline _iszero(x::_MF{T,N}) where {T,N} =
    (&)(ntuple(i -> iszero(x._limbs[i]), Val{N}())...)
@inline _isone(x::_MF{T,N}) where {T,N} =
    isone(x._limbs[1]) & (&)(ntuple(
        i -> iszero(x._limbs[i+1]),
        Val{N - 1}()
    )...)

@inline Base.iszero(x::_MF{T,1}) where {T} = iszero(x._limbs[1])
@inline Base.isone(x::_MF{T,1}) where {T} = isone(x._limbs[1])
@inline Base.iszero(x::_MF{T,N}) where {T,N} = _iszero(renormalize(x))
@inline Base.isone(x::_MF{T,N}) where {T,N} = _isone(renormalize(x))

@inline _head(x::_MF{T,N}) where {T,N} = renormalize(x)._limbs[1]
@inline Base.exponent(x::_MF{T,N}) where {T,N} = exponent(_head(x))
@inline Base.signbit(x::_MF{T,N}) where {T,N} = signbit(_head(x))
@inline Base.issubnormal(x::_MF{T,N}) where {T,N} = issubnormal(_head(x))
@inline Base.isfinite(x::_MF{T,N}) where {T,N} = isfinite(_head(x))
@inline Base.isinf(x::_MF{T,N}) where {T,N} = isinf(_head(x))
@inline Base.isnan(x::_MF{T,N}) where {T,N} = isnan(_head(x))
@inline Base.isinteger(x::_MF{T,N}) where {T,N} =
    all(isinteger.(renormalize(x)._limbs))

@inline function Base.nextfloat(x::_MF{T,N}) where {T,N}
    y = renormalize(x)
    return renormalize(_MF{T,N}((
        ntuple(i -> y._limbs[i], Val{N - 1}())...,
        nextfloat(y._limbs[N]))))
end

@inline function Base.prevfloat(x::_MF{T,N}) where {T,N}
    y = renormalize(x)
    return renormalize(_MF{T,N}((
        ntuple(i -> y._limbs[i], Val{N - 1}())...,
        prevfloat(y._limbs[N]))))
end

import LinearAlgebra: floatmin2
@inline floatmin2(::Type{_MF{T,N}}) where {T,N} = _MF{T,N}(ldexp(one(T),
    div(exponent(floatmin(T)) - N * exponent(eps(T)), 2)))

##################################################################### COMPARISON

@inline Base.:(==)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    multifloat_eq(renormalize(x), renormalize(y))
@inline Base.:(!=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    multifloat_ne(renormalize(x), renormalize(y))
@inline Base.:(<)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    multifloat_lt(renormalize(x), renormalize(y))
@inline Base.:(>)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    multifloat_gt(renormalize(x), renormalize(y))
@inline Base.:(<=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    multifloat_le(renormalize(x), renormalize(y))
@inline Base.:(>=)(x::_MF{T,N}, y::_MF{T,N}) where {T,N} =
    multifloat_ge(renormalize(x), renormalize(y))

##################################################################### ARITHMETIC

@inline Base.:/(x::_MF{T,N}, y::_MF{T,N}) where {T,N} = multifloat_div(x, y)

@inline Base.:-(x::_MF{T,N}) where {T,N} =
    _MF{T,N}(ntuple(i -> -x._limbs[i], Val{N}()))

@inline Base.inv(x::_MF{T,N}) where {T,N} = one(_MF{T,N}) / x

@inline function Base.abs(x::_MF{T,N}) where {T,N}
    x = renormalize(x)
    ifelse(signbit(x._limbs[1]), -x, x)
end

@inline function Base.abs2(x::_MF{T,N}) where {T,N}
    x = renormalize(x)
    renormalize(x * x)
end

@inline unsafe_sqrt(x::Float32) = Base.sqrt_llvm(x)
@inline unsafe_sqrt(x::Float64) = Base.sqrt_llvm(x)
@inline unsafe_sqrt(x::T) where {T<:Real} = sqrt(x)

@inline function Base.sqrt(x::_MF{T,N}) where {T,N}
    x = renormalize(x)
    if iszero(x)
        return x
    else
        return multifloat_sqrt(x)
    end
end

######################################################## EXPONENTIATION (BASE 2)

@inline function Base.ldexp(x::_MF{T,N}, n::U) where {T,N,U<:Integer}
    x = renormalize(x)
    return MultiFloat{T,N}(ntuple(i -> ldexp(x._limbs[i], n), Val{N}()))
end

######################################################## EXPONENTIATION (BASE E)

const INVERSE_FACTORIALS_F64 = setprecision(
    () -> [
        MultiFloat{Float64,20}(inv(BigFloat(factorial(BigInt(i)))))
        for i = 1:170
    ],
    BigFloat, 1200
)

const LOG2_F64 = setprecision(
    () -> MultiFloat{Float64,20}(log(BigFloat(2))),
    BigFloat, 1200
)

log2_f64_literal(n::Int) = :(
    MultiFloat{Float64,$n}($(LOG2_F64._limbs[1:n])))

inverse_factorial_f64_literal(n::Int, i::Int) = :(
    MultiFloat{Float64,$n}($(INVERSE_FACTORIALS_F64[i]._limbs[1:n])))

meta_y(n::Int) = Symbol("y_", n)

meta_y_definition(n::Int) = Expr(:(=), meta_y(n),
    Expr(:call, :*, meta_y(div(n, 2)), meta_y(div(n + 1, 2))))

meta_exp_term(n::Int, i::Int) = :(
    $(Symbol("y_", i)) * $(inverse_factorial_f64_literal(n, i)))

function multifloat_exp_func(
    n::Int, num_terms::Int, reduction_power::Int; sloppy::Bool=false
)
    return Expr(:function,
        :(multifloat_exp(x::MultiFloat{Float64,$n})),
        Expr(:block,
            :(exponent_f = Base.rint_llvm(
                x._limbs[1] / $(LOG2_F64._limbs[1]))),
            :(exponent_i = Base.fptosi(Int, exponent_f)),
            :(y_1 = scale(
                $(ldexp(1.0, -reduction_power)),
                MultiFloat{Float64,$n}(
                    MultiFloat{Float64,$(n + !sloppy)}(x) -
                    exponent_f * $(log2_f64_literal(n + !sloppy))
                )
            )),
            [meta_y_definition(i) for i = 2:num_terms]...,
            :(exp_y = $(meta_exp_term(n, num_terms))),
            [
                :(exp_y += $(meta_exp_term(n, i)))
                for i = num_terms-1:-1:3
            ]...,
            :(exp_y += scale(0.5, y_2)),
            :(exp_y += y_1),
            :(exp_y += 1.0),
            [:(exp_y *= exp_y) for _ = 1:reduction_power]...,
            :(return scale(
                reinterpret(Float64, UInt64(1023 + exponent_i) << 52),
                exp_y
            ))
        )
    )
end

@inline multifloat_exp(x::_MF{T,1}) where {T} = _MF{T,1}(exp(x._limbs[1]))

function Base.exp(x::MultiFloat{T,N}) where {T,N}
    x = renormalize(x)
    if x._limbs[1] >= log(floatmax(Float64))
        return typemax(MultiFloat{T,N})
    elseif x._limbs[1] <= log(floatmin(Float64))
        return zero(MultiFloat{T,N})
    else
        return multifloat_exp(x)
    end
end

function Base.log(x::MultiFloat{T,N}) where {T,N}
    y = MultiFloat{T,N}(log(x._limbs[1]))
    for _ = 1:ceil(Int, log2(N))+1
        y += x * exp(-y) - one(T)
    end
    return y
end

function Base.log1p(x::MultiFloat{T,N}) where {T,N}
    return 2 * atanh(x / (2 + x))
end

function Base.expm1(x::MultiFloat{T,N}) where {T,N}
    t = tanh(x / 2)
    return 2 * t / (1 - t)
end

############################################## BIGFLOAT TRANSCENDENTAL STOP-GAPS

BASE_TRANSCENDENTAL_FUNCTIONS = [
    :exp2, :exp10, :log2, :log10,
    :sin, :cos, :tan, :sec, :csc, :cot,
    :sinpi, :cospi,
    :sinh, :cosh, :tanh, :sech, :csch, :coth,
    :sind, :cosd, :tand, :secd, :cscd, :cotd,
    :asin, :acos, :atan, :asec, :acsc, :acot,
    :asinh, :acosh, :atanh, :asech, :acsch, :acoth,
    :asind, :acosd, :atand, :asecd, :acscd, :acotd
]

for name in BASE_TRANSCENDENTAL_FUNCTIONS
    eval(:(Base.$name(x::MultiFloat{T,N}) where {T,N} = error($(
        "$name(MultiFloat) is not yet implemented. For a temporary workaround,\n" *
        "call MultiFloats.use_bigfloat_transcendentals() immediately after\n" *
        "importing MultiFloats. This will use the BigFloat implementation of\n" *
        "$name, which will not be as fast as a pure-MultiFloat implementation.\n"
    ))))
end

eval_bigfloat(f::Function, x::MultiFloat{T,N}, k::Int) where {T,N} =
    setprecision(
        () -> MultiFloat{T,N}(f(BigFloat(x))),
        BigFloat, precision(MultiFloat{T,N}) + k
    )

function use_bigfloat_transcendentals(k::Int=20)
    for name in BASE_TRANSCENDENTAL_FUNCTIONS
        eval(:(
            Base.$name(x::MultiFloat{T,N}) where {T,N} =
                eval_bigfloat($name, x, $k)
        ))
    end
end

####################################################### RANDOM NUMBER GENERATION

import Random
using Random: AbstractRNG, SamplerTrivial, CloseOpen01

@inline function _rand_f64(rng::AbstractRNG, k::Int)
    expnt = reinterpret(UInt64,
        exponent(floatmax(Float64)) + k) << (precision(Float64) - 1)
    mntsa = rand(rng, Random.UInt52())
    return reinterpret(Float64, expnt | mntsa)
end

@inline function _rand_sf64(rng::AbstractRNG, k::Int)
    expnt = reinterpret(UInt64,
        exponent(floatmax(Float64)) + k) << (precision(Float64) - 1)
    mntsa = rand(rng, UInt64) & 0x800FFFFFFFFFFFFF
    return reinterpret(Float64, expnt | mntsa)
end

@inline function _rand_mf64(
    rng::AbstractRNG, offset::Int, padding::NTuple{N,Int}
) where {N}
    exponents = (
        cumsum(padding) .+
        (precision(Float64) + 1) .* ntuple(identity, Val{N}())
    )
    return Float64x{N + 1}((
        _rand_f64(rng, offset),
        _rand_sf64.(rng, offset .- exponents)...
    ))
end

function multifloat_rand_func(n::Int)
    return :(
        Random.rand(
            rng::AbstractRNG,
            ::SamplerTrivial{CloseOpen01{Float64x{$n}}}
        ) = _rand_mf64(
            rng,
            -leading_zeros(rand(rng, UInt64)) - 1,
            $(Expr(:tuple, ntuple(
                _ -> :(leading_zeros(rand(rng, UInt64))),
                Val{n - 1}()
            )...))
        )
    )
end

################################################################ PRECISION MODES

function use_clean_multifloat_arithmetic(n::Integer=8)
    for i = 1:n
        eval(multifloat_eq_func(i))
        eval(multifloat_ne_func(i))
        eval(multifloat_lt_func(i))
        eval(multifloat_gt_func(i))
        eval(multifloat_le_func(i))
        eval(multifloat_ge_func(i))
        eval(multifloat_rand_func(i))
    end
    for i = 2:n+1
        eval(two_pass_renorm_func(i, sloppy=false))
        eval(multifloat_add_func(i, sloppy=false))
        eval(multifloat_mul_func(i, sloppy=false))
        eval(multifloat_div_func(i, sloppy=false))
        eval(multifloat_float_add_func(i, sloppy=false))
        eval(multifloat_float_mul_func(i, sloppy=false))
        eval(multifloat_sqrt_func(i, sloppy=false))
    end
    eval(MultiFloats.multifloat_exp_func(2, 20, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(3, 28, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(4, 35, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(5, 42, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(6, 49, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(7, 56, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(8, 63, 1, sloppy=false))
    for (_, v) in Arithmetic.MPADD_CACHE
        eval(v)
    end
end

function use_standard_multifloat_arithmetic(n::Integer=8)
    for i = 1:n
        eval(multifloat_eq_func(i))
        eval(multifloat_ne_func(i))
        eval(multifloat_lt_func(i))
        eval(multifloat_gt_func(i))
        eval(multifloat_le_func(i))
        eval(multifloat_ge_func(i))
        eval(multifloat_rand_func(i))
    end
    for i = 2:n
        eval(two_pass_renorm_func(i, sloppy=true))
        eval(two_pass_renorm_func(i, sloppy=false))
        eval(multifloat_add_func(i, sloppy=false))
        eval(multifloat_mul_func(i, sloppy=true))
        eval(multifloat_div_func(i, sloppy=true))
        eval(multifloat_float_add_func(i, sloppy=false))
        eval(multifloat_float_mul_func(i, sloppy=true))
        eval(multifloat_sqrt_func(i, sloppy=true))
    end
    eval(MultiFloats.multifloat_exp_func(2, 17, 2, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(3, 19, 4, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(4, 20, 6, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(5, 23, 7, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(6, 23, 9, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(7, 22, 12, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(8, 24, 13, sloppy=true))
    for (_, v) in Arithmetic.MPADD_CACHE
        eval(v)
    end
end

function use_sloppy_multifloat_arithmetic(n::Integer=8)
    for i = 1:n
        eval(multifloat_eq_func(i))
        eval(multifloat_ne_func(i))
        eval(multifloat_lt_func(i))
        eval(multifloat_gt_func(i))
        eval(multifloat_le_func(i))
        eval(multifloat_ge_func(i))
        eval(multifloat_rand_func(i))
    end
    for i = 2:n
        eval(one_pass_renorm_func(i, sloppy=true))
        eval(multifloat_add_func(i, sloppy=true))
        eval(multifloat_mul_func(i, sloppy=true))
        eval(multifloat_div_func(i, sloppy=true))
        eval(multifloat_float_add_func(i, sloppy=true))
        eval(multifloat_float_mul_func(i, sloppy=true))
        eval(multifloat_sqrt_func(i, sloppy=true))
    end
    eval(MultiFloats.multifloat_exp_func(2, 17, 2, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(3, 19, 4, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(4, 20, 6, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(5, 23, 7, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(6, 23, 9, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(7, 22, 12, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(8, 24, 13, sloppy=true))
    for (_, v) in Arithmetic.MPADD_CACHE
        eval(v)
    end
end

use_standard_multifloat_arithmetic()

=#

end # module MultiFloats
