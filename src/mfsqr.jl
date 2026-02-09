@inline function mfsqr(
    x::NTuple{1,T},
    ::Val{1},
) where {T}
    return (x[1] * x[1],)
end


@inline function mfsqr(
    x::NTuple{2,T},
    ::Val{2},
) where {T}
    p00, e00 = two_prod(x[1], x[1])
    e00 = fma(x[1], twice(x[2]), e00)
    p00, e00 = fast_two_sum(p00, e00)
    return (p00, e00)
end


@inline function mfsqr(
    x::NTuple{3,T},
    ::Val{3},
) where {T}
    p00, e00 = two_prod(x[1], x[1])
    p01, e01 = two_prod(x[1], twice(x[2]))
    e00, p01 = two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    p01 += e01 + fma(x[1], twice(x[3]), one_prod(x[2], x[2]))
    e00, p01 = two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    e00, p01 = fast_two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    return (p00, e00, p01)
end


@inline function mfsqr(
    x::NTuple{4,T},
    ::Val{4},
) where {T}
    p00, e00 = two_prod(x[1], x[1])
    p01, e01 = two_prod(x[1], twice(x[2]))
    p02, e02 = two_prod(x[1], twice(x[3]))
    p11, e11 = two_prod(x[2], x[2])
    e00, p01 = two_sum(e00, p01)
    e01, p11 = two_sum(e01, p11)
    p00, e00 = fast_two_sum(p00, e00)
    e01, p02 = two_sum(e01, p02)
    p11 += e11
    p01, e01 = two_sum(p01, e01)
    p01, p10 = two_sum(p01, p11 + e01)
    e00, p01 = two_sum(e00, p01)
    p10 += (e02 + fma(x[1], twice(x[4]), one_prod(x[2], twice(x[3])))) + p02
    p00, e00 = fast_two_sum(p00, e00)
    p01, p10 = two_sum(p01, p10)
    e00, p01 = two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    p01, p10 = fast_two_sum(p01, p10)
    e00, p01 = fast_two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    p01, p10 = fast_two_sum(p01, p10)
    e00, p01 = fast_two_sum(e00, p01)
    p01, p10 = fast_two_sum(p01, p10)
    return (p00, e00, p01, p10)
end
