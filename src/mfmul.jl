@inline function mfmul(
    x::NTuple{1,T},
    y::NTuple{1,T},
    ::Val{1},
) where {T}
    return (one_prod(x[1], y[1]),)
end


@inline function mfmul(
    x::NTuple{2,T},
    y::NTuple{2,T},
    ::Val{2},
) where {T}
    p00, e00 = two_prod(x[1], y[1])
    p01 = one_prod(x[1], y[2])
    p10 = one_prod(x[2], y[1])
    p01 += p10
    e00 += p01
    p00, e00 = fast_two_sum(p00, e00)
    return (p00, e00)
end


@inline function mfmul(
    x::NTuple{3,T},
    y::NTuple{3,T},
    ::Val{3},
) where {T}
    p00, e00 = two_prod(x[1], y[1])
    p01, e01 = two_prod(x[1], y[2])
    p10, e10 = two_prod(x[2], y[1])
    p02 = one_prod(x[1], y[3])
    p11 = one_prod(x[2], y[2])
    p20 = one_prod(x[3], y[1])
    p01, p10 = two_sum(p01, p10)
    e01 += e10
    p02 += p20
    e00, p01 = two_sum(e00, p01)
    p02 += p11
    p00, e00 = fast_two_sum(p00, e00)
    p01 += p10
    e01 += p02
    p01 += e01
    e00, p01 = two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    e00, p01 = fast_two_sum(e00, p01)
    p00, e00 = fast_two_sum(p00, e00)
    return (p00, e00, p01)
end


@inline function mfmul(
    x::NTuple{4,T},
    y::NTuple{4,T},
    ::Val{4},
) where {T}
    p00, e00 = two_prod(x[1], y[1])
    p01, e01 = two_prod(x[1], y[2])
    p10, e10 = two_prod(x[2], y[1])
    p02, e02 = two_prod(x[1], y[3])
    p11, e11 = two_prod(x[2], y[2])
    p20, e20 = two_prod(x[3], y[1])
    p03 = one_prod(x[1], y[4])
    p12 = one_prod(x[2], y[3])
    p21 = one_prod(x[3], y[2])
    p30 = one_prod(x[4], y[1])
    p01, p10 = two_sum(p01, p10)
    e01, e10 = two_sum(e01, e10)
    p02, p20 = two_sum(p02, p20)
    e02 += e20
    p03 += p30
    p12 += p21
    e00, p01 = two_sum(e00, p01)
    e01, p11 = two_sum(e01, p11)
    e10 += e02
    p20 += e11
    p03 += p12
    p00, e00 = fast_two_sum(p00, e00)
    p01, p10 = fast_two_sum(p01, p10)
    e01, p02 = two_sum(e01, p02)
    e10 += p03
    p11 += p20
    p01, e01 = two_sum(p01, e01)
    p10 += p11
    e10 += p02
    p10 += e01
    p01, p10 = two_sum(p01, p10)
    e00, p01 = two_sum(e00, p01)
    p10 += e10
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
