@inline function mfadd(
    x::NTuple{1,T},
    y::NTuple{1,T},
    ::Val{1},
) where {T}
    return (x[1] + y[1],)
end


@inline function mfadd(
    x::NTuple{2,T},
    y::NTuple{2,T},
    ::Val{2},
) where {T}
    a, b = two_sum(x[1], y[1])
    c, d = two_sum(x[2], y[2])
    a, c = fast_two_sum(a, c)
    b += d
    b += c
    a, b = fast_two_sum(a, b)
    return (a, b)
end


@inline function mfadd(
    x::NTuple{3,T},
    y::NTuple{3,T},
    ::Val{3},
) where {T}
    a, b = two_sum(x[1], y[1])
    c, d = two_sum(x[2], y[2])
    e, f = two_sum(x[3], y[3])
    a, c = fast_two_sum(a, c)
    b += f
    d, e = two_sum(d, e)
    a, d = fast_two_sum(a, d)
    b, c = two_sum(b, c)
    c += e
    c, d = two_sum(c, d)
    b, c = two_sum(b, c)
    a, b = fast_two_sum(a, b)
    c += d
    b, c = fast_two_sum(b, c)
    a, b = fast_two_sum(a, b)
    b, c = fast_two_sum(b, c)
    return (a, b, c)
end


@inline function mfadd(
    x::NTuple{4,T},
    y::NTuple{4,T},
    ::Val{4},
) where {T}
    a, b = two_sum(x[1], y[1])
    c, d = two_sum(x[2], y[2])
    e, f = two_sum(x[3], y[3])
    g, h = two_sum(x[4], y[4])
    a, c = fast_two_sum(a, c)
    b += h
    d, e = two_sum(d, e)
    f, g = two_sum(f, g)
    b, g = two_sum(b, g)
    c, d = fast_two_sum(c, d)
    e, f = two_sum(e, f)
    a, c = fast_two_sum(a, c)
    d, e = fast_two_sum(d, e)
    b, d = two_sum(b, d)
    c, g = fast_two_sum(c, g)
    e += f
    b, c = two_sum(b, c)
    d, e = two_sum(d, e)
    a, b = fast_two_sum(a, b)
    c, d = two_sum(c, d)
    e += g
    b, c = fast_two_sum(b, c)
    d, e = two_sum(d, e)
    a, b = fast_two_sum(a, b)
    c, d = fast_two_sum(c, d)
    b, c = fast_two_sum(b, c)
    d += e
    a, b = fast_two_sum(a, b)
    c, d = fast_two_sum(c, d)
    b, c = fast_two_sum(b, c)
    c, d = fast_two_sum(c, d)
    return (a, b, c, d)
end
