module Remez

using LinearAlgebra: ColumnNorm, dot, ldiv!, qr!

export minimax_polynomial


@inline _halve(x::T) where {T} = ldexp(x, -1)
@inline _halve(x::Float64) = 0.5 * x


@inline _twice(x::T) where {T} = x + x
@inline _twice(x::BigFloat) = ldexp(x, 1)


@inline _to_ntuple(x::Vector{T}, ::Val{N}) where {T,N} =
    ntuple(i -> (@inbounds x[i]), Val{N}())


function chebyshev_nodes(a::T, b::T, ::Val{N}) where {T,N}
    midpoint = _halve(a + b)
    half_width = _halve(b - a)
    pi_over_2N = T(pi) / (2 * N)
    return ntuple(k -> clamp(muladd(
                cos((2 * (N - k) + 1) * pi_over_2N),
                half_width, midpoint), a, b), Val{N}())
end


function chebyshev_values(x::T, ::Val{N}) where {T,N}
    _one = one(T)
    result = ntuple(_ -> _one, Val{N + 1}())
    if N >= 1
        result = Base.setindex(result, x, 2)
    end
    if N >= 2
        two_x = _twice(x)
        for k = 3:N+1
            result = Base.setindex(result,
                muladd(two_x, result[k-1], -result[k-2]), k)
        end
    end
    return result
end


function chebyshev_derivatives(x::T, ::Val{N}) where {T,N}
    _zero = zero(T)
    _one = one(T)
    result = ntuple(_ -> _zero, Val{N + 1}())
    if N >= 1
        result = Base.setindex(result, _one, 2)
    end
    if N >= 2
        two_x = _twice(x)
        t_prev = _one
        t_curr = x
        for k = 3:N+1
            result = Base.setindex(result,
                muladd(two_x, result[k-1], -result[k-2]) +
                _twice(t_curr), k)
            t_next = muladd(two_x, t_curr, -t_prev)
            t_prev, t_curr = t_curr, t_next
        end
    end
    return result
end


function chebyshev_second_derivatives(x::T, ::Val{N}) where {T,N}
    _zero = zero(T)
    result = ntuple(_ -> _zero, Val{N + 1}())
    if N >= 2
        _one = one(T)
        two_x = _twice(x)
        t_prev = _one
        t_curr = x
        d_prev = _zero
        d_curr = _one
        for k = 3:N+1
            result = Base.setindex(result,
                muladd(two_x, result[k-1], -result[k-2]) +
                _twice(_twice(d_curr)), k)
            d_next = muladd(two_x, d_curr, -d_prev) + _twice(t_curr)
            d_prev, d_curr = d_curr, d_next
            t_next = muladd(two_x, t_curr, -t_prev)
            t_prev, t_curr = t_curr, t_next
        end
    end
    return result
end


function chebyshev_to_monomial_matrix(a::T, b::T, n::Int) where {T}
    @assert !signbit(n)
    _zero = zero(T)
    _one = one(T)

    inv_width = inv(b - a)
    scale = _twice(inv_width)
    shift = -(a + b) * inv_width
    twice_scale = _twice(scale)
    twice_shift = _twice(shift)

    B = Matrix{T}(undef, n + 1, n + 1)
    @inbounds begin
        B[1, 1] = _one
        B[2:n+1, 1] .= _zero
        if n >= 1
            B[1, 2] = shift
            B[2, 2] = scale
            B[3:n+1, 2] .= _zero
            for j = 3:n+1
                B[1, j] = muladd(twice_shift, B[1, j-1], -B[1, j-2])
                for i = 2:n+1
                    B[i, j] = muladd(twice_scale, B[i-1, j-1],
                        muladd(twice_shift, B[i, j-1], -B[i, j-2]))
                end
            end
        end
    end
    return B
end


function find_root(f::F, df::G, x::T, a::T, b::T) where {F,G,T}
    @assert a <= b

    fa = f(a)
    fb = f(b)
    if signbit(fa) == signbit(fb)
        if abs(fa) < abs(fb)
            return (a, a)
        elseif abs(fa) > abs(fb)
            return (b, b)
        else
            return (a, b)
        end
    end

    x = clamp(x, a, b)
    fx = f(x)
    dfx = df(x)
    x_lo = a
    x_hi = b
    f_lo = fa
    f_hi = fb
    dx_prev = b - a

    while true
        @assert x_lo <= x <= x_hi
        @assert !isnan(f_lo)
        @assert !isnan(f_hi)
        @assert signbit(f_lo) != signbit(f_hi)
        @assert !isnan(fx)
        @assert !isnan(dfx)

        x_mid = _halve(x_lo + x_hi)
        if !(x_lo < x_mid < x_hi)
            # The interval has become so small that its midpoint coincides
            # with one of its endpoints. We cannot make any more progress.
            return (x_lo, x_hi)
        end

        x_next = x_mid
        dx = fx / dfx
        if abs(dx) < _halve(abs(dx_prev))
            # Take Newton step if consecutive steps are getting smaller.
            x_next = x - dx
            if !(x_lo < x_next < x_hi)
                # If Newton lands outside the interval, revert to bisection.
                x_next = x_mid
                dx = x_hi - x_lo
            end
        else
            # If consecutive steps are not getting smaller, Newton's
            # method may be failing to converge. Revert to bisection.
            dx = x_hi - x_lo
        end

        # Accept step and update interval endpoints.
        x = x_next
        fx = f(x)
        dfx = df(x)
        if signbit(fx) == signbit(f_lo)
            x_lo = x
            f_lo = fx
        else
            x_hi = x
            f_hi = fx
        end
        dx_prev = dx

    end
end


function minimax_polynomial(
    f::F, df::G, d2f::H, a::T, b::T, ::Val{N};
    fixed_coefficients::NTuple{K,Pair{Int,T}}=(),
    initial_nodes=nothing,
    objective::Symbol=:absolute,
) where {F,G,H,T,N,K}

    # Validate inputs.
    @assert a <= b
    M = N - K + 2
    @assert M >= 2
    for (d, _) in fixed_coefficients
        @assert 0 <= d <= N
    end
    @assert isnothing(initial_nodes) || issorted(initial_nodes)
    if isnothing(initial_nodes)
        initial_nodes = chebyshev_nodes(a, b, Val{M}())
    end
    x = collect(initial_nodes)
    @assert eltype(x) <: T
    @assert length(x) == M
    @assert (objective == :absolute) || (objective == :relative)

    _zero = zero(T)
    _one = one(T)

    # Compute scaling parameters that map [a, b] to [-1, +1].
    inv_width = inv(b - a)
    scale = _twice(inv_width)
    scale2 = scale^2
    shift = -(a + b) * inv_width

    # Allocate workspace arrays.
    A = Matrix{T}(undef, N + 2, N + 2)
    B = chebyshev_to_monomial_matrix(a, b, N)
    v = Vector{T}(undef, N + 2)
    x_next = Vector{T}(undef, M)
    prev_deviation = nothing
    @inbounds while true

        # Set up equioscillation equations (rows 1:M).
        for i = 1:M
            fi = f(x[i])
            A[i, 1:N+1] .= chebyshev_values(
                muladd(x[i], scale, shift), Val{N}())
            if objective == :absolute
                A[i, N+2] = ifelse(isodd(i), +_one, -_one)
            else # objective == :relative
                A[i, N+2] = ifelse(isodd(i), +fi, -fi)
            end
            v[i] = fi
        end

        # Set up constraint equations (rows M+1:N+2).
        for (i, (d, c)) in enumerate(fixed_coefficients)
            A[M+i, 1:N+1] .= view(B, d + 1, 1:N+1)
            A[M+i, N+2] = _zero
            v[M+i] = c
        end

        # Solve linear system using column-pivoted QR.
        ldiv!(qr!(A, ColumnNorm()), v)
        c = _to_ntuple(v, Val{N + 1}())
        E_nominal = v[N+2]

        E_max = _zero
        max_deviation = _zero
        for i = 1:M

            # Find new equioscillation nodes.
            x_lo = isone(i) ? a : max(a, _halve(x[i-1] + x[i]))
            x_hi = (i == M) ? b : min(b, _halve(x[i] + x[i+1]))
            if objective == :absolute
                r_lo, r_hi = find_root(
                    x -> begin
                        u = muladd(x, scale, shift)
                        dpx = scale * dot(c,
                            chebyshev_derivatives(u, Val{N}()))
                        return dpx - df(x)
                    end,
                    x -> begin
                        u = muladd(x, scale, shift)
                        d2px = scale2 * dot(c,
                            chebyshev_second_derivatives(u, Val{N}()))
                        return d2px - d2f(x)
                    end,
                    x[i], x_lo, x_hi)
            else # objective == :relative
                r_lo, r_hi = find_root(
                    x -> begin
                        u = muladd(x, scale, shift)
                        px = dot(c, chebyshev_values(u, Val{N}()))
                        dpx = scale * dot(c,
                            chebyshev_derivatives(u, Val{N}()))
                        return dpx * f(x) - px * df(x)
                    end,
                    x -> begin
                        u = muladd(x, scale, shift)
                        px = dot(c, chebyshev_values(u, Val{N}()))
                        d2px = scale2 * dot(c,
                            chebyshev_second_derivatives(u, Val{N}()))
                        return d2px * f(x) - px * d2f(x)
                    end,
                    x[i], x_lo, x_hi)
            end

            # Check both endpoints of both intervals.
            x_best = nothing
            E_best = nothing
            for z in (r_lo, r_hi, x_lo, x_hi)
                pz = dot(c, chebyshev_values(
                    muladd(z, scale, shift), Val{N}()))
                if objective == :absolute
                    E = pz - f(z)
                else # objective == :relative
                    E = pz / f(z) - _one
                end
                if isnothing(E_best) || (abs(E) > abs(E_best))
                    x_best = z
                    E_best = E
                end
            end
            @assert !isnothing(x_best)
            @assert !isnothing(E_best)

            x_next[i] = x_best
            E_max = max(E_max, abs(E_best))
            max_deviation = max(max_deviation,
                abs(abs(E_best) - abs(E_nominal)))
        end

        # Terminate Remez iteration when E_best converges to E_nominal or
        # equioscillation nodes become invalid.
        if (isnothing(prev_deviation) || (max_deviation < prev_deviation)) &&
           issorted(x_next) && allunique(x_next)
            prev_deviation = max_deviation
            x, x_next = x_next, x
        else
            coefficients = _to_ntuple(B * view(v, 1:N+1), Val{N + 1}())
            nodes = _to_ntuple(x, Val{M}())
            return (coefficients, nodes, E_max)
        end

    end
end


end # module Remez
