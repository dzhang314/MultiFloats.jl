module Remez

using LinearAlgebra: ColumnNorm, dot, ldiv!, qr!

export minimax_polynomial, minimax_rational


@inline _halve(x::T) where {T} = ldexp(x, -1)
@inline _halve(x::Float64) = 0.5 * x


@inline _twice(x::T) where {T} = x + x
@inline _twice(x::BigFloat) = ldexp(x, 1)


@inline _to_ntuple(x::AbstractVector{T}, ::Val{N}) where {T,N} =
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

# Use the [ModAB method](https://iopscience.iop.org/article/10.1088/1757-899X/1276/1/012010/)
function find_root(f::F, a::T, b::T) where {F,T}
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

    x_lo = a
    x_hi = b
    f_lo = fa
    f_hi = fb
    x = x_lo
    fx = f_lo
    side = 0
    bisecting = true

    while true
        @assert x_lo <= x <= x_hi
        @assert !isnan(f_lo)
        @assert !isnan(f_hi)
        @assert !isnan(fx)
        @assert signbit(f_lo) != signbit(f_hi)
        
        if bisecting
            x = _halve(x_lo + x_hi)
            fx = f(x)
            fmid =  _halve(f_lo + f_hi)
            if 4abs(fmid - fx) < abs(fmid) + abs(fx)
                bisecting = false
            end
        else
            x = (x_lo * f_hi - x_hi * f_lo) / (f_hi - r_lo)
            fx = f(x)
        end

        if !(x_lo < x < x_hi)
            # The interval has become so small that its midpoint coincides
            # with one of its endpoints. We cannot make any more progress.
            return (x_lo, x_hi)
        end
        
        # Apply Anderson-Bjork modification
        if side == 1
            m = 1 - fx / f_lo
            f_lo *= m <= 0 ? inv(2 * one(T)) : m
        elseif side == 2
            m = 1 - fx / f_hi
            f_hi *= m <= 0 ? inv(2 * one(T)) : m
        end
        
        # Accept step and update interval endpoints.
        x = x_next
        fx = f(x)
        if signbit(fx) == signbit(f_lo)
            if !bisecting
                side  = 1
            end
            x_lo = x
            f_lo = fx
        else
            if !bisecting
                side = 2
            end
            x_hi = x
            f_hi = fx
        end
    end
end


function minimax_polynomial(
    f::F, df::G, a::T, b::T, ::Val{N};
    fixed_coefficients::NTuple{K,Pair{Int,T}}=(),
    initial_nodes=nothing,
    objective::Symbol=:absolute,
) where {F,G,H,T,N,K}

    # Validate inputs.
    @assert a <= b
    L = N - K + 2
    @assert L >= 2
    for (d, _) in fixed_coefficients
        @assert 0 <= d <= N
    end
    @assert isnothing(initial_nodes) || issorted(initial_nodes)
    if isnothing(initial_nodes)
        initial_nodes = chebyshev_nodes(a, b, Val{L}())
    end
    nodes = collect(initial_nodes)
    @assert eltype(nodes) <: T
    @assert length(nodes) == L
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
    next_nodes = Vector{T}(undef, L)
    prev_deviation = nothing
    @inbounds while true

        # Set up equioscillation equations (rows 1:L).
        for i = 1:L
            fi = f(nodes[i])
            u = muladd(nodes[i], scale, shift)
            A[i, 1:N+1] .= chebyshev_values(u, Val{N}())
            if objective == :absolute
                A[i, N+2] = ifelse(isodd(i), +_one, -_one)
            else # objective == :relative
                A[i, N+2] = ifelse(isodd(i), +fi, -fi)
            end
            v[i] = fi
        end

        # Set up constraint equations (rows L+1:N+2).
        for (i, (d, c)) in enumerate(fixed_coefficients)
            A[L+i, 1:N+1] .= view(B, d + 1, 1:N+1)
            A[L+i, N+2] = _zero
            v[L+i] = c
        end

        # Solve linear system using column-pivoted QR.
        ldiv!(qr!(A, ColumnNorm()), v)
        c = _to_ntuple(v, Val{N + 1}())
        E_nominal = v[N+2]

        E_max = _zero
        max_deviation = _zero
        for i = 1:L

            # Find new equioscillation nodes.
            x_lo = isone(i) ? a : max(a, _halve(nodes[i-1] + nodes[i]))
            x_hi = (i == L) ? b : min(b, _halve(nodes[i] + nodes[i+1]))
            if objective == :absolute
                r_lo, r_hi = find_root(
                    x -> begin
                        u = muladd(x, scale, shift)
                        dpx = scale * dot(c,
                            chebyshev_derivatives(u, Val{N}()))
                        return dpx - df(x)
                    end,
                    x_lo, x_hi)
            else # objective == :relative
                r_lo, r_hi = find_root(
                    x -> begin
                        u = muladd(x, scale, shift)
                        px = dot(c, chebyshev_values(u, Val{N}()))
                        dpx = scale * dot(c,
                            chebyshev_derivatives(u, Val{N}()))
                        return dpx * f(x) - px * df(x)
                    end,
                    x_lo, x_hi)
            end

            # Check both endpoints of both intervals.
            x_best = nothing
            E_best = nothing
            for x in (r_lo, r_hi, x_lo, x_hi)
                u = muladd(x, scale, shift)
                px = dot(c, chebyshev_values(u, Val{N}()))
                if objective == :absolute
                    E = px - f(x)
                else # objective == :relative
                    E = px / f(x) - _one
                end
                if isnothing(E_best) || (abs(E) > abs(E_best))
                    x_best = x
                    E_best = E
                end
            end
            @assert !isnothing(x_best)
            @assert !isnothing(E_best)

            next_nodes[i] = x_best
            E_max = max(E_max, abs(E_best))
            max_deviation = max(max_deviation,
                abs(abs(E_best) - abs(E_nominal)))
        end

        # Terminate Remez iteration when E_best converges to E_nominal or
        # equioscillation nodes become invalid.
        if (isnothing(prev_deviation) || (max_deviation < prev_deviation)) &&
           issorted(next_nodes) && allunique(next_nodes)
            prev_deviation = max_deviation
            nodes, next_nodes = next_nodes, nodes
        else
            coefficients = _to_ntuple(B * view(v, 1:N+1), Val{N + 1}())
            return (coefficients, _to_ntuple(nodes, Val{L}()), E_max)
        end

    end
end


function minimax_rational(
    f::F, df::G, a::T, b::T, ::Val{M}, ::Val{N};
    fixed_numerator_coefficients::NTuple{Kp,Pair{Int,T}}=(),
    fixed_denominator_coefficients::NTuple{Kq,Pair{Int,T}}=(),
    initial_nodes=nothing,
    objective::Symbol=:absolute,
) where {F,G,H,T,M,N,Kp,Kq}

    # Validate inputs.
    @assert a <= b
    L = (M + N) - (Kp + Kq) + 2
    @assert L >= 2
    for (d, _) in fixed_numerator_coefficients
        @assert 0 <= d <= M
    end
    for (d, _) in fixed_denominator_coefficients
        @assert 0 <= d <= N
    end
    @assert isnothing(initial_nodes) || issorted(initial_nodes)
    if isnothing(initial_nodes)
        initial_nodes = chebyshev_nodes(a, b, Val{L}())
    end
    nodes = collect(initial_nodes)
    @assert eltype(nodes) <: T
    @assert length(nodes) == L
    @assert (objective == :absolute) || (objective == :relative)

    _zero = zero(T)
    _one = one(T)

    # Compute scaling parameters that map [a, b] to [-1, +1].
    inv_width = inv(b - a)
    scale = _twice(inv_width)
    scale2 = scale^2
    shift = -(a + b) * inv_width

    # Allocate workspace arrays.
    A = Matrix{T}(undef, M + N + 2, M + N + 2)
    Bp = chebyshev_to_monomial_matrix(a, b, M)
    Bq = chebyshev_to_monomial_matrix(a, b, N)
    cq_prev = ntuple(j -> ifelse(isone(j), _one, _zero), Val{N + 1}())
    v = Vector{T}(undef, M + N + 2)
    next_nodes = Vector{T}(undef, L)
    prev_deviation = nothing
    @inbounds while true

        # Set up linearized equioscillation equations (rows 1:L).
        for i = 1:L
            fi = f(nodes[i])
            u = muladd(nodes[i], scale, shift)
            A[i, 1:M+1] .= chebyshev_values(u, Val{M}())
            A[i, M+2:M+N+1] .= -fi .* chebyshev_values(u, Val{N}())[2:N+1]
            qi_prev = dot(cq_prev, chebyshev_values(u, Val{N}()))
            if objective == :absolute
                A[i, M+N+2] = ifelse(isodd(i), +qi_prev, -qi_prev)
            else # objective == :relative
                A[i, M+N+2] = ifelse(isodd(i), +fi * qi_prev, -fi * qi_prev)
            end
            v[i] = fi
        end

        # Set up numerator constraint equations (rows L+1:L+Kp).
        for (i, (d, c)) in enumerate(fixed_numerator_coefficients)
            A[L+i, 1:M+1] .= view(Bp, d + 1, 1:M+1)
            A[L+i, M+2:M+N+2] .= _zero
            v[L+i] = c
        end

        # Set up denominator constraint equations (rows L+Kp+1:M+N+2).
        for (i, (d, c)) in enumerate(fixed_denominator_coefficients)
            A[L+Kp+i, 1:M+1] .= _zero
            A[L+Kp+i, M+2:M+N+1] .= view(Bq, d + 1, 2:N+1)
            A[L+Kp+i, M+N+2] = _zero
            v[L+Kp+i] = c - Bq[d+1, 1]
        end

        # Solve linear system using column-pivoted QR.
        ldiv!(qr!(A, ColumnNorm()), v)
        cp = _to_ntuple(v, Val{M + 1}())
        cq = _to_ntuple(view(v, M+1:M+N+1), Val{N + 1}())
        cq = Base.setindex(cq, _one, 1)
        E_nominal = v[M+N+2]

        E_max = _zero
        max_deviation = _zero
        for i = 1:L

            # Find new equioscillation nodes.
            x_lo = isone(i) ? a : max(a, _halve(nodes[i-1] + nodes[i]))
            x_hi = (i == L) ? b : min(b, _halve(nodes[i] + nodes[i+1]))
            if objective == :absolute
                r_lo, r_hi = find_root(
                    x -> begin
                        u = muladd(x, scale, shift)
                        px = dot(cp, chebyshev_values(u, Val{M}()))
                        qx = dot(cq, chebyshev_values(u, Val{N}()))
                        dpx = scale * dot(cp,
                            chebyshev_derivatives(u, Val{M}()))
                        dqx = scale * dot(cq,
                            chebyshev_derivatives(u, Val{N}()))
                        return dpx * qx - px * dqx - qx^2 * df(x)
                    end,
                    x_lo, x_hi)
            else # objective == :relative
                r_lo, r_hi = find_root(
                    x -> begin
                        u = muladd(x, scale, shift)
                        px = dot(cp, chebyshev_values(u, Val{M}()))
                        qx = dot(cq, chebyshev_values(u, Val{N}()))
                        dpx = scale * dot(cp,
                            chebyshev_derivatives(u, Val{M}()))
                        dqx = scale * dot(cq,
                            chebyshev_derivatives(u, Val{N}()))
                        return (dpx * qx - px * dqx) * f(x) - px * qx * df(x)
                    end,
                    x_lo, x_hi)
            end

            # Check both endpoints of both intervals.
            x_best = nothing
            E_best = nothing
            for x in (r_lo, r_hi, x_lo, x_hi)
                u = muladd(x, scale, shift)
                px = dot(cp, chebyshev_values(u, Val{M}()))
                qx = dot(cq, chebyshev_values(u, Val{N}()))
                if objective == :absolute
                    E = px / qx - f(x)
                else # objective == :relative
                    E = px / (qx * f(x)) - _one
                end
                if isnothing(E_best) || (abs(E) > abs(E_best))
                    x_best = x
                    E_best = E
                end
            end
            @assert !isnothing(x_best)
            @assert !isnothing(E_best)

            next_nodes[i] = x_best
            E_max = max(E_max, abs(E_best))
            max_deviation = max(max_deviation,
                abs(abs(E_best) - abs(E_nominal)))
        end

        # Terminate Remez iteration when E_best converges to E_nominal or
        # equioscillation nodes become invalid.
        if (isnothing(prev_deviation) || (max_deviation < prev_deviation)) &&
           issorted(next_nodes) && allunique(next_nodes)
            prev_deviation = max_deviation
            cq_prev = cq
            nodes, next_nodes = next_nodes, nodes
        else
            coefficients_p = _to_ntuple(Bp * view(v, 1:M+1), Val{M + 1}())
            v[M+1] = _one
            coefficients_q = _to_ntuple(Bq * view(v, M+1:M+N+1), Val{N + 1}())
            return (coefficients_p, coefficients_q,
                _to_ntuple(nodes, Val{L}()), E_max)
        end

    end
end


end # module Remez
