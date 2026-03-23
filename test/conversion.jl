function test_construction(
    ::Type{MultiFloat{T,N}},
    ::Type{I},
    num_trials::Int,
) where {T,N,I<:Integer}
    setprecision(BigFloat, 2 * MultiFloats._full_precision(T) + 1) do
        setrounding(BigFloat, RoundNearest) do
            for _ = 1:num_trials
                x = rand(I)
                num_allocations = @allocated begin
                    y = MultiFloat{T,N}(x)
                end
                @test num_allocations == 0
                @test MultiFloats.isnormalized(y)
                @test MultiFloats.iscanonical(y)

                if isfinite(y)
                    big_x = BigFloat(x)
                    big_y = BigFloat(y)
                    big_prev_y = BigFloat(prevfloat(y))
                    big_next_y = BigFloat(nextfloat(y))
                    @test big_prev_y < big_y < big_next_y

                    center_diff = abs(big_y - big_x)
                    prev_diff = abs(big_prev_y - big_x)
                    next_diff = abs(big_next_y - big_x)
                    prev_cmp = cmp(center_diff, prev_diff)
                    next_cmp = cmp(center_diff, next_diff)
                    @test ((prev_cmp <= 0) & (next_cmp < 0)) |
                          ((prev_cmp < 0) & (next_cmp <= 0))
                end

                num_allocations = @allocated begin
                    v1 = MultiFloatVec{4,T,N}(x)
                    v2 = MultiFloatVec{4,T,N}(x, x, x, x)
                    v3 = MultiFloatVec{4,T,N}((x, x, x, x))
                    w1 = MultiFloatVec{4,T,N}(y)
                    w2 = MultiFloatVec{4,T,N}(y, y, y, y)
                    w3 = MultiFloatVec{4,T,N}((y, y, y, y))
                end
                @test iszero(num_allocations)
                @test v1 === v2 === v3 === w1 === w2 === w3
            end
        end
    end
end


@testset "construction from integers" begin
    for T in _MF_TYPES, I in _FIXED_INT_TYPES
        test_construction(T, I, 2^12)
    end
end


function test_construction(
    ::Type{MultiFloat{T,N}},
    ::Type{F},
    num_trials::Int,
) where {T,N,F<:AbstractFloat}
    setprecision(BigFloat, 2 * MultiFloats._full_precision(T) + 1) do
        setrounding(BigFloat, RoundNearest) do
            big_max = BigFloat(floatmax(MultiFloat{T,N}))
            for _ = 1:num_trials
                x = _bit_rand(F)
                num_allocations = @allocated begin
                    y = MultiFloat{T,N}(x)
                end
                @test iszero(num_allocations)

                z = MultiFloats._clear_signed_zeros(y)
                @test MultiFloats.isnormalized(z)
                @test MultiFloats.iscanonical(z)

                big_x = BigFloat(x)
                if isfinite(y)
                    big_y = BigFloat(y)
                    big_prev_y = BigFloat(prevfloat(y))
                    big_next_y = BigFloat(nextfloat(y))
                    @test big_prev_y < big_y < big_next_y

                    center_diff = abs(big_y - big_x)
                    prev_diff = abs(big_prev_y - big_x)
                    next_diff = abs(big_next_y - big_x)
                    prev_cmp = cmp(center_diff, prev_diff)
                    next_cmp = cmp(center_diff, next_diff)
                    @test ((prev_cmp <= 0) & (next_cmp < 0)) |
                          ((prev_cmp < 0) & (next_cmp <= 0))
                else
                    @test !isnan(y)
                    if signbit(y)
                        @test big_x < -big_max
                    else
                        @test big_x > +big_max
                    end
                end

                num_allocations = @allocated begin
                    v1 = MultiFloatVec{4,T,N}(x)
                    v2 = MultiFloatVec{4,T,N}(x, x, x, x)
                    v3 = MultiFloatVec{4,T,N}((x, x, x, x))
                    w1 = MultiFloatVec{4,T,N}(y)
                    w2 = MultiFloatVec{4,T,N}(y, y, y, y)
                    w3 = MultiFloatVec{4,T,N}((y, y, y, y))
                end
                @test iszero(num_allocations)
                @test v1 === v2 === v3 === w1 === w2 === w3
            end
        end
    end
end


@testset "construction from floating-point numbers" begin
    for T in _MF_TYPES, F in _FIXED_FLOAT_TYPES
        test_construction(T, F, 2^16)
    end
end


function test_prev_next(x::MultiFloat{T,N}, k::Int) where {T,N}
    y = x
    if k > 0
        for _ = 1:k
            y = prevfloat(y)
        end
        for _ = 1:k
            y = nextfloat(y)
        end
    elseif k < 0
        for _ = k:-1
            y = nextfloat(y)
        end
        for _ = k:-1
            y = prevfloat(y)
        end
    end
    @test (y == x) || (y == MultiFloats.canonize(x))
end


@testset "prevfloat/nextfloat round-trip" begin
    for T in _MF_TYPES
        for _ = 1:4096
            test_prev_next(_bit_rand(T), rand(-256:+256))
        end
        for k = -4:+4
            test_prev_next(+floatmin(T), k)
            test_prev_next(-floatmin(T), k)
            if k >= -1
                test_prev_next(+floatmax(T), k)
            end
            if k <= +1
                test_prev_next(-floatmax(T), k)
            end
        end
    end
end


function test_string_round_trip(x::MultiFloat{T,N}) where {T,N}
    y = MultiFloat{T,N}(string(x))
    @test (y == x) || (y == MultiFloats.canonize(x))
end


function test_string_round_trip(x::MultiFloatVec{M,T,N}) where {M,T,N}
    s = string(x)
    i = only(findall(==('['), s))
    j = only(findall(==(']'), s))
    y = MultiFloatVec{M,T,N}(MultiFloat{T,N}.(split(s[i+1:j-1], ", "))...)
    @test all((y == x) | (y == MultiFloats.canonize(x)))
end


@testset "scalar string conversion round-trip" begin
    for T in _MF_TYPES
        for _ = 1:1024
            test_string_round_trip(_bit_rand(T))
        end
        test_string_round_trip(+floatmin(T))
        test_string_round_trip(-floatmin(T))
        test_string_round_trip(+floatmax(T))
        test_string_round_trip(-floatmax(T))
    end
end


@testset "vector string conversion round-trip" begin
    for T in _MFV_TYPES
        for _ = 1:64
            test_string_round_trip(_bit_rand(T))
        end
    end
end
