using MultiFloats
using Random: seed!
using SIMD: Vec
using Test: @test

# Type and value versions of Base.zero and Base.one coincide.
@test zero(one(Float64x4)) == zero(Float64x4)
@test one(zero(Float64x4)) == one(Float64x4)
@test all(zero(one(v8Float64x4)) == zero(v8Float64x4))
@test all(one(zero(v8Float64x4)) == one(v8Float64x4))

# Zero and one are not equal.
@test zero(Float64x4) != one(Float64x4)
@test all(zero(v8Float64x4) != one(v8Float64x4))

# All MultiFloat types should be bits types.
for T in [
    Float64x1, Float64x2, Float64x3, Float64x4,
    Float64x5, Float64x6, Float64x7, Float64x8,
    v1Float64x1, v1Float64x2, v1Float64x3, v1Float64x4,
    v1Float64x5, v1Float64x6, v1Float64x7, v1Float64x8,
    v2Float64x1, v2Float64x2, v2Float64x3, v2Float64x4,
    v2Float64x5, v2Float64x6, v2Float64x7, v2Float64x8,
    v4Float64x1, v4Float64x2, v4Float64x3, v4Float64x4,
    v4Float64x5, v4Float64x6, v4Float64x7, v4Float64x8,
    v8Float64x1, v8Float64x2, v8Float64x3, v8Float64x4,
    v8Float64x5, v8Float64x6, v8Float64x7, v8Float64x8,
]
    @test isbitstype(T)
end

@test iszero(Float64x4((0.0, 0.5, -0.25, -0.25)))
@test isone(Float64x4((1.0, 0.5, -0.25, -0.25)))
@test all(iszero(v2Float64x4((
    Vec(0.0, 0.0), Vec(0.5, 0.0), Vec(-0.25, 0.0), Vec(-0.25, 0.0)))))
@test all(isone(v2Float64x4((
    Vec(1.0, 1.0), Vec(0.5, 0.0), Vec(-0.25, 0.0), Vec(-0.25, 0.0)))))

for T in [
    Float64x1, Float64x2, Float64x3, Float64x4,
    Float64x5, Float64x6, Float64x8, #= Float64x7, =#
]
    # TODO: Why does this test fail for Float64x7?
    setprecision(BigFloat, 500) do
        x = T(2)
        y = sqrt(x)
        z = sqrt(y)
        x_big = BigFloat(2)
        y_big = sqrt(x_big)
        z_big = sqrt(y_big)
        @test abs(y - y_big) < eps(T)
        @test abs(z - z_big) < eps(T)
    end
end

for T in [
    Float64x1, Float64x2, Float64x3, Float64x4,
    Float64x5, Float64x6, Float64x7, Float64x8,
]
    seed!(0)
    for _ = 1:100
        x = rand(T)
        y = T(string(x))
        @test x._limbs == y._limbs
    end
end

let
    x = Float64x2((-floatmax(Float64), floatmin(Float64)))
    y = Float64x2(string(x))
    @test x._limbs == y._limbs
end
