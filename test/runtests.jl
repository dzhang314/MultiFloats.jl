using MultiFloats
using SIMD: Vec
using Test

# Type and value versions of Base.zero and Base.one coincide.
@test zero(one(Float64x4)) == zero(Float64x4)
@test one(zero(Float64x4)) == one(Float64x4)
@test all(zero(one(v8Float64x4)) == zero(v8Float64x4))
@test all(one(zero(v8Float64x4)) == one(v8Float64x4))

# Zero and one are not equal.
@test zero(Float64x4) != one(Float64x4)
@test all(zero(v8Float64x4) != one(v8Float64x4))

# All MultiFloat types should be bits types.
for T in [Float64x1, Float64x2, Float64x3, Float64x4,
    Float64x5, Float64x6, Float64x7, Float64x8,
    v1Float64x1, v1Float64x2, v1Float64x3, v1Float64x4,
    v1Float64x5, v1Float64x6, v1Float64x7, v1Float64x8,
    v2Float64x1, v2Float64x2, v2Float64x3, v2Float64x4,
    v2Float64x5, v2Float64x6, v2Float64x7, v2Float64x8,
    v4Float64x1, v4Float64x2, v4Float64x3, v4Float64x4,
    v4Float64x5, v4Float64x6, v4Float64x7, v4Float64x8,
    v8Float64x1, v8Float64x2, v8Float64x3, v8Float64x4,
    v8Float64x5, v8Float64x6, v8Float64x7, v8Float64x8]
    @test isbitstype(T)
end

@test iszero(Float64x4((0.0, 0.5, -0.25, -0.25)))
@test isone(Float64x4((1.0, 0.5, -0.25, -0.25)))
@test all(iszero(v2Float64x4((
    Vec(0.0, 0.0), Vec(0.5, 0.0), Vec(-0.25, 0.0), Vec(-0.25, 0.0)))))
@test all(isone(v2Float64x4((
    Vec(1.0, 1.0), Vec(0.5, 0.0), Vec(-0.25, 0.0), Vec(-0.25, 0.0)))))
