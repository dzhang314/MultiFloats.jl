using MultiFloats
using Test: @test

for T in [
    Float64x1, Float64x2, Float64x3, Float64x4,
    PreciseFloat64x1, PreciseFloat64x2, PreciseFloat64x3, PreciseFloat64x4,
    Vec1Float64x1, Vec1Float64x2, Vec1Float64x3, Vec1Float64x4,
    Vec2Float64x1, Vec2Float64x2, Vec2Float64x3, Vec2Float64x4,
    Vec4Float64x1, Vec4Float64x2, Vec4Float64x3, Vec4Float64x4,
    Vec8Float64x1, Vec8Float64x2, Vec8Float64x3, Vec8Float64x4,
    Vec1PreciseFloat64x1, Vec1PreciseFloat64x2,
    Vec1PreciseFloat64x3, Vec1PreciseFloat64x4,
    Vec2PreciseFloat64x1, Vec2PreciseFloat64x2,
    Vec2PreciseFloat64x3, Vec2PreciseFloat64x4,
    Vec4PreciseFloat64x1, Vec4PreciseFloat64x2,
    Vec4PreciseFloat64x3, Vec4PreciseFloat64x4,
    Vec8PreciseFloat64x1, Vec8PreciseFloat64x2,
    Vec8PreciseFloat64x3, Vec8PreciseFloat64x4,
]

    # All MultiFloat types should be bits types.
    @test isbitstype(T)

    # # Zero and one are not equal.
    # @test zero(T) != one(T)
    # @test all(zero(T) != one(T))

    # Type and value versions of Base.zero and Base.one coincide.
    @test zero(one(T)) === zero(T)
    @test one(zero(T)) === one(T)
    @test all(zero(one(T)) === zero(T))
    @test all(one(zero(T)) === one(T))
end

# # Test accuracy of sqrt(2) and sqrt(sqrt(2)).
# for T in [
#     Float64x1, Float64x2, Float64x3, Float64x4,
#     Float64x5, Float64x6, Float64x7, Float64x8,
# ]
#     setprecision(BigFloat, 500) do
#         x = T(2)
#         y = sqrt(x)
#         z = sqrt(y)
#         x_big = BigFloat(2)
#         y_big = sqrt(x_big)
#         z_big = sqrt(y_big)
#         @test abs(y - y_big) < eps(T)
#         @test abs(z - z_big) < eps(T)
#     end
# end

# # Test round-trip string conversion on random numbers.
# for T in [
#     Float64x1, Float64x2, Float64x3, Float64x4,
#     Float64x5, Float64x6, Float64x7, Float64x8,
# ]
#     seed!(0)
#     for _ = 1:100
#         x = rand(T)
#         y = T(string(x))
#         @test x._limbs == y._limbs
#     end
# end

# # Test round-trip string conversion when limbs have very different magnitudes.
# let
#     x = Float64x2((-floatmax(Float64), floatmin(Float64)))
#     y = Float64x2(string(x))
#     @test x._limbs == y._limbs
# end
