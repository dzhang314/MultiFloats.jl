@inline _exp2_min(::Type{Float32}) = Float32(-0x1.F80000p+006)
@inline _exp2_max(::Type{Float32}) = Float32(+0x1.FFFFFEp+006)
@inline _exp2_min(::Type{Float64}) = -0x1.FF00000000000p+0009
@inline _exp2_max(::Type{Float64}) = +0x1.FFFFFFFFFFFFFp+0009


@inline _exp2_coefficients(::Type{Float32}, ::Val{1}) = (
    (Float32(+0x1.000000p+000),),
    (Float32(+0x1.62E430p-001),),
    (Float32(+0x1.EBFBE0p-003),),
    (Float32(+0x1.C6BE34p-005),),
    (Float32(+0x1.3B338Cp-007),),
)

@inline _exp2_coefficients(::Type{Float32}, ::Val{2}) = (
    (Float32(+0x1.000000p+000), Float32(-0x1.62C458p-059)),
    (Float32(+0x1.62E430p-001), Float32(-0x1.05C610p-029)),
    (Float32(+0x1.EBFBE0p-003), Float32(-0x1.F4DEB0p-033)),
    (Float32(+0x1.C6B08Ep-005), Float32(-0x1.1F6B9Cp-030)),
    (Float32(+0x1.3B2AB6p-007), Float32(+0x1.DB9286p-032)),
    (Float32(+0x1.5D87FEp-010),),
    (Float32(+0x1.430E9Ep-013),),
    (Float32(+0x1.FFD486p-017),),
)

@inline _exp2_coefficients(::Type{Float32}, ::Val{3}) = (
    (Float32(+0x1.000000p+000), Float32(-0x1.C3C196p-094), Float32(-0x1.C4440Ep-119)),
    (Float32(+0x1.62E430p-001), Float32(-0x1.05C610p-029), Float32(-0x1.950D86p-054)),
    (Float32(+0x1.EBFBE0p-003), Float32(-0x1.F4E9C6p-033), Float32(+0x1.4378BCp-058)),
    (Float32(+0x1.C6B08Ep-005), Float32(-0x1.1F6BE8p-030), Float32(-0x1.D67986p-059)),
    (Float32(+0x1.3B2AB6p-007), Float32(+0x1.F749CEp-032), Float32(+0x1.CA50FEp-057)),
    (Float32(+0x1.5D87FEp-010), Float32(+0x1.E299F2p-036)),
    (Float32(+0x1.430912p-013), Float32(+0x1.F0D902p-038)),
    (Float32(+0x1.FFCBFCp-017), Float32(+0x1.382A76p-043)),
    (Float32(+0x1.62C022p-020),),
    (Float32(+0x1.B52A7Ep-024),),
    (Float32(+0x1.E4D50Ep-028),),
)

@inline _exp2_coefficients(::Type{Float32}, ::Val{4}) = (
    (Float32(+0x1.000000p+000), Float32(+0x1.314BDAp-112), Float32(-0x0.003C76p-126), Float32(+0x0.000000p+000)),
    (Float32(+0x1.62E430p-001), Float32(-0x1.05C610p-029), Float32(-0x1.950D88p-054), Float32(+0x1.D9CB66p-079)),
    (Float32(+0x1.EBFBE0p-003), Float32(-0x1.F4E9C6p-033), Float32(+0x1.4378B6p-058), Float32(-0x1.F22D28p-084)),
    (Float32(+0x1.C6B08Ep-005), Float32(-0x1.1F6BE8p-030), Float32(-0x1.D33162p-059), Float32(+0x1.39D488p-084)),
    (Float32(+0x1.3B2AB6p-007), Float32(+0x1.F749CEp-032), Float32(+0x1.CA7330p-057), Float32(-0x1.F93600p-082)),
    (Float32(+0x1.5D87FEp-010), Float32(+0x1.E299CCp-036), Float32(+0x1.102D00p-062)),
    (Float32(+0x1.430912p-013), Float32(+0x1.F0D8F0p-038), Float32(+0x1.DBC3A2p-063)),
    (Float32(+0x1.FFCBFCp-017), Float32(+0x1.622C4Ep-043), Float32(+0x1.F99E0Cp-068)),
    (Float32(+0x1.62C022p-020), Float32(+0x1.D2E43Ap-047)),
    (Float32(+0x1.B5253Ep-024), Float32(-0x1.997E54p-049)),
    (Float32(+0x1.E4CF52p-028), Float32(-0x1.5BD28Cp-053)),
    (Float32(+0x1.E8CFACp-032), Float32(-0x1.1FF800p-058)),
    (Float32(+0x1.C3C1DEp-036),),
)

@inline _exp2_coefficients(::Type{Float64}, ::Val{1}) = (
    (+0x1.0000000000000p+0000,),
    (+0x1.62E42FEFA39EFp-0001,),
    (+0x1.EBFBDFF82C58Fp-0003,),
    (+0x1.C6B08D704A259p-0005,),
    (+0x1.3B2AB6FBA4F80p-0007,),
    (+0x1.5D87FE6D1F9D9p-0010,),
    (+0x1.430912EE3E876p-0013,),
    (+0x1.FFD3AB8D38F1Fp-0017,),
    (+0x1.62C5577D34F86p-0020,),
)

@inline _exp2_coefficients(::Type{Float64}, ::Val{2}) = (
    (+0x1.0000000000000p+0000, +0x1.314BACF0323FFp-0113),
    (+0x1.62E42FEFA39EFp-0001, +0x1.ABC9E3B39803Fp-0056),
    (+0x1.EBFBDFF82C58Fp-0003, -0x1.5E43A53E454F1p-0057),
    (+0x1.C6B08D704A0C0p-0005, -0x1.D3316275139AEp-0059),
    (+0x1.3B2AB6FBA4E77p-0007, +0x1.4E65DFEF67D34p-0062),
    (+0x1.5D87FE78A6731p-0010, +0x1.0717F88815ADFp-0066),
    (+0x1.430912F86C787p-0013, +0x1.BC7CDBCDC0339p-0067),
    (+0x1.FFCBFC588B0C7p-0017, -0x1.E645E286FE571p-0071),
    (+0x1.62C0223A5C863p-0020, -0x1.99EF542AA8E1Ep-0074),
    (+0x1.B5253D395E80Fp-0024,),
    (+0x1.E4CF5152FBB30p-0028,),
    (+0x1.E8CAC72F6E9E5p-0032,),
    (+0x1.C3C1919538484p-0036,),
    (+0x1.816519F74C4AFp-0040,),
)

@inline _exp2_coefficients(::Type{Float64}, ::Val{3}) = (
    (+0x1.0000000000000p+0000, -0x1.45AE8ADE8BE00p-0171, +0x1.1A4EE3B642AEFp-0225),
    (+0x1.62E42FEFA39EFp-0001, +0x1.ABC9E3B39803Fp-0056, +0x1.7B57A079A1934p-0111),
    (+0x1.EBFBDFF82C58Fp-0003, -0x1.5E43A53E44DA3p-0057, -0x1.406AB8BB15A7Dp-0112),
    (+0x1.C6B08D704A0C0p-0005, -0x1.D331627513351p-0059, +0x1.2DEE9EB88E96Bp-0113),
    (+0x1.3B2AB6FBA4E77p-0007, +0x1.4E65DF05A9F75p-0062, +0x1.8A0E48B03BB1Fp-0116),
    (+0x1.5D87FE78A6731p-0010, +0x1.0717F69A514BFp-0066, -0x1.E67D45B2B54C0p-0121),
    (+0x1.430912F86C787p-0013, +0x1.BD2C2A261AC8Dp-0067, +0x1.FA7C10E68194Bp-0125),
    (+0x1.FFCBFC588B0C7p-0017, -0x1.E53AB8CDE09C6p-0071, -0x1.6CABBD89673B2p-0125),
    (+0x1.62C0223A5C824p-0020, -0x1.3800CFC92CEC7p-0079, +0x1.F725ACFE88614p-0133),
    (+0x1.B5253D395E7C4p-0024, -0x1.2DAC78D2D8104p-0079),
    (+0x1.E4CF5158B8ECAp-0028, -0x1.204BC43674356p-0085),
    (+0x1.E8CAC7351BB25p-0032, -0x1.F8543329B1980p-0087),
    (+0x1.C3BD650FC2986p-0036, -0x1.D55E5BE260085p-0092),
    (+0x1.816193166D0F9p-0040, +0x1.89E0D888E9E3Dp-0094),
    (+0x1.314964D5878B9p-0044, +0x1.6BCCC8ED7C9FFp-0098),
    (+0x1.C36E843B04039p-0049, +0x1.4569838910831p-0105),
    (+0x1.38E89AE5EF001p-0053,),
    (+0x1.98444B3F935E3p-0058,),
    (+0x1.F71A9A0DA0F0Ep-0063,),
    (+0x1.25A9C5B4980F7p-0067,),
)

@inline _exp2_coefficients(::Type{Float64}, ::Val{4}) = (
    (+0x1.0000000000000p+0000, +0x1.D3EDD82C8CCC3p-0231, -0x1.FCE0410A40696p-0286, -0x1.83E7CCE217359p-0340),
    (+0x1.62E42FEFA39EFp-0001, +0x1.ABC9E3B39803Fp-0056, +0x1.7B57A079A1934p-0111, -0x1.ACE93A4EBE5ECp-0165),
    (+0x1.EBFBDFF82C58Fp-0003, -0x1.5E43A53E44DA3p-0057, -0x1.406AB8BB15C7Ap-0112, +0x1.9CD3A9857D230p-0168),
    (+0x1.C6B08D704A0C0p-0005, -0x1.D331627513351p-0059, +0x1.2DEE9EB88E88Ap-0113, -0x1.2FF778B5F48F7p-0167),
    (+0x1.3B2AB6FBA4E77p-0007, +0x1.4E65DF05A9F75p-0062, +0x1.8A0E48F1D4A7Dp-0116, -0x1.C6EE295EFFA51p-0171),
    (+0x1.5D87FE78A6731p-0010, +0x1.0717F69A514BFp-0066, -0x1.E67D449ACB48Cp-0121, -0x1.CD71FE75E0E48p-0175),
    (+0x1.430912F86C787p-0013, +0x1.BD2C2A261AC8Dp-0067, +0x1.F3ECC53FF3312p-0125, -0x1.164337514743Dp-0181),
    (+0x1.FFCBFC588B0C7p-0017, -0x1.E53AB8CDE09C6p-0071, -0x1.6D4C7BAB45DE9p-0125, -0x1.9E19679D56D7Ep-0182),
    (+0x1.62C0223A5C824p-0020, -0x1.3800CFC92C41Ep-0079, +0x1.61B8683DC437Ep-0133, +0x1.9F2865517954Cp-0188),
    (+0x1.B5253D395E7C4p-0024, -0x1.2DAC78D2D8038p-0079, +0x1.14632E2950C3Dp-0133),
    (+0x1.E4CF5158B8ECAp-0028, -0x1.204BC4D5A312Dp-0085, +0x1.C540C0C34226Dp-0140),
    (+0x1.E8CAC7351BB25p-0032, -0x1.F8543350DC6F6p-0087, +0x1.3BA02AC151168p-0141),
    (+0x1.C3BD650FC2986p-0036, -0x1.D4A9781E85D12p-0092, +0x1.878EF706B383Bp-0146),
    (+0x1.816193166D0F9p-0040, +0x1.8A06B0C03DC5Ap-0094, -0x1.15AB03421C584p-0148),
    (+0x1.314964D5878A9p-0044, +0x1.CFC50D89572ADp-0098, -0x1.F016CF6E85F37p-0152),
    (+0x1.C36E843B04022p-0049, -0x1.984EA7FA0FCEDp-0105, +0x1.48371AFAC73B6p-0159),
    (+0x1.38E89AE79F8B4p-0053, -0x1.B71C06C164314p-0107),
    (+0x1.98444B41C25A8p-0058, -0x1.550F0CEFFFEF1p-0113),
    (+0x1.F7176BDB43696p-0063, -0x1.468CE03F0CD83p-0118),
    (+0x1.25A7ECB835C6Cp-0067, +0x1.6D0D8914C9ADFp-0122),
    (+0x1.45ACC4B50513Ep-0072, -0x1.BC8D95D5A48E6p-0126),
    (+0x1.57FC5782BA01Ep-0077, -0x1.E1A633F20B03Ap-0132),
    (+0x1.5ACFAD35DAA9Ep-0082,),
    (+0x1.4E76C2600CB49p-0087,),
    (+0x1.351C3A3C99042p-0092,),
)

@generated _exp2_polynomial(x::NTuple{N,T}) where {N,T} =
    _horner_expr_mf(_exp2_coefficients(T, Val{N}()))
@generated _exp2_polynomial(x::NTuple{N,Vec{M,T}}) where {N,M,T} =
    _horner_expr_mfv(_exp2_coefficients(T, Val{N}()), M)


@inline _float_to_int(x::Float32) = unsafe_trunc(Int32, x)
@inline _float_to_int(x::Float64) = unsafe_trunc(Int64, x)
@inline _float_to_int(x::Vec{M,Float32}) where {M} = convert(Vec{M,Int32}, x)
@inline _float_to_int(x::Vec{M,Float64}) where {M} = convert(Vec{M,Int64}, x)


@inline function _exp2_kernel(x::NTuple{N,T}) where {N,T}
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _four = _two + _two
    _eight = _four + _four
    _half = inv(_two)
    _one_eighth = inv(_eight)

    n_float = trunc(first(x) + copysign(_half, first(x)))
    neg_n = ntuple(i -> isone(i) ? -n_float : _zero, Val{N}())
    p = _exp2_polynomial(scale(_one_eighth, mfadd(x, neg_n, Val{N}())))
    result = mfsqr(mfsqr(mfsqr(p, Val{N}()), Val{N}()), Val{N}())
    n = _float_to_int(n_float)
    half_n = n >> 1
    result = scale(unsafe_ldexp(_one, half_n), result)
    result = scale(unsafe_ldexp(_one, n - half_n), result)
    return result
end


function Base.exp2(x::_MF{T,N}) where {T,N}
    head = first(x._limbs)
    result = _MF{T,N}(_exp2_kernel(x._limbs))
    result = ifelse(head < _exp2_min(T), zero(_MF{T,N}), result)
    result = ifelse(head > _exp2_max(T), typemax(_MF{T,N}), result)
    return result
end

function Base.exp2(x::_MFV{M,T,N}) where {M,T,N}
    head = first(x._limbs)
    result = _MFV{M,T,N}(_exp2_kernel(x._limbs))
    result = vifelse(head < _exp2_min(T), zero(_MFV{M,T,N}), result)
    result = vifelse(head > _exp2_max(T), typemax(_MFV{M,T,N}), result)
    return result
end


const _LOG2_E_FULL_F32 = (
    Float32(+0x1.715476p+000), Float32(+0x1.4AE0C0p-026),
    Float32(-0x1.E88830p-052), Float32(+0x1.FFB41Ap-077),
    Float32(+0x1.1D3E88p-103), Float32(+0x0.75ABBEp-126),
)

const _LOG2_E_FULL_F64 = (
    +0x1.71547652B82FEp+0000, +0x1.777D0FFDA0D24p-0056,
    -0x1.60BB8A5442AB9p-0110, -0x1.4B52D3BA6D74Dp-0166,
    +0x1.9A342648FBC39p-0220, -0x1.E0455744994EEp-0274,
    +0x1.B25EEB82D7C16p-0328, +0x1.F5485CF306255p-0382,
    -0x1.EC07680A1F958p-0436, -0x1.06326680EB5B6p-0490,
    -0x1.B3D04C549BC98p-0544, +0x1.EABCEAD10305Bp-0598,
    -0x1.4440C57D7AB97p-0655, -0x1.7185D42A4E6D6p-0710,
    -0x1.F332B5BE48526p-0766, +0x1.2CE4F199E108Dp-0820,
    -0x1.8DAFCC6077F2Ap-0877, +0x1.9ABB71EC25E12p-0932,
    -0x1.1473D7A3366BDp-0989, -0x0.000004977D38Ap-1022,
)

@inline _log2_e(::Type{_MF{Float32,N}}) where {N} =
    _MF{Float32,N}(ntuple(i -> _LOG2_E_FULL_F32[i], Val{N}()))
@inline _log2_e(::Type{_MF{Float64,N}}) where {N} =
    _MF{Float64,N}(ntuple(i -> _LOG2_E_FULL_F64[i], Val{N}()))
@inline _log2_e(::Type{_MFV{M,T,N}}) where {M,T,N} =
    _MFV{M,T,N}(_log2_e(_MF{T,N}))


const _LOG2_10_FULL_F32 = (
    Float32(+0x1.A934F0p+001), Float32(+0x1.2F346Ep-024),
    Float32(+0x1.5FC926p-051), Float32(-0x1.02402Cp-076),
    Float32(-0x1.28125Ap-101), Float32(+0x0.D96C56p-126),
)

const _LOG2_10_FULL_F64 = (
    +0x1.A934F0979A371p+0001, +0x1.7F2495FB7FA6Dp-0053,
    +0x1.FB699B2D8ABFCp-0107, +0x1.BD9D6A748DB56p-0161,
    +0x1.0105CF0B3A0CDp-0215, -0x1.7810B1157062Ep-0269,
    -0x1.DB41549C52F43p-0326, +0x1.850D7B9201597p-0380,
    +0x1.4A99FE2C59A9Fp-0435, -0x1.FCC7E09B0693Fp-0490,
    +0x1.F66D7537FB9B1p-0544, -0x1.947073642FC7Ap-0598,
    -0x1.A06FE33D2E537p-0654, -0x1.708A88AEF6ACFp-0710,
    -0x1.1E69CE4805704p-0764, -0x1.28ECFB0774BC1p-0818,
    -0x1.351E4155B270Cp-0872, -0x1.8F99DCECA58EDp-0928,
    -0x1.EFBB0F04F4A22p-0985, -0x0.00008D3C165ABp-1022,
)

@inline _log2_10(::Type{_MF{Float32,N}}) where {N} =
    _MF{Float32,N}(ntuple(i -> _LOG2_10_FULL_F32[i], Val{N}()))
@inline _log2_10(::Type{_MF{Float64,N}}) where {N} =
    _MF{Float64,N}(ntuple(i -> _LOG2_10_FULL_F64[i], Val{N}()))
@inline _log2_10(::Type{_MFV{M,T,N}}) where {M,T,N} =
    _MFV{M,T,N}(_log2_10(_MF{T,N}))


Base.exp(x::_MF{T,N}) where {T,N} = exp2(x * _log2_e(_MF{T,N}))
Base.exp(x::_MFV{M,T,N}) where {M,T,N} = exp2(x * _log2_e(_MFV{M,T,N}))
Base.exp10(x::_MF{T,N}) where {T,N} = exp2(x * _log2_10(_MF{T,N}))
Base.exp10(x::_MFV{M,T,N}) where {M,T,N} = exp2(x * _log2_10(_MFV{M,T,N}))
