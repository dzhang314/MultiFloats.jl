@inline _log2_kernel_coefficients_narrow(::Type{Float32}, ::Val{1}) = (
    (Float32(+0x1.715476p+001),),
    (Float32(+0x1.EC752Ap-001),),
)

@inline _log2_kernel_coefficients_narrow(::Type{Float32}, ::Val{2}) = (
    (Float32(+0x1.715476p+001), Float32(+0x1.4AE0C0p-025)),
    (Float32(+0x1.EC709Ep-001), Float32(-0x1.E2FDB2p-028)),
    (Float32(+0x1.2776C6p-001),),
    (Float32(+0x1.A6217Cp-002),),
)

@inline _log2_kernel_coefficients_narrow(::Type{Float32}, ::Val{3}) = (
    (Float32(+0x1.715476p+001), Float32(+0x1.4AE0C0p-025), Float32(-0x1.E8882Ep-051)),
    (Float32(+0x1.EC709Ep-001), Float32(-0x1.E2FE02p-028), Float32(+0x1.73DA8Cp-053)),
    (Float32(+0x1.2776C6p-001), Float32(-0x1.E20C4Ep-026)),
    (Float32(+0x1.A61762p-002), Float32(+0x1.0827C8p-027)),
    (Float32(+0x1.485568p-002),),
)

@inline _log2_kernel_coefficients_narrow(::Type{Float32}, ::Val{4}) = (
    (Float32(+0x1.715476p+001), Float32(+0x1.4AE0C0p-025), Float32(-0x1.E88830p-051), Float32(+0x1.FFB41Ap-076)),
    (Float32(+0x1.EC709Ep-001), Float32(-0x1.E2FE02p-028), Float32(+0x1.749FC2p-053), Float32(-0x1.55BCAEp-078)),
    (Float32(+0x1.2776C6p-001), Float32(-0x1.E20C80p-026), Float32(-0x1.86D354p-053)),
    (Float32(+0x1.A61762p-002), Float32(+0x1.4F5BDCp-027), Float32(-0x1.B37950p-052)),
    (Float32(+0x1.484B14p-002), Float32(-0x1.41F9DAp-029)),
    (Float32(+0x1.0C9A84p-002), Float32(+0x1.7F3E5Ap-028)),
    (Float32(+0x1.C6A48Ep-003),),
)

@inline _log2_kernel_coefficients_narrow(::Type{Float64}, ::Val{1}) = (
    (+0x1.71547652B82FEp+0001,),
    (+0x1.EC709DC3A049Ap-0001,),
    (+0x1.2776C5028B217p-0001,),
    (+0x1.A6217C9F8B113p-0002,),
)

@inline _log2_kernel_coefficients_narrow(::Type{Float64}, ::Val{2}) = (
    (+0x1.71547652B82FEp+0001, +0x1.777D0FFDA0D24p-0055),
    (+0x1.EC709DC3A03FDp-0001, +0x1.D27F055486A5Ap-0055),
    (+0x1.2776C50EF9BFEp-0001, +0x1.E4B2AE3DCBD87p-0055),
    (+0x1.A61762A7ADED9p-0002, +0x1.90D60ECEBE07Fp-0057),
    (+0x1.484B13D7C0C4Dp-0002,),
    (+0x1.0C9A845FCF968p-0002,),
    (+0x1.C6A48D52BA6C7p-0003,),
)

@inline _log2_kernel_coefficients_narrow(::Type{Float64}, ::Val{3}) = (
    (+0x1.71547652B82FEp+0001, +0x1.777D0FFDA0D24p-0055, -0x1.60BB8A5442AB9p-0109),
    (+0x1.EC709DC3A03FDp-0001, +0x1.D27F05548AF0Cp-0055, -0x1.D64F631B0395Fp-0111),
    (+0x1.2776C50EF9BFEp-0001, +0x1.E4B29CCC535D4p-0055, -0x1.1A2FA19B026E1p-0111),
    (+0x1.A61762A7ADED9p-0002, +0x1.FB22E490EE2F0p-0057, -0x1.952A567E4FAF9p-0112),
    (+0x1.484B13D7C02A9p-0002, -0x1.E55FC724E0606p-0060),
    (+0x1.0C9A84994022Dp-0002, +0x1.42B91D0C93E21p-0057),
    (+0x1.C68F568D31760p-0003, +0x1.CE40D24D3E567p-0059),
    (+0x1.89F3B1694CFFDp-0003, +0x1.A670A8D35CE77p-0059),
    (+0x1.5B9AC9B74823Dp-0003,),
    (+0x1.3703C129714EEp-0003,),
    (+0x1.197AAA771958Bp-0003,),
)

@inline _log2_kernel_coefficients_narrow(::Type{Float64}, ::Val{4}) = (
    (+0x1.71547652B82FEp+0001, +0x1.777D0FFDA0D24p-0055, -0x1.60BB8A5442AB9p-0109, -0x1.4B52D3BA6D74Dp-0165),
    (+0x1.EC709DC3A03FDp-0001, +0x1.D27F05548AF0Cp-0055, -0x1.D64F631B038F7p-0111, +0x1.CDC8C82E65AE0p-0166),
    (+0x1.2776C50EF9BFEp-0001, +0x1.E4B29CCC535D4p-0055, -0x1.1A2FA1DD0222Ep-0111, +0x1.5755B4C7027FCp-0165),
    (+0x1.A61762A7ADED9p-0002, +0x1.FB22E490EE2F0p-0057, -0x1.931F7984DE7AFp-0112, +0x1.DF0BDCB650232p-0166),
    (+0x1.484B13D7C02A9p-0002, -0x1.E55FC724E0A2Cp-0060, +0x1.BCB13433C4F05p-0116),
    (+0x1.0C9A84994022Dp-0002, +0x1.42B91D166906Ap-0057, +0x1.CF5382BACBA41p-0114),
    (+0x1.C68F568D31760p-0003, +0x1.CE23C4E96378Ep-0059, +0x1.D0F3A64AB98B9p-0114),
    (+0x1.89F3B1694CFFEp-0003, -0x1.2339777C83672p-0060, +0x1.64B7B47AADC01p-0115),
    (+0x1.5B9AC9B743F0Dp-0003, +0x1.0D0E5DDA4CD13p-0057),
    (+0x1.3703C1F4D0FFEp-0003, +0x1.9288A3C5C0D3Dp-0057),
    (+0x1.1964EC6FC948Fp-0003, -0x1.BB1CD233136E6p-0057),
    (+0x1.00ECD7E087C6Dp-0003, +0x1.357017D596B58p-0057),
    (+0x1.D8BE05FBB38B0p-0004,),
    (+0x1.B5E5550A951D8p-0004,),
)

@generated _log2_kernel_narrow(x::NTuple{N,T}) where {N,T} =
    _horner_expr_mf(_log2_kernel_coefficients_narrow(T, Val{N}()))
@generated _log2_kernel_narrow(x::NTuple{N,Vec{M,T}}) where {N,M,T} =
    _horner_expr_mfv(_log2_kernel_coefficients_narrow(T, Val{N}()), M)


@inline _log2_kernel_coefficients_wide(::Type{Float32}, ::Val{1}) = (
    (Float32(+0x1.715476p+001),),
    (Float32(+0x1.EC7096p-001),),
    (Float32(+0x1.27CB2Ep-001),),
)

@inline _log2_kernel_coefficients_wide(::Type{Float32}, ::Val{2}) = (
    (Float32(+0x1.715476p+001), Float32(+0x1.4AE0C0p-025)),
    (Float32(+0x1.EC709Ep-001), Float32(-0x1.E2FE88p-028)),
    (Float32(+0x1.2776C6p-001), Float32(-0x1.DE12AAp-026)),
    (Float32(+0x1.A61738p-002),),
    (Float32(+0x1.48FE3Ap-002),),
)

@inline _log2_kernel_coefficients_wide(::Type{Float32}, ::Val{3}) = (
    (Float32(+0x1.715476p+001), Float32(+0x1.4AE0C0p-025), Float32(-0x1.E8882Ep-051)),
    (Float32(+0x1.EC709Ep-001), Float32(-0x1.E2FE02p-028), Float32(+0x1.749E06p-053)),
    (Float32(+0x1.2776C6p-001), Float32(-0x1.E20C80p-026), Float32(-0x1.1EE414p-053)),
    (Float32(+0x1.A61762p-002), Float32(+0x1.4F5992p-027)),
    (Float32(+0x1.484B14p-002), Float32(-0x1.BFE938p-030)),
    (Float32(+0x1.0C9A42p-002),),
    (Float32(+0x1.C7FF60p-003),),
)

@inline _log2_kernel_coefficients_wide(::Type{Float32}, ::Val{4}) = (
    (Float32(+0x1.715476p+001), Float32(+0x1.4AE0C0p-025), Float32(-0x1.E88830p-051), Float32(+0x1.FFB41Ap-076)),
    (Float32(+0x1.EC709Ep-001), Float32(-0x1.E2FE02p-028), Float32(+0x1.749FC2p-053), Float32(-0x1.55BFAAp-078)),
    (Float32(+0x1.2776C6p-001), Float32(-0x1.E20C80p-026), Float32(-0x1.86D358p-053), Float32(+0x1.A0521Cp-080)),
    (Float32(+0x1.A61762p-002), Float32(+0x1.4F5BDCp-027), Float32(-0x1.B03A40p-052)),
    (Float32(+0x1.484B14p-002), Float32(-0x1.41FEA8p-029), Float32(-0x1.325050p-054)),
    (Float32(+0x1.0C9A84p-002), Float32(+0x1.32785Ep-027)),
    (Float32(+0x1.C68F56p-003), Float32(+0x1.7F1516p-028)),
    (Float32(+0x1.89F2F6p-003), Float32(+0x1.226E66p-028)),
    (Float32(+0x1.5D108Cp-003),),
)

@inline _log2_kernel_coefficients_wide(::Type{Float64}, ::Val{1}) = (
    (+0x1.71547652B82FEp+0001,),
    (+0x1.EC709DC3A02EEp-0001,),
    (+0x1.2776C510F6AA9p-0001,),
    (+0x1.A61738E04AEC5p-0002,),
    (+0x1.48FE397189FCBp-0002,),
)

@inline _log2_kernel_coefficients_wide(::Type{Float64}, ::Val{2}) = (
    (+0x1.71547652B82FEp+0001, +0x1.777D0FFDA0D24p-0055),
    (+0x1.EC709DC3A03FDp-0001, +0x1.D27F055480ABFp-0055),
    (+0x1.2776C50EF9BFEp-0001, +0x1.E4B2A0D0290E2p-0055),
    (+0x1.A61762A7ADED9p-0002, +0x1.F8B804528988Ap-0057),
    (+0x1.484B13D7C02AFp-0002, -0x1.9282874496178p-0057),
    (+0x1.0C9A84993C2EEp-0002, +0x1.0E0FE2C5542A6p-0056),
    (+0x1.C68F56BF8A8ACp-0003,),
    (+0x1.89F2F69137330p-0003,),
    (+0x1.5D108CD7E21EBp-0003,),
)

@inline _log2_kernel_coefficients_wide(::Type{Float64}, ::Val{3}) = (
    (+0x1.71547652B82FEp+0001, +0x1.777D0FFDA0D24p-0055, -0x1.60BB8A5442AB9p-0109),
    (+0x1.EC709DC3A03FDp-0001, +0x1.D27F05548AF0Cp-0055, -0x1.D64F631B0228Bp-0111),
    (+0x1.2776C50EF9BFEp-0001, +0x1.E4B29CCC535D4p-0055, -0x1.1A2FA332EBECDp-0111),
    (+0x1.A61762A7ADED9p-0002, +0x1.FB22E490EE2F0p-0057, -0x1.921ECAA6DE0F6p-0112),
    (+0x1.484B13D7C02A9p-0002, -0x1.E55FC724E0A5Ep-0060, -0x1.4D296361EE5D5p-0115),
    (+0x1.0C9A84994022Dp-0002, +0x1.42B91D1674D1Cp-0057, +0x1.BFAAFC67A69EAp-0111),
    (+0x1.C68F568D31760p-0003, +0x1.CE23C153D8C0Dp-0059),
    (+0x1.89F3B1694CFFEp-0003, -0x1.22DADA4E49DAEp-0060),
    (+0x1.5B9AC9B743F0Dp-0003, +0x1.409A169E103EFp-0058),
    (+0x1.3703C1F4D10AEp-0003, +0x1.E0F8A9C7672CAp-0057),
    (+0x1.1964EC6F974A6p-0003, +0x1.87BB3B29D9C92p-0059),
    (+0x1.00ECD87C6E1D0p-0003,),
    (+0x1.D8BB8C4186C27p-0004,),
    (+0x1.B8B44488476ADp-0004,),
)

@inline _log2_kernel_coefficients_wide(::Type{Float64}, ::Val{4}) = (
    (+0x1.71547652B82FEp+0001, +0x1.777D0FFDA0D24p-0055, -0x1.60BB8A5442AB9p-0109, -0x1.4B52D3BA6D74Ep-0165),
    (+0x1.EC709DC3A03FDp-0001, +0x1.D27F05548AF0Cp-0055, -0x1.D64F631B038F7p-0111, +0x1.CDC8C82F8DEE2p-0166),
    (+0x1.2776C50EF9BFEp-0001, +0x1.E4B29CCC535D4p-0055, -0x1.1A2FA1DD0222Ep-0111, +0x1.5754E8A9EF2D5p-0165),
    (+0x1.A61762A7ADED9p-0002, +0x1.FB22E490EE2F0p-0057, -0x1.931F7984DE7AFp-0112, +0x1.5C26D7DA85B05p-0166),
    (+0x1.484B13D7C02A9p-0002, -0x1.E55FC724E0A2Cp-0060, +0x1.BCB1343516E67p-0116, -0x1.810ABC36681F3p-0170),
    (+0x1.0C9A84994022Dp-0002, +0x1.42B91D166906Ap-0057, +0x1.CF52CBE7F6AA7p-0114, -0x1.A714A214E4B34p-0168),
    (+0x1.C68F568D31760p-0003, +0x1.CE23C4E96378Ep-0059, +0x1.617A7724338F3p-0113),
    (+0x1.89F3B1694CFFEp-0003, -0x1.2339777C86C80p-0060, +0x1.DE619D15AD599p-0115),
    (+0x1.5B9AC9B743F0Dp-0003, +0x1.0D0E5E1D8F1F3p-0057, +0x1.E48E4DFE09178p-0112),
    (+0x1.3703C1F4D0FFEp-0003, +0x1.926B2BCB44368p-0057, +0x1.15B2304660DB5p-0111),
    (+0x1.1964EC6FC9491p-0003, -0x1.58938781F92FFp-0058, +0x1.F0514DB5F4CA4p-0113),
    (+0x1.00ECD7E080215p-0003, -0x1.0F3CF9689A4DFp-0059),
    (+0x1.D8BE0817F5FFCp-0004, +0x1.39485EC2CBD66p-0058),
    (+0x1.B5B96FCA55E15p-0004, +0x1.4DA502E510D97p-0060),
    (+0x1.9789566BEE621p-0004, +0x1.3285F7B71F1DFp-0058),
    (+0x1.7D3E6BD55BCBAp-0004, +0x1.F2C9826FF7FBAp-0058),
    (+0x1.66200C820B395p-0004,),
    (+0x1.54ADD42A59755p-0004,),
)

@generated _log2_kernel_wide(x::NTuple{N,T}) where {N,T} =
    _horner_expr_mf(_log2_kernel_coefficients_wide(T, Val{N}()))
@generated _log2_kernel_wide(x::NTuple{N,Vec{M,T}}) where {N,M,T} =
    _horner_expr_mfv(_log2_kernel_coefficients_wide(T, Val{N}()), M)


@generated function _log2_table(::Type{T}, ::Val{N}) where {T,N}
    setprecision(BigFloat, 2 * _full_precision(T) + 1) do
        setrounding(BigFloat, RoundNearest) do
            centers = ntuple(
                i -> _MF{T,N}(1 + (2 * i - 1) // 64),
                Val{32}())
            values = ntuple(
                i -> _MF{T,N}(log2(BigFloat(1 + (2 * i - 1) // 64))),
                Val{32}())
            return :((centers=$centers, values=$values))
        end
    end
end


@inline function _log2_table_index(x::T) where {T<:Base.IEEEFloat}
    U = Base.uinttype(T)
    bits = reinterpret(U, x)
    return (bits >> (Base.significand_bits(T) - 5)) & U(0x1F)
end

@inline function _log2_table_index(x::Vec{M,T}) where {M,T<:Base.IEEEFloat}
    U = Base.uinttype(T)
    bits = reinterpret(Vec{M,U}, x)
    return (bits >> (Base.significand_bits(T) - 5)) & U(0x1F)
end


@inline function unsafe_log2(x::_MF{T,N}) where {T,N}
    _one = one(T)
    _direct_lo = T(15) / T(16)
    _direct_hi = T(17) / T(16)
    _table = _log2_table(T, Val{N}())

    first_limb = first(x._limbs)
    index = _log2_table_index(first_limb)
    e = unsafe_exponent(first_limb)
    m = unsafe_ldexp(x, -e)
    center = _table.centers[index+1]
    value = _table.values[index+1]

    t_direct = (x - _one) / (x + _one)
    t_table = (m - center) / (m + center)
    p_direct = _MF{T,N}(_log2_kernel_wide(mfsqr(t_direct._limbs, Val{N}())))
    p_table = _MF{T,N}(_log2_kernel_narrow(mfsqr(t_table._limbs, Val{N}())))
    return ifelse((_direct_lo < first_limb) & (first_limb < _direct_hi),
        t_direct * p_direct,
        convert(T, e) + value + t_table * p_table)
end

@inline function unsafe_log2(x::_MFV{M,T,N}) where {M,T,N}
    _one = one(T)
    _direct_lo = T(15) / T(16)
    _direct_hi = T(17) / T(16)
    _table = _log2_table(T, Val{N}())

    first_limb = first(x._limbs)
    index = _log2_table_index(first_limb)
    e = unsafe_exponent(first_limb)
    m = unsafe_ldexp(x, -e)
    centers = _MFV{M,T,N}(ntuple(i -> Vec{M,T}(ntuple(j ->
                _table.centers[extractelement(index.data, j - 1)+1]._limbs[i],
            Val{M}())), Val{N}()))
    values = _MFV{M,T,N}(ntuple(i -> Vec{M,T}(ntuple(j ->
                _table.values[extractelement(index.data, j - 1)+1]._limbs[i],
            Val{M}())), Val{N}()))

    t_direct = (x - _one) / (x + _one)
    t_table = (m - centers) / (m + centers)
    p_direct = _MFV{M,T,N}(_log2_kernel_wide(mfsqr(t_direct._limbs, Val{N}())))
    p_table = _MFV{M,T,N}(_log2_kernel_narrow(mfsqr(t_table._limbs, Val{N}())))
    return vifelse((_direct_lo < first_limb) & (first_limb < _direct_hi),
        t_direct * p_direct,
        convert(Vec{M,T}, e) + values + t_table * p_table)
end


const _LN_2_FULL_F32 = (
    Float32(+0x1.62E430p-001), Float32(-0x1.05C610p-029),
    Float32(-0x1.950D88p-054), Float32(+0x1.D9CC02p-079),
    Float32(-0x1.A12A18p-109), Float32(+0x0.003CD0p-126),
)

const _LN_2_FULL_F64 = (
    +0x1.62E42FEFA39EFp-0001, +0x1.ABC9E3B39803Fp-0056,
    +0x1.7B57A079A1934p-0111, -0x1.ACE93A4EBE5D1p-0165,
    -0x1.23A2A82EA0C24p-0219, +0x1.D881B7AEB2615p-0274,
    +0x1.9552FB4AFA1B1p-0328, +0x1.DA5D5C6B82704p-0385,
    +0x1.4427573B29117p-0440, -0x1.91F6B05A4D7A7p-0494,
    -0x1.DB5173AE53426p-0548, +0x1.1317C387EB9EBp-0604,
    -0x1.90F13B267F137p-0658, +0x1.6FA0EC7657F75p-0712,
    -0x1.234C5E1398A6Bp-0766, +0x1.195EBBF4D7A70p-0821,
    +0x1.8192432AFD0C4p-0875, -0x1.A1BE38BA4BA4Dp-0929,
    -0x1.D7860151CFC06p-0987, +0x0.000032847ED70p-1022,
)

@inline _ln_2(::Type{_MF{Float32,N}}) where {N} =
    _MF{Float32,N}(ntuple(i -> _LN_2_FULL_F32[i], Val{N}()))
@inline _ln_2(::Type{_MF{Float64,N}}) where {N} =
    _MF{Float64,N}(ntuple(i -> _LN_2_FULL_F64[i], Val{N}()))
@inline _ln_2(::Type{_MFV{M,T,N}}) where {M,T,N} =
    _MFV{M,T,N}(_ln_2(_MF{T,N}))


const _LOG10_2_FULL_F32 = (
    Float32(+0x1.344136p-002), Float32(-0x1.EC10C0p-027),
    Float32(-0x1.0CEE0Ep-054), Float32(-0x1.A994FEp-079),
    Float32(+0x1.BE48BCp-104), Float32(+0x0.04D5A6p-126),
)

const _LOG10_2_FULL_F64 = (
    +0x1.34413509F79FFp-0002, -0x1.9DC1DA994FD21p-0059,
    +0x1.22F04D5A618A8p-0114, +0x1.E8F9A4C52F379p-0168,
    +0x1.ADF318F2CA1A9p-0223, -0x1.27E6E60542F62p-0277,
    +0x1.72F1CCD1C6F45p-0331, -0x1.6EB8CB6A4E747p-0385,
    +0x1.0F2C5A93FA92Bp-0440, -0x1.A5A6740F90424p-0494,
    +0x1.2AAE5E4E78E8Dp-0552, -0x1.E77F3FA345B28p-0606,
    +0x1.397EF38817A76p-0660, -0x1.BBB593A9B4989p-0715,
    +0x1.235FA967AC54Fp-0775, +0x1.80B0833C5941Bp-0829,
    -0x1.1F7A572611FB1p-0883, +0x1.E991F7BDD462Fp-0938,
    -0x1.B83B9AAA00400p-0996, +0x0.000000142B5B1p-1022,
)

@inline _log10_2(::Type{_MF{Float32,N}}) where {N} =
    _MF{Float32,N}(ntuple(i -> _LOG10_2_FULL_F32[i], Val{N}()))
@inline _log10_2(::Type{_MF{Float64,N}}) where {N} =
    _MF{Float64,N}(ntuple(i -> _LOG10_2_FULL_F64[i], Val{N}()))
@inline _log10_2(::Type{_MFV{M,T,N}}) where {M,T,N} =
    _MFV{M,T,N}(_log10_2(_MF{T,N}))


@inline unsafe_log(x::_MF{T,N}) where {T,N} =
    unsafe_log2(x) * _ln_2(_MF{T,N})
@inline unsafe_log(x::_MFV{M,T,N}) where {M,T,N} =
    unsafe_log2(x) * _ln_2(_MFV{M,T,N})
@inline unsafe_log10(x::_MF{T,N}) where {T,N} =
    unsafe_log2(x) * _log10_2(_MF{T,N})
@inline unsafe_log10(x::_MFV{M,T,N}) where {M,T,N} =
    unsafe_log2(x) * _log10_2(_MFV{M,T,N})


@inline function _handle_special_log(
    x::_MF{T,N},
    y::_MF{T,N},
) where {T,N}
    _nan = _MF{T,N}(ntuple(_ -> T(NaN), Val{N}()))
    y = ifelse(iszero(x), typemin(_MF{T,N}), y)
    y = ifelse(isinf(x) & !signbit(x), typemax(_MF{T,N}), y)
    y = ifelse(isnan(x) | (signbit(x) & !iszero(x)), _nan, y)
    return y
end

@inline function _handle_special_log(
    x::_MFV{M,T,N},
    y::_MFV{M,T,N},
) where {M,T,N}
    _nan = _MFV{M,T,N}(ntuple(_ -> Vec{M,T}(T(NaN)), Val{N}()))
    y = vifelse(iszero(x), typemin(_MFV{M,T,N}), y)
    y = vifelse(isinf(x) & !signbit(x), typemax(_MFV{M,T,N}), y)
    y = vifelse(isnan(x) | (signbit(x) & !iszero(x)), _nan, y)
    return y
end


@inline Base.log(x::_MF{T,N}) where {T,N} =
    _handle_special_log(x, unsafe_log(x))
@inline Base.log(x::_MFV{M,T,N}) where {M,T,N} =
    _handle_special_log(x, unsafe_log(x))
@inline Base.log2(x::_MF{T,N}) where {T,N} =
    _handle_special_log(x, unsafe_log2(x))
@inline Base.log2(x::_MFV{M,T,N}) where {M,T,N} =
    _handle_special_log(x, unsafe_log2(x))
@inline Base.log10(x::_MF{T,N}) where {T,N} =
    _handle_special_log(x, unsafe_log10(x))
@inline Base.log10(x::_MFV{M,T,N}) where {M,T,N} =
    _handle_special_log(x, unsafe_log10(x))
