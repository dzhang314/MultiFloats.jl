module MultiFloatsAMDGPUExt

using AMDGPU
import MultiFloats.Reproducible: sqrt_r


AMDGPU.Device.@device_override @inline sqrt_r(x::Float32) =
    ccall("extern __ocml_sqrt_f32", llvmcall, Float32, (Float32,), x)


AMDGPU.Device.@device_override @inline sqrt_r(x::Float64) =
    ccall("extern __ocml_sqrt_f64", llvmcall, Float64, (Float64,), x)


end # module MultiFloatsAMDGPUExt
