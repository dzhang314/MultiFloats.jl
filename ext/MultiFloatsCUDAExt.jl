module MultiFloatsCUDAExt

using CUDA
import MultiFloats: one_prod


CUDA.@device_override @inline one_prod(a::Float32, b::Float32) = Base.llvmcall(
    """
    %res = call float asm "mul.rn.f32 \$0, \$1, \$2;", "=f,f,f"(float %0, float %1)
    ret float %res
    """,
    Float32, Tuple{Float32,Float32}, a, b)


CUDA.@device_override @inline one_prod(a::Float64, b::Float64) = Base.llvmcall(
    """
    %res = call double asm "mul.rn.f64 \$0, \$1, \$2;", "=d,d,d"(double %0, double %1)
    ret double %res
    """,
    Float64, Tuple{Float64,Float64}, a, b)


end # module MultiFloatsCUDAExt
