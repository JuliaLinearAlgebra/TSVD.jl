# Copyright 2015-2018 Andreas Noack

__precompile__()

module TSVD

    export tsvd

    import Base: *, hcat, maximum, size
    import LinearAlgebra: A_mul_B!, Ac_mul_B!, axpy!, mul!

    using LinearAlgebra
    using LinearAlgebra: BlasComplex, BlasFloat, BlasInt, BlasReal

    include("common.jl")
    include("eig.jl")
    include("svd.jl")

end