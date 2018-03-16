# Copyright 2015-2018 Andreas Noack
module TSVD

    export tsvd

    import Base: *, hcat, maximum, size
    import Base.LinAlg: A_mul_B!, Ac_mul_B!, axpy!

    using Base.LinAlg: BlasComplex, BlasFloat, BlasReal, givensAlgorithm

    include("common.jl")
    include("eig.jl")
    include("svd.jl")

end