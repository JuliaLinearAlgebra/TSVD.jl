# Copyright 2015-2018 Andreas Noack

# __precompile__() # not sure from which Julia version on this is obsolete

module TSVD

    export tsvd

    import Base: *, hcat, maximum, size
    import LinearAlgebra: axpy!, mul!

    using LinearAlgebra
    using LinearAlgebra: BlasComplex, BlasFloat, BlasInt, BlasReal

    include("common.jl")
    include("eig.jl")
    include("svd.jl")

end
