var documenterSearchIndex = {"docs":
[{"location":"#The-TSVD-documentation","page":"The TSVD documentation","title":"The TSVD documentation","text":"","category":"section"},{"location":"#Functions","page":"The TSVD documentation","title":"Functions","text":"","category":"section"},{"location":"","page":"The TSVD documentation","title":"The TSVD documentation","text":"CurrentModule = TSVD\nDocTestSetup = quote\n    using MatrixDepot, TSVD\n    try\n        matrixdepot(\"LPnetlib/lp_osa_30\")\n    catch\n    \tnothing\n    end\nend","category":"page"},{"location":"","page":"The TSVD documentation","title":"The TSVD documentation","text":"tsvd","category":"page"},{"location":"#TSVD.tsvd","page":"The TSVD documentation","title":"TSVD.tsvd","text":"tsvd(A, nvals = 1; [maxiter, initvec, tolconv, tolreorth, debug])\n\nComputes the truncated singular value decomposition (TSVD) by Lanczos bidiagonalization of the operator A. The Lanczos vectors are partially orthogonalized as described in\n\nR. M. Larsen, Lanczos bidiagonalization with partial reorthogonalization, Department of Computer Science, Aarhus University, Technical report, DAIMI PB-357, September 1998.\n\nPositional arguments:\n\nA: Anything that supports the in place update operations\n\nmul!(y::AbstractVector, A, x::AbstractVector, α::Number, β::Number)\n\nand\n\nmul!(y::AbstractVector, A::Adjoint, x::AbstractVector, α::Number, β::Number)\n\ncorresponding to the operations y := α*op(A)*x + β*y where op can be either the identity or the conjugate transpose of A. If the initvec argument is not supplied then it is furthermore required that A supports eltype and size.\n\nnvals: The number of singular values and vectors to compute. Default is one (the largest).\n\nKeyword arguments:\n\nmaxiter: The maximum number of iterations of the Lanczos bidiagonalization. Default is 1000, but usually much fewer iterations are needed.\ninitvec: Initial U vector for the Lanczos procesdure. Default is a vector of Gaussian random variates. The length and eltype of the initvec will control the size and element types of the basis vectors in U and V.\ntolconv: Relative convergence criterion for the singular values. Default is sqrt(eps(real(eltype(A)))).\ntolreorth: Absolute tolerance for the inner product of the Lanczos vectors as measured by the ω recurrence. Default is sqrt(eps(real(eltype(initvec)))). 0.0 and Inf correspond to complete and no reorthogonalization respectively.\ndebug: Boolean flag for printing debug information\n\nOutput:\n\nThe output of the procesure it the truple tuple (U, s, V)\n\nU: size(A, 1) times nvals matrix of left singular vectors.\ns: Vector of length nvals of the singular values of A.\nV: size(A, 2) times nvals matrix of right singular vectors.\n\nExamples\n\njulia> A = matrixdepot(\"LPnetlib/lp_osa_30\")\n4350×104374 SparseArrays.SparseMatrixCSC{Float64, Int64} with 604488 stored entries:\n⠙⠮⠷⠶⠽⠶⠽⠶⠮⠷⠮⠷⠶⠽⠶⠽⠶⠬⠷⠮⠷⠦⠽⠶⠽⠶⠽⠶⠮⠷⠮⠷⠶⠽⠶⠽⠶⠭⠷⠦\n\njulia> U, s, V = tsvd(A, 5);\n\njulia> round.(s, digits=7)\n5-element Vector{Float64}:\n 1365.8944098\n 1033.2125634\n  601.3524529\n  554.107656\n  506.0414587\n\n\n\n\n\n","category":"function"},{"location":"","page":"The TSVD documentation","title":"The TSVD documentation","text":"DocTestSetup = nothing","category":"page"}]
}