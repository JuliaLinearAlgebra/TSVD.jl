#Partial reorthogonalization along the lines of Larsen (1999)'s adaptation of
#the method described by Simon
type OmegaRecurrence{Tr}
    tolReorth::Tr
    reorth_μ::Bool
    reorth_ν::Bool
    τ::Tr
    μs::Vector{Tr}
    νs::Vector{Tr}
    maxμs::Vector{Tr}
    maxνs::Vector{Tr}
    nReorth::Int
    nReorthVecs::Int
end

function update1!{Tr}(ω::OmegaRecurrence{Tr}, αs, βs, α::Tr, β::Tr)
    #Update estimate of matrix norm
    ##FIXME Use tighter bounds, see Larsen's thesis page 33
    ω.τ = max(ω.τ, eps(Tr) * (α + β))

    reorth_ν = false
    for i in eachindex(αs)
        ν = βs[i]*ω.μs[i+1] + αs[i]*ω.μs[i] - β*ω.νs[i]
        ν = (ν + copysign(ω.τ, ν))/α
        reorth_ν |= (abs(ν) > ω.tolReorth)
        ω.νs[i] = ν
    end
    if length(αs) > 1
        push!(ω.maxνs, maxabs(ω.νs))
    end
    push!(ω.νs, 1)
    ω.reorth_ν = reorth_ν
    nothing
end

function update2!{Tr}(ω::OmegaRecurrence{Tr}, αs, βs, α::Tr, β::Tr)
    #Update estimate of matrix norm
    ##FIXME Use tighter bounds, see Larsen's thesis page 33
    ω.τ = max(ω.τ, eps(Tr) * (α + β))

    reorth_μ = false
    for i in eachindex(αs)
        μ = αs[i]*ω.νs[i] - α*ω.μs[i]
        if i > 1
            μ += βs[i-1]*ω.νs[i-1]
        end
        μ = (μ + copysign(ω.τ, μ))/β
        reorth_μ |= (abs(μ) > ω.tolReorth)
        ω.μs[i] = μ
    end
    push!(ω.maxμs, maxabs(ω.μs))
    push!(ω.μs, 1)
    ω.reorth_μ = reorth_μ
    nothing
end

function reorthogonalize1!{Tr}(v, V, α::Tr, ω)
    if ω.reorth_ν || ω.reorth_μ
        for i in 1:size(V, 1)
            axpy!(-Base.dot(V[i], v), V[i], v)
            ω.νs[i] = eps(Tr) #reset ω-recurrences
            ω.nReorthVecs += 1
        end
        return norm(v)
    else
        return α
    end
end

function reorthogonalize2!{Tr}(u, U, β::Tr, ω)
    if ω.reorth_ν || ω.reorth_μ
        for i in 1:size(U, 1)
            axpy!(-Base.dot(U[i], u), U[i], u)
            ω.μs[i] = eps(Tr) #reset ω-recurrences
            ω.nReorthVecs += 1
        end
        ω.nReorth += 1
        return norm(u)
    else
        return β
    end
end

function biLanczosIterations(A, stepSize, αs, βs, U, V, ω::OmegaRecurrence, debug::Bool)

    m, n = size(A)

    T = eltype(A)
    Tr = real(T)

    iter = length(αs)

    u = U[iter + 1]
    v = V[iter]
    β = βs[iter]

    for j = iter + (1:stepSize)

        # The v step
        vOld = v
        v = A'u ## apply operator
        axpy!(T(-β), vOld, v)
        α = norm(v)

        update1!(ω, αs, βs, α, β)         ## run ω recurrence
        α = reorthogonalize1!(v, V, α, ω) ## reorthogonalize if necessary

        ## update the result vectors
        push!(αs, α)
        push!(V, scale!(v, inv(α)))

        # The u step
        uOld = u
        u = A*v ## apply operator
        axpy!(T(-α), uOld, u)
        β = norm(u)

        update2!(ω, αs, βs, α, β)         ## run ω recurrence
        β = reorthogonalize2!(u, U, β, ω) ## reorthogonalize if necessary

        ## update the result vectors
        push!(βs, β)
        push!(U, scale!(u, inv(β)))
    end

    return αs, βs, U, V, ω
end

function _tsvd(A,
    nVals = 1;
    maxIter = 1000,
    initVec = convert(Vector{eltype(A)}, randn(size(A,1))),
    tolConv = sqrt(eps(real(eltype(A)))),
    tolReorth = sqrt(eps(real(eltype(A)))),
    stepSize = max(1, div(nVals, 10)),
    debug = false)

    Tv = eltype(initVec)
    Tr = real(Tv)

    # I need to append βs with a zero at each iteration. Tt is much easier for type inference if it is a vector with the right element type
    z = zeros(Tr, 1)

    # initialize the αs, βs, U and V. Use result of first matvec to infer the correct types.
    # So the first iteration is run here, but slightly differently from the rest of the iterations
    nrmInit = norm(initVec)
    v = A'initVec
    scale!(v, inv(nrmInit))
    α = norm(v)
    scale!(v, inv(α))

    u = A*v
    uOld = similar(u)
    copy!(uOld, initVec)
    scale!(uOld, inv(nrmInit))
    axpy!(eltype(u)(-α), uOld, u)
    β = norm(u)
    scale!(u, inv(β))

    # error estimates used in ω recurrence
    τ = eps(Tr)*(α + β)
    ν = 1 + τ/α
    μ = τ/β
    ω = OmegaRecurrence{Tr}(tolReorth, false, false,
        τ, [μ, 1], [one(μ)], Tr[], Tr[], 0, 0) #???? [one(μ)] NOT [ν]?

    αs,βs,U,V,ω = biLanczosIterations(
        A, nVals - 1, [α], [β], typeof(u)[uOld, u], typeof(v)[v], ω, debug)

    # Iteration count
    iter = nVals

    # vals0 = svdvals(Bidiagonal(αs, βs[1:end-1], false))
    vals0 = svdvals(Bidiagonal([αs;z], βs, false))
    vals1 = vals0

    hasConv = false
    while iter <= maxIter
        _1, _2, _3, _4, ω =
            biLanczosIterations(A, stepSize, αs, βs, U, V, ω, debug)
        iter += stepSize

        # vals1 = svdvals(Bidiagonal(αs, βs[1:end-1], false))
        vals1 = svdvals(Bidiagonal([αs;z], βs, false))

        debug && @show vals1[nVals]/vals0[nVals] - 1

        if vals0[nVals]*(1 - tolConv) < vals1[nVals] < vals0[nVals]*(1 + tolConv)
            # UU, ss, VV = svd(Bidiagonal([αs;z], βs[1:end-1], false))
            # This is more expensive than necessary because we only need the last components. However, LAPACK doesn't support this.
            UU, ss, VV = svd(Bidiagonal([αs;z], βs, false))
            # @show UU[end, 1:iter]*βs[end]
            if all(abs(UU[end, 1:nVals])*βs[end] .< tolConv*ss[1:nVals]) && all(abs(VV[end, 1:nVals])*βs[end] .< tolConv*ss[1:nVals])
                hasConv = true
                break
            end
        end
        vals0 = vals1
        ω.τ = eps(eltype(vals1))*vals1[1]

        debug && @show iter
        debug && @show ω.τ
    end
    hasConv || error("no convergence")

    # Form upper bidiagonal square matrix
    # m = length(U[1])
    # for j = 1:length(αs)
    #     # Calculate Givens rotation
    #     c, s, αs[j] = givensAlgorithm(αs[j], βs[j])

    #     # Update left vector
    #     # for i = 1:m
    #     #     uij       = U[j][i]
    #     #     uij1      = U[j+1][i]
    #     #     U[j][i]   = uij*c + uij1*s'
    #     #     U[j+1][i] = uij1*c - uij*s
    #     # end

    #     # Update bidiagonal matrix
    #     if j < length(αs)
    #         αj1 = αs[j + 1]
    #         αs[j + 1] = c*αj1
    #         βs[j] = s*αj1
    #     end
    # end

    # Calculate the bidiagonal SVD and update U and V
    mU = hcat(U[1:end-1])
    mV = hcat(V)
    B = Bidiagonal(αs, βs[1:end-1], false)
    smU, sms, smV = svd(B)

    return (mU*smU)[:,1:nVals],
        sms[1:nVals],
        (mV*smV)[:,1:nVals],
        Bidiagonal(αs, βs[1:end-1], false),
        mU,
        mV,
        ω
end

"""
## tsvd(A, nVals = 1, [maxIter = 1000, initVec = randn(m), tolConv = 1e-12, tolReorth = 0.0, debug = false])

Computes the truncated singular value decomposition (TSVD) by Lanczos bidiagonalization of the operator `A`. The Lanczos vectors are partially orthogonalized as described in

R. M. Larsen, *Lanczos bidiagonalization with partial reorthogonalization*, Department of Computer Science, Aarhus University, Technical report, DAIMI PB-357, September 1998.

**Arguments:**

- `A`: Anything that supports the in place update operations

```julia
A_mul_B!(α::Number, A, x::Vector, β::Number, y::Vector)
``` and
```julia
Ac_mul_B!(α::Number, A, x::Vector, β::Number, y::Vector)
```
corresponding to the operations `y := α*op(A)*x + β*y` where `op` can be either
the identity or the conjugate transpose of `A`.

- `nVals`: The number of singular values and vectors to compute. Default is one (the largest).

- `maxIter`: The maximum number of iterations of the Lanczos bidiagonalization. Default is 1000, but usually much fewer iterations are needed.

- `initVec`: Initial `U` vector for the Lanczos procesdure. Default is a real vector of real Gaussian random variates. Should have the same element type as the operator `A`.

- `tolConv`: Relative convergence criterion for the singular values. Default is `sqrt(eps(real(eltype(A))))`.

- `tolReorth`: Absolute tolerance for the inner product of the Lanczos vectors as measured by the ω recurrence. Default is `sqrt(eps(real(eltype(A))))`. '0.0' and `Inf` corresponds to complete and no reorthogonalization respectively.

- `debug`: Boolean flag for printing debug information

**Output:**

The output of the procesure it the truple tuple `(U,s,V)`

- `U`: `size(A,1)` times `nVals` matrix of left singular vectors.
- `s`: Vector of length `nVals` of the singular values of `A`.
- `V`: `size(A,2)` times `nVals` matrix of right singular vectors.
"""
tsvd(A,
    nVals = 1;
    maxIter = 1000,
    initVec = convert(Vector{eltype(A)}, randn(size(A,1))),
    tolConv = sqrt(eps(real(eltype(A)))),
    tolReorth = sqrt(eps(real(eltype(A)))),
    stepSize = max(1, div(nVals, 10)),
    debug = false) =
        _tsvd(A, nVals, maxIter = maxIter, initVec = initVec, tolConv = tolConv,
            tolReorth = tolReorth, debug = debug)[1:3]


### SVD by Lanczos on A'A

type AtA{T,S<:AbstractMatrix,V<:AbstractVecOrMat} <: AbstractMatrix{T}
    matrix::S
    vector::V
end
function AtA(A::AbstractMatrix, x::AbstractVecOrMat)
    y = A*x
    return AtA{eltype(y),typeof(A),typeof(y)}(A,y)
end

function size(A::AtA, i::Integer)
    if i < 1
        error("arraysize: dimension out of range")
    elseif i < 3
        return size(A.matrix, 2)
    else
        return 1
    end
end
size(A::AtA) = (size(A, 1), size(A, 2))

function A_mul_B!{T,S,V}(α::T, A::AtA{T,S,V}, x::AbstractVecOrMat{T}, β::T, y::AbstractVecOrMat{T})
    A_mul_B!(one(T), A.matrix, x, zero(T), A.vector)
    Ac_mul_B!(α, A.matrix, A.vector, β, y)
    return y
end
function (*){T}(A::AtA{T}, x::AbstractVecOrMat)
    A_mul_B!(one(T), A.matrix, convert(typeof(A.vector), x), zero(T), A.vector)
    return Ac_mul_B!(one(T), A.matrix, A.vector, zero(T), similar(A.vector, size(x)))
end

function tsvd2(A,
    nVals = 1;
    maxIter = minimum(size(A)),
    initVec = convert(Vector{eltype(A)}, randn(size(A,2))),
    tolConv = sqrt(eps(real(eltype(A)))),
    stepSize = max(1, div(nVals, 10)),
    debug = false)
    values, vectors, S, lanczosVecs = _teig(AtA(A, initVec), nVals, maxIter = maxIter,
        initVec = initVec, tolConv = tolConv, stepSize = stepSize, debug = debug)
    mV = hcat(lanczosVecs[1:end-1])*vectors
    return sqrt(reverse(values)[1:nVals]), mV[:,end:-1:1][:,1:nVals]
end
