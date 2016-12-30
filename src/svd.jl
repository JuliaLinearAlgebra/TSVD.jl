function biLanczosIterations(A, stepSize, αs, βs, U, V, μs, νs, τ, reorth_in, tolReorth, debug)

    m, n = size(A)
    reorth_μ = reorth_in
    nReorth = 0
    nReorthVecs = 0

    T = eltype(A)
    Tr = real(T)

    maxνs = Tr[]
    maxμs = Tr[]

    iter = length(αs)

    u = U[iter + 1]
    v = V[iter]
    β = βs[iter]

    for j = iter + (1:stepSize)

        # The v step
        vOld = v
        ## apply operator
        v = A'u
        axpy!(T(-β), vOld, v)
        α = norm(v)

        ## update norm(A) estimate. FixMe! Use tighter bounds, see Larsen's thesis page 33
        τ = max(τ, eps(Tr) * (α + β))
        debug && @show τ

        ## run ω recurrence
        reorth_ν = false
        for i = 1:j - 1
            ν = βs[i]*μs[i + 1] + αs[i]*μs[i] - β*νs[i]
            ν = (ν + copysign(τ, ν))/α
            if abs(ν) > tolReorth
                reorth_ν |= true
            end
            νs[i] = ν
        end
        if j > 1
            push!(maxνs, maximum(abs, νs))
        end
        push!(νs, 1)

        ## reorthogonalize if necessary
        if reorth_ν || reorth_μ
            debug && println("Reorth v")
            for i in 1:j - 1
                axpy!(-Base.dot(V[i], v), V[i], v)
                νs[i] = eps(Tr)
                nReorthVecs += 1
            end
            α = norm(v)
        end

        ## update the result vectors
        push!(αs, α)
        scale!(v, inv(α))
        push!(V, v)

        # The u step
        uOld = u
        ## apply operator
        u = A*v
        axpy!(T(-α), uOld, u)
        β = norm(u)

        ## update norm(A) estimate. FixMe! Use tighter bounds, see Larsen's thesis page 33
        τ = max(τ, eps(Tr) * (α + β))
        debug && @show τ

        ## run ω recurrence
        reorth_μ = false
        for i = 1:j
            μ = αs[i]*νs[i] - α*μs[i]
            if i > 1
                μ += βs[i - 1]*νs[i-1]
            end
            μ = (μ + copysign(τ, μ))/β
            if abs(μ) > tolReorth
                reorth_μ |= true
            end
            μs[i] = μ
        end
        push!(maxμs, maximum(μs))
        push!(μs, 1)

        ## reorthogonalize if necessary
        if reorth_ν || reorth_μ
            debug && println("Reorth u")
            for i in 1:j
                axpy!(-Base.dot(U[i], u), U[i], u)
                μs[i] = eps(Tr)
                nReorthVecs += 1
            end
            β = norm(u)
            nReorth += 1
        end

        ## update the result vectors
        push!(βs, β)
        scale!(u, inv(β))
        push!(U, u)
    end

    return αs, βs, U, V, μs, νs, reorth_μ, maxμs, maxνs, nReorth, nReorthVecs
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
    V = fill(v, 1)
    αs = fill(α, 1)

    u = A*v
    uOld = similar(u)
    copy!(uOld, initVec)
    scale!(uOld, inv(nrmInit))
    axpy!(eltype(u)(-α), uOld, u)
    β = norm(u)
    scale!(u, inv(β))
    U = typeof(u)[uOld, u]
    βs = fill(β, 1)

    # error estimate used in ω recurrence
    τ = eps(Tr)*(α + β)
    ν = 1 + τ/α
    μ = τ/β

    # Arrays for saving the estimates of the maximum angles between Lanczos vectors
    maxμs = Tr[]
    maxνs = Tr[]

    # return types can only be inferred by man, not the machine
    αs::Vector{Tr},
    βs::Vector{Tr},
    U::Vector{typeof(v)}, # for some reason typeof(U) doesn't work here
    V::Vector{typeof(v)},
    μs::Vector{Tr},
    νs::Vector{Tr},
    reorth_μ::Bool, maxμ, maxν, _ =
        biLanczosIterations(A, nVals - 1, αs, βs, U, V, [μ, 1], [one(μ)], τ, false, tolReorth, debug)

    # Iteration count
    iter = nVals

    # Save the estimates of the maximum angles between Lanczos vectors
    append!(maxμs, maxμ)

    # vals0 = svdvals(Bidiagonal(αs, βs[1:end-1], false))
    vals0 = svdvals(Bidiagonal([αs;z], βs, false))
    vals1 = vals0

    hasConv = false
    while iter <= maxIter
        _, _, _, _, _, _, reorth_μ, maxμ, maxν, _ =
            biLanczosIterations(A, stepSize, αs, βs, U, V, μs, νs, τ, reorth_μ, tolReorth, debug)
        append!(maxμs, maxμ)
        append!(maxνs, maxν)
        iter += stepSize

        # vals1 = svdvals(Bidiagonal(αs, βs[1:end-1], false))
        vals1 = svdvals(Bidiagonal([αs;z], βs, false))

        debug && @show vals1[nVals]/vals0[nVals] - 1

        if vals0[nVals]*(1 - tolConv) < vals1[nVals] < vals0[nVals]*(1 + tolConv)
            # UU, ss, VV = svd(Bidiagonal([αs;z], βs[1:end-1], false))
            # This is more expensive than necessary because we only need the last components. However, LAPACK doesn't support this.
            UU, ss, VV = svd(Bidiagonal([αs;z], βs, false))
            # @show UU[end, 1:iter]*βs[end]
            if all(abs.(UU[end, 1:nVals])*βs[end] .< tolConv*ss[1:nVals]) && all(abs.(VV[end, 1:nVals])*βs[end] .< tolConv*ss[1:nVals])
                hasConv = true
                break
            end
        end
        vals0 = vals1
        τ = eps(eltype(vals1))*vals1[1]

        debug && @show iter
        debug && @show τ
    end
    if !hasConv
        error("no convergence")
    end

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
        maxμs,
        maxνs, _
end

"""
## tsvd(A, nVals = 1, [maxIter = 1000, initVec = randn(m), tolConv = 1e-12, tolReorth = 0.0, debug = false])

Computes the truncated singular value decomposition (TSVD) by Lanczos bidiagonalization of the operator `A`. The Lanczos vectors are partially orthogonalized as described in

R. M. Larsen, *Lanczos bidiagonalization with partial reorthogonalization*, Department of Computer Science, Aarhus University, Technical report, DAIMI PB-357, September 1998.

Note! At the moment the default is complete orthogonalization because the ω recurrences that measure the orthogonality of the Lanczos vectors still requires some fine tuning.

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
# Split Vector and Matrix to avoid ambiguity
function (*){T}(A::AtA{T}, x::AbstractVector)
    A_mul_B!(one(T), A.matrix, convert(typeof(A.vector), x), zero(T), A.vector)
    return Ac_mul_B!(one(T), A.matrix, A.vector, zero(T), similar(A.vector, size(x)))
end
function (*){T}(A::AtA{T}, x::AbstractMatrix)
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
    return sqrt.(reverse(values)[1:nVals]), mV[:,end:-1:1][:,1:nVals]
end
