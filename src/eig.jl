function lanczosIterations(A, stepSize, αs, βs, vs, reorthogonalize)

    iter = length(αs)
    v1 = vs[iter]
    v = vs[iter + 1]
    β = βs[iter]

    for i = iter + (1:stepSize)
        w = A*v
        α = real(dot(w,v))

        axpy!(-α, v, w)
        axpy!(-β, v1, w)

        # Orthogonlize (FixMe! consider a partial reorthogonal strategy here)
        if reorthogonalize
            for j = 1:i - 1
                Base.axpy!(-dot(vs[j], w), vs[j], w)
            end
        end

        v1 = v
        v, β = qr!(w)

        push!(αs, α)
        push!(βs, β)
        push!(vs, v)
    end
    return αs, βs, vs
end

function _teig(A,
    nVals = 1;
    maxIter = size(A, 1),
    initVec = convert(Vector{eltype(A)}, randn(size(A, 1))),
    tolConv = sqrt(eps(real(eltype(A)))),
    reorthogonalize = true,
    stepSize = max(1, div(nVals, 10)),
    debug = false)

    # LinAlg.chksquare(A)

    ### initialize αs, βs and vs ###
    # run a single iteration of the Lanczos procedure. Note that this step handles promotion Lanczos vectors as well. They will in in general be different from `initVec`.
    v1 = A*initVec
    v0 = convert(typeof(v1), initVec)
    v0, nm0 = qr!(v0)
    scale!(v1, inv(nm0)) # because we don't assume that initVec is normalized
    α = real(dot(v1, v0))
    axpy!(-α, v0, v1)
    v1, β = qr!(v1)
    αs = [α]
    βs = [β]
    vs = typeof(v1)[v0, v1]

    ### initial iterations ###
    lanczosIterations(A, nVals - 1, αs, βs, vs, reorthogonalize)

    ### Iteration count ###
    iter = nVals

    # values0 = sort!(abs(eigvals(SymTridiagonal(αs, βs[1:end - 1]))), rev = true)
    # values1 = values0

    z = zeros(eltype(αs), 1)

    ### run itertions ###
    # continue until convergence or the maximum number of iterations has been reached.
    converged = UInt[]
    while iter <= maxIter
        lanczosIterations(A, stepSize, αs, βs, vs, reorthogonalize)
        # values, vectors = LinearAlgebra.EigenSelfAdjoint.eigQL!(SymTridiagonal(copy(αs), βs[1:end - 1]), [zeros(length(αs) - 1); ones(eltype(αs), 1)]')
        values, vectors = eig(SymTridiagonal(αs, βs[1:end-1]))

        ### Use criteria (9.18) of Peter Arbenz' lecture notes ###
        βj = βs[end]
        converged = Int[]
        # for i = maximum(converged) + 1:length(αs)
        for i = 1:length(αs)
            if βj*abs(vectors[end,i]) < abs(values[i])*tolConv
                push!(converged, i)
            end
        end

        debug && @show converged

        # if length(converged) >= nVals
        #     break
        # end

        # For now we hardcode convergence for largest positive value. This should be a choice later on
        if all([length(αs) - i + 1 in converged for i = 1:nVals])
            break
        end

        iter += stepSize
        debug && @show iter
    end
    if length(converged) < nVals
        error("no convergence")
    end

    S = SymTridiagonal(αs, βs[1:end-1])
    values, vectors = eig(S)
    return values[converged], vectors[:, converged], S, vs, βs[end]
end

teig(A,
    nVals = 1;
    maxIter = size(A, 1),
    initVec = convert(Vector{eltype(A)}, randn(size(A, 1))),
    tolConv = 1e-12,
    reorthogonalize = true,
    stepSize = max(1, div(nVals, 10)),
    debug = false) = _teig(A, nVals, maxIter = maxIter, initVec = initVec,
        tolConv = tolConv, reorthogonalize = reorthogonalize, stepSize = stepSize, debug = debug)
