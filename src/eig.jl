function lanczosIterations(A, steps, αs, βs, vs)

    iter = length(αs)
    v1 = vs[iter]
    v = vs[iter + 1]
    β = βs[iter]

    for i = iter + (1:steps)
        w = A*v
        α = real(dot(w,v))

        axpy!(-α, v, w)
        axpy!(-β, v1, w)

        # Orthogonlize (FixMe! consider a partial reorthogonal strategy here)
        for j = 1:i - 1
            Base.axpy!(-dot(vs[j], w), vs[j], w)
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
    nVals = 1,
    maxIter = div(size(A, 1), 5),
    initVec = convert(Vector{eltype(A)}, randn(size(A, 1))),
    tolConv = 1e-12)

    LinAlg.chksquare(A)

    ### Iteration count and step size ###
    cc = max(5, nVals)
    steps = 2

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
    lanczosIterations(A, cc, αs, βs, vs)

    values0 = sort!(abs(eigvals(SymTridiagonal(αs, βs[1:end - 1]))), rev = true)
    values1 = values0

    ### run itertions ###
    # continue until convergence or the maximum number of iterations has been reached.
    hasConv = false
    while cc <= maxIter
        lanczosIterations(A, steps, αs, βs, vs)
        values1 = sort!(abs(eigvals(SymTridiagonal(αs, βs[1:end - 1]))), rev = true)
        if values0[nVals]*(1 - tolConv) < values1[nVals] < values0[nVals]*(1 + tolConv)
            hasConv = true
            break
        end
        values0 = values1
        cc += steps
    end
    if !hasConv
        error("no convergence")
    end

    S = SymTridiagonal(αs, βs[1:end-1])
    values, vectors = eig(S)
    sp = sortperm(abs(values), rev = true)
    return values[sp[1:nVals]], vectors[:, sp[1:nVals]], S, vs
end

teig(A,
    nVals = 1,
    maxIter = div(size(A, 1), 5),
    initVec = convert(Vector{eltype(A)}, randn(size(A, 1))),
    tolConv = 1e-12) = _teig(A, nVals, maxIter, initVec, tolConv)
