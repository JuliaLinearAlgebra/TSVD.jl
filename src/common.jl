function hcat(x::Vector{T}) where T<:AbstractVecOrMat
    l    = length(x)
    if l == 0
        throw(ArgumentError("cannot flatten empty vector"))
    else
        x1   = x[1]
        m, n = size(x1, 1), size(x1, 2)
        B    = similar(x1, eltype(x1), (m, l*n))
        for i = 1:l
            B[:, (i - 1)*n + 1:i*n] = x[i]
        end
        return B
    end
end

A_mul_B!(α::Number, A::StridedMatrix{T}, x::StridedVector{T}, β::Number, y::StridedVector{T}) where {T<:BlasFloat} = BLAS.gemv!('N', convert(T, α), A, x, convert(T, β), y)
Ac_mul_B!(α::Number, A::StridedMatrix{T}, x::StridedVector{T}, β::Number, y::StridedVector{T}) where {T<:BlasReal} = BLAS.gemv!('T', convert(T, α), A, x, convert(T, β), y)
Ac_mul_B!(α::Number, A::StridedMatrix{T}, x::StridedVector{T}, β::Number, y::StridedVector{T}) where {T<:BlasComplex} = BLAS.gemv!('C', convert(T, α), A, x, convert(T, β), y)

function A_mul_B!(α::Number, A::StridedMatrix, x::StridedVector, β::Number, y::StridedVector)
    n = length(y)
    for i = 1:n
        y[i] *= β
    end
    for i = 1:n
        for l = 1:size(A,2)
            y[i] += α*A[i,l]*x[l]
        end
    end
    return y
end

function Ac_mul_B!(α::Number, A::StridedMatrix, x::StridedVector, β::Number, y::StridedVector)
    n = length(y)
    for i = 1:n
        y[i] *= β
    end
    for i = 1:n
        for l = 1:size(A,2)
            y[i] += α*A[l,i]'*x[l]
        end
    end
    return y
end

function qr!(x::AbstractArray)
    nm = norm(x)
    scale!(x, inv(nm))
    return x, nm
end
