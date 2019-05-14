using Test, TSVD, LinearAlgebra, SparseArrays

@testset "test tsvd with m = $m, n = $n, and p = $p" for
    (m, n, p) in ((10, 6, 0.8),
                  (6, 10, 0.8),
                  (10, 10, 0.8),
                  (100, 60, 0.1),
                  (60, 100, 0.1),
                  (100, 100, 0.1))

    mnp = round(Integer, m*n*p)

    @testset "Matrix type is $tp" for
        (tp, A) in (("Real dense", randn(m, n)),
                    ("Complex dense", complex.(randn(m, n), randn(m, n))),
                    ("Real sparse", sprandn(m, n, p)),
                    ("Complex sparse", sparse(rand(1:m, mnp), rand(1:n, mnp), complex.(randn(mnp), randn(mnp)), m, n)))

        Uf, sf, Vf = svd(Array(A))

        @testset "computing $k triplets" for k = 1:5
            U, s, V = TSVD.tsvd(A, k)
            @test norm(s - sf[1:k]) < sqrt(eps(real(eltype(A))))*mnp
            @test norm(abs.(U'Uf[:,1:k]) - I) < sqrt(eps(real(eltype(A))))*mnp
            @test norm(abs.(V'Vf[:,1:k]) - I) < sqrt(eps(real(eltype(A))))*mnp

            s, V = TSVD.tsvd2(A, k)
            # tmp = TSVD._teig(TSVD.AtA(A, randn(n)), k, debug = true)
            # @test norm(sqrt(reverse(tmp[1]))[1:k] - sf[1:k]) < sqrt(eps(real(eltype(A))))*mnp
            @test norm(s - sf[1:k]) < sqrt(eps(real(eltype(A))))*mnp
            @test norm(abs.(V'Vf[:,1:k]) - I) < sqrt(eps(real(eltype(A))))*mnp
        end
    end
end

@testset "Issue 9" begin
    data = rand(1:100, 50, 50)
    @test TSVD.tsvd(data, 2)[2] ≈ svdvals(data)[1:2]
end
