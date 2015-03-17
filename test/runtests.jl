using TSVD
using Base.Test

m, n, p = 10, 6, 0.8
mnp = round(Integer, m*n*p)

for A in (randn(m, n),
          complex(randn(m, n),
          randn(m, n)),
          sprandn(m, n, p),
          sparse(rand(1:m, mnp), rand(1:n, mnp), complex(randn(mnp), randn(mnp)), m, n))
          U, s, V = tsvd(A, n)
          @test_approx_eq A U*Diagonal(s)*V'
end
