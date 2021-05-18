using Documenter, TSVD

DocMeta.setdocmeta!(
    TSVD,
    :DocTestSetup,
    :(using TSVD, MatrixDepot);
    recursive=true)

makedocs(sitename="TSVD Documentation")

deploydocs(
    repo = "github.com/andreasnoack/TSVD.jl.git",
)
