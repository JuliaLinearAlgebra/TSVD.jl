push!(LOAD_PATH,"../src/")

using Documenter, TSVD

makedocs(sitename="TSVD Documentation")

deploydocs(
    repo = "github.com/andreasnoack/TSVD.jl.git",
)
