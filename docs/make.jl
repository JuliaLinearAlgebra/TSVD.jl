using Documenter, TSVD

makedocs()

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math", "mkdocs-material"),
    repo = "github.com/andreasnoack/TSVD.jl.git",
    julia  = "1.0"
)
