if Base.HOME_PROJECT[] !== nothing
    # JuliaLang/julia/pull/28625
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end

using Documenter, TSVD

makedocs()

deploydocs(
    deps   = Deps.pip("mkdocs==0.17.5", "python-markdown-math", "mkdocs-material==2.9.4"),
    repo = "github.com/andreasnoack/TSVD.jl.git",
    julia  = "1.0"
)
