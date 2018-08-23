# The TSVD documentation

## Functions
```@meta
CurrentModule = TSVD
DocTestSetup = quote
    using MatrixDepot, TSVD, Random
    Random.seed!(123)
    try
        matrixdepot("Rucci/Rucci1", :get)
    catch
    	nothing
    end
end
```

```@docs
tsvd
```

```@meta
DocTestSetup = nothing
```
