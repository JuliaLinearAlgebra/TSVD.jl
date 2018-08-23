# The TSVD documentation

## Functions
```@meta
CurrentModule = TSVD
DocTestSetup = quote
    using MatrixDepot, TSVD
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
