# The TSVD documentation

## Functions
```@meta
CurrentModule = TSVD
DocTestSetup = quote
    using MatrixDepot, TSVD
    try
        matrixdepot("LPnetlib/lp_osa_30", :get)
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
