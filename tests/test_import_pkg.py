def test_import_top_level():
    import curvefit
    from curvefit import Data, DataUnit, Infer, BayesInfer, Plot
    from curvefit.model.local import ln, pl
    assert Data is not None and DataUnit is not None
    assert Infer is not None and BayesInfer is not None and Plot is not None
    assert issubclass(BayesInfer, Infer)
    assert ln is not None and pl is not None
