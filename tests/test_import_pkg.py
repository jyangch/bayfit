def test_import_top_level():
    from bayfit import BayesInfer, Data, DataUnit, Infer, Plot
    from bayfit.model.local import line, pl

    assert Data is not None and DataUnit is not None
    assert Infer is not None and BayesInfer is not None and Plot is not None
    assert issubclass(BayesInfer, Infer)
    assert line is not None and pl is not None
