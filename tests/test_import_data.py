def test_import_data_module():
    from curvefit.data import DataUnit as DU
    from curvefit.data.data import DataUnit

    assert DU is DataUnit
