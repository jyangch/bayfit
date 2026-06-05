def test_import_data_module():
    from bayfit.data import DataUnit as DU
    from bayfit.data.data import DataUnit

    assert DU is DataUnit
