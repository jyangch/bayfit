def test_import_data_module():
    from curvefit.data.data import Data, DataUnit
    from curvefit.data import DataUnit as DU
    assert DU is DataUnit
