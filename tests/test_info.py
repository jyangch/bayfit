from curvefit.util.info import Info
from curvefit.model.local import ln


def test_from_list_dict_empty_returns_empty_table():
    info = Info.from_list_dict([])
    assert info.data_dict == {}


def test_model_str_with_empty_config():
    # ln has no config; __str__ must not crash on empty all_config
    m = ln()
    assert str(m) == ''
