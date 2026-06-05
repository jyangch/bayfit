import pytest

from bayfit.model.local import line
from bayfit.util.info import Info


def test_from_list_dict_empty_raises():
    # from_list_dict now requires a non-empty list of dicts (see Info docstring).
    with pytest.raises((IndexError, TypeError)):
        Info.from_list_dict([])


def test_model_str_does_not_crash():
    # Model __str__ renders a rich summary string without raising.
    m = line()
    s = str(m)
    assert isinstance(s, str) and s != ''
