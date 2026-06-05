from curvefit.model.model import Additive, Mathematic, Model, Multiplicative


def test_base_classes_lock_type():
    class A(Additive):
        pass

    class M(Multiplicative):
        pass

    class H(Mathematic):
        pass

    assert A().type == 'add'
    assert M().type == 'mul'
    assert H().type == 'math'


def test_type_setter_is_noop_on_subclasses():
    class A(Additive):
        pass

    a = A()
    a.type = 'mul'  # locked; setter is a no-op
    assert a.type == 'add'


def test_base_model_default_type():
    assert Model().type == 'add'


def test_composite_type_add_plus_add():
    from curvefit.model.local import line, pl

    assert (line() + pl()).type == 'add'


def test_composite_type_add_times_mul():
    from curvefit.model.local import expcut, pl

    assert (pl() * expcut()).type == 'add'


def test_illegal_composite_raises():
    import pytest

    from curvefit.model.local import line, pl

    with pytest.raises(ValueError):
        _ = (line() * pl()).type  # add * add is illegal


def test_const_plus_add_is_add():
    from curvefit.model.local import const, line

    assert (line() + const()).type == 'add'
