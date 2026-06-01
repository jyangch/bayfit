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
