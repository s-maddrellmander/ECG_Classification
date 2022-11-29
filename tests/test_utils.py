import logging

from utils import Timer


def test_Timer():
    T = Timer("test timer")
    T.__enter__()
    assert type(T.t) is float
    T.__exit__()
