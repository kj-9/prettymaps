from prettymaps.new_api import GetArg
from dataclasses  import asdict

def test_get_arg():
    get_arg = GetArg("test-query", {})
    query, layers, radius, dilate, rotation, circle = asdict(get_arg).values()

    assert query == get_arg.query
    assert layers == get_arg.layers
    assert radius == get_arg.radius
    assert dilate == get_arg.dilate
    assert rotation == get_arg.rotation
    assert circle == get_arg.circle
