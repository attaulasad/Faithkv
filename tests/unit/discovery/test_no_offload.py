import pytest

from kvcot.discovery.no_offload import ModelOffloadError, assert_no_offloaded_parameters


class _FakeDevice:
    def __init__(self, type_: str):
        self.type = type_


class _FakeParam:
    def __init__(self, device_type: str):
        self.device = _FakeDevice(device_type)


class _FakeModel:
    def __init__(self, params: dict):
        self._params = params

    def named_parameters(self):
        return iter(self._params.items())


def test_all_cuda_parameters_pass():
    model = _FakeModel({"layer0.weight": _FakeParam("cuda"), "layer1.weight": _FakeParam("cuda")})
    assert_no_offloaded_parameters(model)  # must not raise


def test_single_offloaded_parameter_raises_with_name_and_device():
    model = _FakeModel({"layer0.weight": _FakeParam("cuda"), "layer1.weight": _FakeParam("cpu")})
    with pytest.raises(ModelOffloadError) as excinfo:
        assert_no_offloaded_parameters(model)
    assert "layer1.weight" in str(excinfo.value)
    assert "cpu" in str(excinfo.value)
    assert "layer0.weight" not in str(excinfo.value)


def test_multiple_offloaded_parameters_all_named():
    model = _FakeModel(
        {
            "a": _FakeParam("cpu"),
            "b": _FakeParam("cuda"),
            "c": _FakeParam("meta"),
        }
    )
    with pytest.raises(ModelOffloadError) as excinfo:
        assert_no_offloaded_parameters(model)
    message = str(excinfo.value)
    assert "'a'" in message
    assert "'c'" in message
    assert "'b'" not in message


def test_empty_model_passes_trivially():
    model = _FakeModel({})
    assert_no_offloaded_parameters(model)
