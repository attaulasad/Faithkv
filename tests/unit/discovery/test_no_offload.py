import pytest

from kvcot.discovery.no_offload import ModelOffloadError, assert_no_offloaded_parameters


class _FakeDevice:
    def __init__(self, type_: str):
        self.type = type_


class _FakeParam:
    def __init__(self, device_type: str):
        self.device = _FakeDevice(device_type)


class _FakeModel:
    def __init__(self, params: dict, hf_device_map: dict | None = None, misleading_device: str | None = None):
        self._params = params
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map
        if misleading_device is not None:
            # A `.device` property that reports something other than the
            # truth -- the assertion must never read this at all (Blocker 1:
            # "the assertion cannot be bypassed by a misleading model.device
            # property").
            self.device = _FakeDevice(misleading_device)

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


def test_empty_model_fails_it_does_not_vacuously_pass():
    # Required correction: a model with zero parameters must FAIL, never
    # vacuously pass (an empty parameter iterator is no evidence anything is
    # actually on cuda).
    model = _FakeModel({})
    with pytest.raises(ModelOffloadError) as excinfo:
        assert_no_offloaded_parameters(model)
    assert "zero named parameters" in str(excinfo.value)


def test_model_device_reports_cpu_while_parameters_are_mixed_fails():
    # model.device lies (reports cpu) while real parameters are mixed
    # cuda/cpu -- the assertion must still fail on the real per-parameter
    # walk, never short-circuit on model.device.
    model = _FakeModel(
        {"layer0.weight": _FakeParam("cuda"), "layer1.weight": _FakeParam("cpu")},
        misleading_device="cpu",
    )
    with pytest.raises(ModelOffloadError):
        assert_no_offloaded_parameters(model)


def test_first_parameter_cuda_later_parameter_cpu_fails():
    model = _FakeModel(
        {
            "layer0.weight": _FakeParam("cuda"),
            "layer1.weight": _FakeParam("cuda"),
            "layer2.weight": _FakeParam("cpu"),
        }
    )
    with pytest.raises(ModelOffloadError) as excinfo:
        assert_no_offloaded_parameters(model)
    assert "layer2.weight" in str(excinfo.value)


def test_all_cuda_but_device_map_contains_disk_fails():
    model = _FakeModel(
        {"layer0.weight": _FakeParam("cuda"), "layer1.weight": _FakeParam("cuda")},
        hf_device_map={"layer0": "cuda:0", "layer1": "disk"},
    )
    with pytest.raises(ModelOffloadError) as excinfo:
        assert_no_offloaded_parameters(model)
    assert "layer1" in str(excinfo.value)
    assert "disk" in str(excinfo.value)


def test_device_map_containing_cpu_fails():
    model = _FakeModel(
        {"layer0.weight": _FakeParam("cuda")},
        hf_device_map={"layer0": "cpu"},
    )
    with pytest.raises(ModelOffloadError):
        assert_no_offloaded_parameters(model)


def test_device_map_containing_meta_fails():
    model = _FakeModel(
        {"layer0.weight": _FakeParam("cuda")},
        hf_device_map={"layer0": "meta"},
    )
    with pytest.raises(ModelOffloadError):
        assert_no_offloaded_parameters(model)


def test_device_map_all_cuda_passes():
    model = _FakeModel(
        {"layer0.weight": _FakeParam("cuda"), "layer1.weight": _FakeParam("cuda")},
        hf_device_map={"layer0": "cuda:0", "layer1": "cuda:1"},
    )
    assert_no_offloaded_parameters(model)  # must not raise


def test_absent_device_map_does_not_affect_all_cuda_pass():
    model = _FakeModel({"layer0.weight": _FakeParam("cuda")})
    assert not hasattr(model, "hf_device_map")
    assert_no_offloaded_parameters(model)  # must not raise


def test_misleading_device_property_alone_never_consulted():
    # model.device claims cuda, but the assertion never reads it at all --
    # a real offloaded parameter is still caught.
    model = _FakeModel(
        {"layer0.weight": _FakeParam("cpu")},
        misleading_device="cuda",
    )
    with pytest.raises(ModelOffloadError):
        assert_no_offloaded_parameters(model)
