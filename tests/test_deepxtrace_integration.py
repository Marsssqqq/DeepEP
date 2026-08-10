import gc
import importlib.util
import inspect
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_buffer_module():
    """Load buffer.py with host-only stubs for torch and the CUDA extension."""
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = type("Tensor", (), {})
    torch_module.Stream = type("Stream", (), {})
    torch_module.Size = tuple
    torch_module.dtype = type("dtype", (), {})
    torch_module.cuda = types.SimpleNamespace()

    dist_module = types.ModuleType("torch.distributed")
    dist_module.ProcessGroup = type("ProcessGroup", (), {})
    dist_module.all_gather_object = lambda outputs, value, group: None
    torch_module.distributed = dist_module

    cpp_module = types.ModuleType("deep_ep_cpp")
    cpp_module.Buffer = type("Buffer", (), {})
    cpp_module.Config = type("Config", (), {})
    cpp_module.EventHandle = type("EventHandle", (), {})

    package_module = types.ModuleType("deep_ep")
    package_module.__path__ = []
    utils_module = types.ModuleType("deep_ep.utils")

    class EventOverlap:

        def __init__(self, *args):
            self.args = args

    utils_module.EventOverlap = EventOverlap
    utils_module.check_nvlink_connections = lambda group: None

    module_name = "deep_ep._buffer_host_test"
    module_path = Path(__file__).resolve().parents[1] / "deep_ep" / "buffer.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "torch": torch_module,
        "torch.distributed": dist_module,
        "deep_ep_cpp": cpp_module,
        "deep_ep": package_module,
        "deep_ep.utils": utils_module,
        module_name: module,
    }
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


buffer_module = _load_buffer_module()


class _FakeGroup:

    def rank(self):
        return 0

    def size(self):
        return 2


class _FakeRuntime:

    def __init__(self, *args):
        self.args = args
        self.synced = False

    def get_local_device_id(self):
        return 0

    def get_local_ipc_handle(self):
        return bytearray()

    def get_num_rdma_ranks(self):
        return 1

    def sync(self, *args):
        self.synced = True

    def is_available(self):
        return self.synced


class TestDeepXTraceIntegration(unittest.TestCase):

    @staticmethod
    def compatible_module():
        return types.SimpleNamespace(NORMAL_STATS_SCHEMA=buffer_module._REQUIRED_NORMAL_STATS_SCHEMA)

    def test_optional_defaults_are_backward_compatible(self):
        parameters = inspect.signature(buffer_module.Buffer.__init__).parameters
        self.assertIs(parameters["low_latency_mode"].default, False)
        self.assertIs(parameters["enable_deepxtrace"].default, False)

    def test_optional_dependency_loading_is_fail_closed(self):
        with patch.object(buffer_module.importlib, "import_module") as import_module:
            diagnose_module, error = buffer_module._load_deepxtrace(False, True)
        import_module.assert_not_called()
        self.assertIsNone(diagnose_module)
        self.assertEqual(error, "disabled by configuration")

        with patch.object(buffer_module.importlib, "import_module", return_value=self.compatible_module()):
            diagnose_module, error = buffer_module._load_deepxtrace(True, True)
        self.assertIsNotNone(diagnose_module)
        self.assertIsNone(error)

        for exception in (
                ModuleNotFoundError("No module named 'deepxtrace'"),
                OSError("broken optional dependency"),
                AttributeError("incomplete installation"),
        ):
            with self.subTest(exception=exception):
                with patch.object(buffer_module.importlib, "import_module", side_effect=exception):
                    diagnose_module, error = \
                        buffer_module._load_deepxtrace(True, True)
                self.assertIsNone(diagnose_module)
                self.assertIn(type(exception).__name__, error)

    def test_schema_mismatch_is_reported_as_unavailable(self):
        incompatible_module = types.SimpleNamespace(NORMAL_STATS_SCHEMA=("legacy", ))
        with self.assertRaisesRegex(RuntimeError, "deepxtrace>=0.2.0"):
            buffer_module._validate_deepxtrace_normal_stats_schema(incompatible_module)

        with patch.object(buffer_module.importlib, "import_module", return_value=incompatible_module):
            diagnose_module, error = buffer_module._load_deepxtrace(True, True)
        self.assertIsNone(diagnose_module)
        self.assertIn("RuntimeError", error)

    def test_rank_wide_mixed_enablement_disables_diagnosis(self):

        def all_gather_object(outputs, value, group):
            if isinstance(value, tuple) and len(value) == 4:
                outputs[:] = [
                    value,
                    (False, False, "disabled by configuration", value[3]),
                ]
            else:
                outputs[:] = [value, value]

        with patch.object(buffer_module.deep_ep_cpp, "Buffer", _FakeRuntime), \
                patch.object(buffer_module.dist, "all_gather_object",
                             side_effect=all_gather_object), \
                patch.object(buffer_module, "_load_deepxtrace",
                             return_value=(self.compatible_module(), None)), \
                self.assertWarnsRegex(RuntimeWarning, "disabled for all ranks"):
            buffer = buffer_module.Buffer(_FakeGroup(), enable_deepxtrace=True)
        self.assertFalse(buffer.deepxtrace_enabled)
        self.assertTrue(buffer.runtime.is_available())

    def test_rank_wide_async_mode_mismatch_raises(self):

        def all_gather_object(outputs, value, group):
            outputs[:] = [value, (True, True, None, not value[3])]

        with patch.object(buffer_module.dist, "all_gather_object",
                          side_effect=all_gather_object), \
                patch.object(buffer_module, "_load_deepxtrace",
                             return_value=(self.compatible_module(), None)), \
                self.assertRaisesRegex(RuntimeError,
                                       "enable_deepxtrace_async"):
            buffer_module.Buffer(_FakeGroup(), enable_deepxtrace=True)

    def test_sync_collection_contract(self):
        buffer = buffer_module.Buffer.__new__(buffer_module.Buffer)
        buffer.diagnose = None
        buffer.enable_deepxtrace_async = False
        self.assertIsNone(buffer.diagnose_normal_sync())

        buffer.diagnose = MagicMock()
        buffer.enable_deepxtrace_async = True
        with self.assertRaisesRegex(RuntimeError, "enable_deepxtrace_async=False"):
            buffer.diagnose_normal_sync()

        expected = [{"probe": "normal", "status": "ok"}]
        buffer.enable_deepxtrace_async = False
        buffer.diagnose.diagnose_normal_sync.return_value = expected
        self.assertIs(buffer.diagnose_normal_sync(17), expected)
        buffer.diagnose.diagnose_normal_sync.assert_called_once_with(17)

    def test_destroy_stops_async_diagnosis_before_runtime(self):
        order = []
        buffer = buffer_module.Buffer.__new__(buffer_module.Buffer)
        buffer.explicitly_destroy = True
        buffer.enable_deepxtrace_async = True
        buffer.diagnose = MagicMock()
        buffer.diagnose.stop_async_diagnose.side_effect = \
            lambda: order.append("diagnose")
        buffer.runtime = MagicMock()
        buffer.runtime.destroy.side_effect = lambda: order.append("runtime")
        buffer._deepxtrace_finalizer = weakref.finalize(buffer, buffer.diagnose.stop_async_diagnose)
        buffer._normal_diagnose_stats = {"probe": object()}
        buffer._normal_notify_full_kernel_timer_states = object()
        buffer._normal_notify_dispatch_full_kernel_timer_state = object()
        buffer._normal_cached_notify_dispatch_full_kernel_timer_state = object()
        buffer._normal_cached_notify_combine_full_kernel_timer_state = object()

        buffer.destroy()

        self.assertEqual(order, ["diagnose", "runtime"])
        self.assertIsNone(buffer.runtime)
        self.assertIsNone(buffer.diagnose)
        self.assertFalse(buffer._deepxtrace_finalizer)
        self.assertTrue(all(value is None for value in buffer._normal_diagnose_stats.values()))

    def test_finalizer_stops_async_diagnosis_on_implicit_cleanup(self):
        stop_async_diagnose = MagicMock()
        buffer = buffer_module.Buffer.__new__(buffer_module.Buffer)
        buffer._deepxtrace_finalizer = weakref.finalize(buffer, stop_async_diagnose)
        buffer_ref = weakref.ref(buffer)

        del buffer
        gc.collect()

        self.assertIsNone(buffer_ref())
        stop_async_diagnose.assert_called_once_with()

    def test_low_latency_stats_remain_caller_owned(self):
        buffer = buffer_module.Buffer.__new__(buffer_module.Buffer)
        buffer.nvshmem_qp_depth = 1024
        buffer.diagnose = MagicMock()
        buffer.runtime = MagicMock()
        buffer.runtime.low_latency_dispatch.return_value = (object(), object(), object(), object(), object(), object(), object())
        x = MagicMock()
        x.size.return_value = 64

        buffer.low_latency_dispatch(x, MagicMock(), 1, 1)

        dispatch_args = buffer.runtime.low_latency_dispatch.call_args.args
        self.assertIsNone(dispatch_args[3])

        buffer.runtime.low_latency_combine.return_value = (object(), object(), object())
        handle = (object(), object(), 1, 64, 1)
        buffer.low_latency_combine(MagicMock(), MagicMock(), MagicMock(), handle)

        combine_args = buffer.runtime.low_latency_combine.call_args.args
        self.assertIsNone(combine_args[11])
        buffer.diagnose.get_stats_ll_stats_tensor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
