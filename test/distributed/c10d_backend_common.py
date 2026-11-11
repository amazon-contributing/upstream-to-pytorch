# Owner(s): ["oncall: distributed"]

import os
import unittest
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_CUDA


STANDARD_DTYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
)

FLOAT8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)

COMPLEX_DTYPES = (
    torch.complex64,
    torch.complex128,
)


@dataclass(frozen=True)
class BackendConfig:
    name: str
    device_type: str
    supports_coalescing: bool = False
    supports_bitwise_reductions: bool = False
    supports_cuda_graph_barrier: bool = False
    supports_dropped_p2p_work: bool = False
    supports_sequence_numbers: bool = True
    supports_collectives_timing: bool = False
    supports_work_sequence_number: bool = False
    supports_work_result: bool = False
    supports_gather_single: bool = False
    supports_uneven_all_gather: bool = False
    # True when all_to_all_single accepts a non-contiguous input instead of
    # raising "Tensors must be contiguous" like Gloo and NCCL.
    supports_noncontiguous_all_to_all: bool = False
    dtypes: tuple[torch.dtype, ...] = STANDARD_DTYPES
    float8_dtypes: tuple[torch.dtype, ...] = ()
    premul_sum_dtypes: tuple[torch.dtype, ...] = ()
    complex_dtypes: tuple[torch.dtype, ...] = COMPLEX_DTYPES


C10D_BACKENDS = (
    BackendConfig(
        "gloo",
        "cpu",
        supports_bitwise_reductions=True,
        supports_work_sequence_number=True,
    ),
    BackendConfig(
        "nccl-legacy",
        "cuda",
        supports_coalescing=True,
        supports_dropped_p2p_work=True,
        supports_collectives_timing=True,
        supports_work_sequence_number=True,
        supports_work_result=True,
        supports_gather_single=True,
        supports_uneven_all_gather=True,
        float8_dtypes=FLOAT8_DTYPES,
        premul_sum_dtypes=(torch.float16, torch.float32, torch.float64),
    ),
    BackendConfig(
        "nccl2",
        "cuda",
        supports_coalescing=True,
        supports_cuda_graph_barrier=True,
        supports_dropped_p2p_work=True,
        supports_collectives_timing=True,
        supports_work_sequence_number=True,
        supports_work_result=True,
        supports_gather_single=True,
        supports_uneven_all_gather=True,
        float8_dtypes=FLOAT8_DTYPES,
        premul_sum_dtypes=(
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ),
    ),
    # nccl-lazy wraps a primary ProcessGroupNCCL (all collectives delegate to
    # it) plus lazily-built per-peer P2P comms. Timing and sequence APIs are not
    # forwarded.
    BackendConfig(
        "nccl-lazy",
        "cuda",
        supports_coalescing=True,
        supports_cuda_graph_barrier=True,
        supports_dropped_p2p_work=True,
        supports_sequence_numbers=False,
        supports_work_result=True,
        supports_gather_single=True,
        supports_uneven_all_gather=True,
        float8_dtypes=FLOAT8_DTYPES,
        premul_sum_dtypes=(
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ),
    ),
)

# The torch-neuronx c10d backend. It exists only when torch_neuronx registered
# it (torch_neuronx/distributed/backend.py, _register_neuron_backend). A flag
# is True only if NeuronBackend implements the feature; each False flag names
# the missing feature.
NEURON_BACKENDS = ()
if dist.is_available() and dist.is_backend_available("neuron"):
    NEURON_BACKENDS = (
        BackendConfig(
            "neuron",
            "neuron",
            # NeuronBackend.h:33 returns True for supportsCoalescing(), and
            # allreduce_coalesced, allgather_into_tensor_coalesced and
            # reduce_scatter_tensor_coalesced are implemented
            # (NeuronBackend.cpp:1530, 1662, 1718). Between start/endCoalescing
            # it queues only send/recv (NeuronBackend.cpp:539-541, 581-583).
            supports_coalescing=True,
            # False: BAND/BOR/BXOR are not supported. _reduce_scatter_base
            # accepts only SUM, AVG, MIN, MAX, PREMUL_SUM
            # (NeuronBackend.cpp:1042-1046).
            supports_bitwise_reductions=False,
            # True: the watchdog keeps each Work until it completes
            # (NeuronBackend.cpp:388-392, NeuronWatchdog.cpp:77-83, 132-134).
            supports_dropped_p2p_work=True,
            # False: NeuronBackend.h does not override
            # getSequenceNumberForGroup, so the c10d::Backend default throws.
            supports_sequence_numbers=False,
            # False: NeuronBackend.h does not override enableCollectivesTiming,
            # and NeuronWork::getDuration requires timing (NeuronWork.cpp:187).
            supports_collectives_timing=False,
            # False: NeuronWork::getSequenceNumber (NeuronWork.h:251) does not
            # override c10d::Work::getSequencenumber.
            supports_work_sequence_number=False,
            # False: NeuronWork does not override c10d::Work::getFutureResult.
            supports_work_result=False,
            # False: NeuronBackend.h does not override gather_single or
            # gather_into_tensor.
            supports_gather_single=False,
            # False: allgather sizes its staging tensor from output_list[0]
            # (NeuronBackend.cpp:989-991), so uneven outputs are not supported.
            supports_uneven_all_gather=False,
            # True: all_to_all_single accepts a strided input and returns the
            # result of its contiguous copy. torch-neuronx tests this in
            # tests/distributed/collective_ops/test_strided_collectives.py
            # (TestStridedAllToAll.test_all_to_all_single_strided_input);
            # context-parallel and MoE code pass strided views.
            supports_noncontiguous_all_to_all=True,
            # All standard dtypes are accepted. f32, f16, bf16, int32 and bool
            # are used as is; f64 is cast to f32, and int64, int8 and uint8 to
            # int32 (DispatchUtils.cpp:227-241, 357; copy-back at
            # NeuronBackend.cpp:309-318).
            dtypes=STANDARD_DTYPES,
            # Empty: test_float8_transport reads the CUDA device capability.
            # The backend casts float8 to f32 (DispatchUtils.cpp:236).
            float8_dtypes=(),
            # PREMUL_SUM multiplies the input by the factor, then runs SUM
            # (NeuronBackend.cpp:781-789).
            premul_sum_dtypes=(
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ),
            # complex64 is used as is; complex128 is cast to complex64
            # (DispatchUtils.cpp:227-229, 239-240).
            complex_dtypes=COMPLEX_DTYPES,
        ),
    )

C10D_BACKENDS += NEURON_BACKENDS

CUDA_BACKENDS = tuple(
    backend for backend in C10D_BACKENDS if backend.device_type == "cuda"
)


class C10dBackendTest:
    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        if self.device_type != "cpu":
            return torch.device(self.device_type, self.rank)
        return torch.device(self.device_type)

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        os.environ["LOCAL_RANK"] = str(self.rank)
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        _set_accelerator_device_after_init(self.device_type, self.rank)


def _set_accelerator_device_after_init(device_type, rank):
    # CUDA sets the device before init_process_group. Other accelerators set
    # it after: the Neuron backend requires that the Neuron runtime is not
    # initialized before init_process_group (torch_neuronx/distributed/
    # backend.py, _neuron_runtime_setup), and a device query can initialize it.
    if device_type not in ("cpu", "cuda"):
        torch.accelerator.set_device_index(rank)


def _accelerator_device_count(device_type):
    acc = torch.accelerator.current_accelerator()
    if acc is None or acc.type != device_type:
        return 0
    return torch.accelerator.device_count()


class C10dBackendTestContinuous:
    """Reuse workers, creating a fresh process group before each test method."""

    world_size = 2
    timeout = timedelta(seconds=60)

    @classmethod
    def backend_str(cls):
        return cls.backend_name

    @property
    def device(self):
        if self.device_type != "cpu":
            return torch.device(self.device_type, self.rank)
        return torch.device(self.device_type)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        if "_store_prefix" not in cls.__dict__:
            cls._store_prefix = rdvz_file
        if cls.device_type == "cuda":
            torch.cuda.set_device(rank)
        super()._init_pg(rank, world_size, rdvz_file)
        _set_accelerator_device_after_init(cls.device_type, rank)

    @classmethod
    def _run_test_given_id(cls, test_id, **kwargs):
        # Reuse Python workers, not communicators. Keep the first collective
        # in every method on a freshly initialized process group.
        iteration = cls.__dict__.get("_test_iteration", 0)
        if iteration:
            dist.destroy_process_group()
            cls.pg = None
            # Never race deletion of the previous FileStore's rendezvous file.
            cls._init_pg(cls.rank, cls.world_size, f"{cls._store_prefix}.{iteration}")
        cls._test_iteration = iteration + 1
        super()._run_test_given_id(test_id, **kwargs)


def instantiate_backend_tests(
    namespace, suite_name, base_class, backends, *, harness=MultiProcessTestCase
):
    for backend in backends:
        backend_name = backend.name.replace("-", " ").title().replace(" ", "")
        class_name = f"{backend_name}{suite_name}Test"
        test_class = type(
            class_name,
            (base_class, harness),
            {
                "__module__": namespace["__name__"],
                "backend_name": backend.name,
                "device_type": backend.device_type,
                "supports_coalescing": backend.supports_coalescing,
                "supports_bitwise_reductions": backend.supports_bitwise_reductions,
                "supports_cuda_graph_barrier": backend.supports_cuda_graph_barrier,
                "supports_dropped_p2p_work": backend.supports_dropped_p2p_work,
                "supports_sequence_numbers": backend.supports_sequence_numbers,
                "supports_collectives_timing": backend.supports_collectives_timing,
                "supports_work_sequence_number": backend.supports_work_sequence_number,
                "supports_work_result": backend.supports_work_result,
                "supports_gather_single": backend.supports_gather_single,
                "supports_uneven_all_gather": backend.supports_uneven_all_gather,
                "supports_noncontiguous_all_to_all": (
                    backend.supports_noncontiguous_all_to_all
                ),
                "dtypes": backend.dtypes,
                "float8_dtypes": backend.float8_dtypes,
                "premul_sum_dtypes": backend.premul_sum_dtypes,
                "complex_dtypes": backend.complex_dtypes,
            },
        )
        test_class = unittest.skipIf(
            not dist.is_backend_available(backend.name),
            f"{backend.name} backend is not available",
        )(test_class)
        if backend.device_type == "cuda":
            test_class = unittest.skipIf(
                not TEST_CUDA or torch.cuda.device_count() < 2,
                f"{backend.name} requires 2+ GPUs",
            )(test_class)
        elif backend.device_type != "cpu":
            test_class = unittest.skipIf(
                _accelerator_device_count(backend.device_type) < 2,
                f"{backend.name} requires 2+ {backend.device_type} devices",
            )(test_class)
        namespace[class_name] = test_class
