# Owner(s): ["oncall: distributed"]
# Adapted from test/distributed/algorithms/quantization/test_quantization.py
# Converts NCCL/CUDA tests to use privateuse1/accelerator backend.
# Gloo-only tests (all_gather) kept as-is. NCCL tests (all_to_all) adapted to accelerator.

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.algorithms._quantization.quantization as quant
from torch.distributed.algorithms._quantization.quantization import DQuantType
from torch.testing._internal.common_distributed import (
    init_multigpu_helper,
    MultiProcessTestCase,
    requires_accelerator_dist_backend,
    requires_gloo,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


torch.backends.cuda.matmul.allow_tf32 = False

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


def _build_tensor(size, value=None, dtype=torch.float, device=None):
    if value is None:
        value = size
    if device is None:
        return torch.empty(size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, dtype=dtype).fill_(value).to(device)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator(check_available=True))
    else "cpu"
)
BACKEND = dist.get_default_backend_for_device(device_type)


class DistQuantizationTests(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()
        torch.backends.cudnn.flags(enabled=True, allow_tf32=False).__enter__()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def op_timeout_sec(self):
        return 1

    @property
    def world_size(self):
        return 2

    @requires_gloo()
    def test_all_gather_fp16(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend="gloo"
        )
        group = list(range(self.world_size))
        group_id = dist.group.WORLD
        self._test_all_gather(
            group, group_id, self.rank, dtype=torch.float32, qtype=DQuantType.FP16
        )

    @requires_gloo()
    def test_all_gather_bfp16(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend="gloo"
        )
        group = list(range(self.world_size))
        group_id = dist.group.WORLD
        self._test_all_gather(
            group, group_id, self.rank, dtype=torch.float32, qtype=DQuantType.BFP16
        )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_all_to_all_fp16(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=BACKEND
        )
        group = list(range(self.world_size))
        group_id = dist.new_group(range(self.world_size))
        rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
        device = torch.device(device_type, rank_to_GPU[self.rank][0])
        self._test_all_to_all(
            group,
            group_id,
            self.rank,
            device=device,
            dtype=torch.float32,
            qtype=DQuantType.FP16,
        )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_all_to_all_bfp16(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=BACKEND
        )
        group = list(range(self.world_size))
        group_id = dist.new_group(range(self.world_size))
        rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
        device = torch.device(device_type, rank_to_GPU[self.rank][0])
        self._test_all_to_all(
            group,
            group_id,
            self.rank,
            device=device,
            dtype=torch.float32,
            qtype=DQuantType.BFP16,
        )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_all_to_all_single_fp16(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=BACKEND
        )
        group = list(range(self.world_size))
        group_id = dist.new_group(range(self.world_size))
        rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
        device = torch.device(device_type, rank_to_GPU[self.rank][0])
        self._test_all_to_all_single(
            group,
            group_id,
            self.rank,
            device=device,
            dtype=torch.float32,
            qtype=DQuantType.FP16,
        )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_all_to_all_single_bfp16(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=BACKEND
        )
        group = list(range(self.world_size))
        group_id = dist.new_group(range(self.world_size))
        rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
        device = torch.device(device_type, rank_to_GPU[self.rank][0])
        self._test_all_to_all_single(
            group,
            group_id,
            self.rank,
            device=device,
            dtype=torch.float32,
            qtype=DQuantType.BFP16,
        )

    # --- Helper methods ---

    def _test_all_gather(
        self,
        group,
        group_id,
        rank,
        device=None,
        dtype=torch.float,
        qtype=None,
    ):
        for dest in group:
            tensor = _build_tensor([dest + 1, dest + 1], rank, dtype=dtype)
            tensors = [
                _build_tensor([dest + 1, dest + 1], -1, dtype=dtype) for i in group
            ]
            expected_tensors = [
                _build_tensor([dest + 1, dest + 1], i, dtype=dtype) for i in group
            ]
            if device is not None:
                tensor = tensor.to(device)
                tensors = [t.to(device) for t in tensors]
            allgather = quant.auto_quantize(dist.all_gather, qtype, quant_loss=None)
            allgather(tensors, tensor, group=group_id, async_op=False)

            for t1, t2 in zip(tensors, expected_tensors):
                self.assertEqual(t1, t2)

    def _test_all_to_all(
        self,
        group,
        group_id,
        rank,
        device=None,
        dtype=torch.float,
        qtype=None,
    ):
        if group_id is not None:
            size = len(group)
            in_splits = [i + 1 for i in group]
            in_tensors = [
                torch.ones([in_splits[i], size], dtype=dtype) * rank
                for i, _ in enumerate(group)
            ]
            out_tensors = [
                torch.ones([(rank + 1), size], dtype=dtype) for _ in group
            ]
            expected_tensors = [
                torch.ones([rank + 1, size], dtype=dtype) * i for i in group
            ]
            if device is not None:
                in_tensors = [t.to(device) for t in in_tensors]
                expected_tensors = [t.to(device) for t in expected_tensors]
                out_tensors = [t.to(device) for t in out_tensors]
            quantize_alltoall = quant.auto_quantize(
                dist.all_to_all, qtype, quant_loss=None
            )
            quantize_alltoall(out_tensors, in_tensors, group=group_id)
            for t1, t2 in zip(out_tensors, expected_tensors):
                self.assertEqual(t1, t2)

    def _test_all_to_all_single(
        self,
        group,
        group_id,
        rank,
        device=None,
        dtype=torch.float,
        qtype=DQuantType.FP16,
    ):
        if group_id is not None:
            size = len(group)
            in_splits = [i + 1 for i in group]
            out_splits = [rank + 1 for _ in group]
            in_tensor = torch.ones([sum(in_splits), size], dtype=dtype) * rank
            out_tensor = torch.ones([(rank + 1) * size, size], dtype=dtype)
            expected_tensor = torch.cat(
                [torch.ones([rank + 1, size], dtype=dtype) * i for i in group]
            )
            if device is not None:
                in_tensor = in_tensor.to(device)
                expected_tensor = expected_tensor.to(device)
                out_tensor = out_tensor.to(device)
                quantize_alltoall_single = quant.auto_quantize(
                    dist.all_to_all_single, qtype, quant_loss=None
                )
                quantize_alltoall_single(
                    out_tensor,
                    in_tensor,
                    out_splits=out_splits,
                    in_splits=in_splits,
                    group=group_id,
                )
                self.assertEqual(out_tensor, expected_tensor)


if __name__ == "__main__":
    run_tests()
