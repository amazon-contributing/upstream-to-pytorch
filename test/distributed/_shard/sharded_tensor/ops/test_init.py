# Adapted from upstream — made device-agnostic for PrivateUse1 backends.
# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

if torch.accelerator.current_accelerator() is None:
    print("No accelerator available, skipping tests", file=sys.stderr)
    sys.exit(0)

DEVICE_TYPE = torch.accelerator.current_accelerator().type
BACKEND = dist.get_default_backend_for_device(DEVICE_TYPE)


class TestShardedTensorNNInit(ShardedTensorTestBase):
    """Testing torch.nn.init functions for ShardedTensor"""

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend()
    def test_init_sharded_tensor_with_uniform(self):
        """Test torch.nn.init.uniform_(ShardedTensor, a, b)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{DEVICE_TYPE}:0",
                f"rank:1/{DEVICE_TYPE}:1",
                f"rank:2/{DEVICE_TYPE}:2",
                f"rank:3/{DEVICE_TYPE}:3",
            ],
        )
        h, w = 8, 2
        a, b = 10, 20

        seed = 1234
        dtype = torch.double

        st = sharded_tensor.empty(spec, h, w, dtype=dtype)
        self.assertEqual(1, len(st.local_shards()))

        # Clone local tensor to ensure torch.nn.init starts from the same input
        local_tensor_clone = torch.clone(st.local_shards()[0].tensor)
        torch.manual_seed(seed)
        torch.nn.init.uniform_(st, a=a, b=b)

        torch.manual_seed(seed)
        torch.nn.init.uniform_(local_tensor_clone, a=a, b=b)
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend()
    def test_init_sharded_tensor_with_normal(self):
        """Test torch.nn.init.normal_(ShardedTensor, mean, std)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{DEVICE_TYPE}:0",
                f"rank:1/{DEVICE_TYPE}:1",
                f"rank:2/{DEVICE_TYPE}:2",
                f"rank:3/{DEVICE_TYPE}:3",
            ],
        )
        h, w = 8, 2
        mean, std = 10, 5

        seed = 1234
        dtype = torch.double

        st = sharded_tensor.empty(spec, h, w, dtype=dtype)
        self.assertEqual(1, len(st.local_shards()))

        # Clone local tensor to ensure torch.nn.init starts from the same input
        local_tensor_clone = torch.clone(st.local_shards()[0].tensor)
        torch.manual_seed(seed)
        torch.nn.init.normal_(st, mean=mean, std=std)

        torch.manual_seed(seed)
        torch.nn.init.normal_(local_tensor_clone, mean=mean, std=std)
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend()
    def test_init_sharded_tensor_with_kaiming_uniform(self):
        """Test torch.nn.init.kaiming_uniform_(ShardedTensor, a, mode, nonlinearit)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{DEVICE_TYPE}:0",
                f"rank:1/{DEVICE_TYPE}:1",
                f"rank:2/{DEVICE_TYPE}:2",
                f"rank:3/{DEVICE_TYPE}:3",
            ],
        )
        h, w = 8, 2
        a, mode, nonlinearity = 0, "fan_in", "leaky_relu"

        seed = 1234
        dtype = torch.double

        st = sharded_tensor.empty(spec, h, w, dtype=dtype)
        self.assertEqual(1, len(st.local_shards()))

        # Clone local tensor to ensure torch.nn.init starts from the same input
        local_tensor_clone = torch.clone(st.local_shards()[0].tensor)
        torch.manual_seed(seed)
        torch.nn.init.kaiming_uniform_(st, a=a, mode=mode, nonlinearity=nonlinearity)

        torch.manual_seed(seed)
        torch.nn.init.kaiming_uniform_(
            local_tensor_clone, a=a, mode=mode, nonlinearity=nonlinearity
        )
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)


if __name__ == "__main__":
    run_tests()
