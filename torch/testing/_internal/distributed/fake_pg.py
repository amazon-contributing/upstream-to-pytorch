# mypy: allow-untyped-defs

# The fake backend lives in torch.distributed._backends.fake and is registered
# through the "torch.distributed.backends" entry point. Importing this module
# still registers it, as it did before the move.
import torch
import torch.distributed as dist
from torch.distributed._backends.fake import (
    _create_fake_pg,
    _register_fake_backend,
    FakeProcessGroup,
    FakeStore,
)


__all__ = ["FakeProcessGroup", "FakeStore"]

_register_fake_backend()

# The built-in registration lists cpu, cuda, hpu and xpu only, so a fake group
# has no backend for a PrivateUse1 device and every collective on its tensors
# fails with "No backend type associated with device type". Add that device.
_pu1_backend = torch._C._get_privateuse1_backend_name()
_fake_devices = dist.Backend.backend_capability[dist.Backend.FAKE]
if _pu1_backend != "privateuseone" and _pu1_backend not in _fake_devices:
    dist.Backend.register_backend(
        dist.Backend.FAKE,
        _create_fake_pg,
        extended_api=True,
        devices=[*_fake_devices, _pu1_backend],
    )
