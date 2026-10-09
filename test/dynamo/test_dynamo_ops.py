# Owner(s): ["module: dynamo"]

"""
Tests for operator behavior during Dynamo tracing.

This file contains tests that use op_db to verify correct behavior of operators
when traced by Dynamo.
"""

import os
from functools import partial

import torch
import torch._dynamo.test_case
import torch._functorch.config as _functorch_config
from torch._dynamo.comptime import comptime, ComptimeContext
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
    skip,
    skipOps,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs,
    foreach_binary_op_db,
    foreach_other_op_db,
    foreach_pointwise_op_db,
    foreach_reduce_op_db,
    foreach_unary_op_db,
    generate_elementwise_binary_extremal_value_tensors,
    generate_elementwise_binary_large_value_tensors,
    generate_elementwise_binary_small_value_tensors,
    generate_elementwise_binary_tensors,
    generate_elementwise_unary_extremal_value_tensors,
    generate_elementwise_unary_large_value_tensors,
    generate_elementwise_unary_small_value_tensors,
    generate_elementwise_unary_tensors,
    op_db,
    reduction_ops,
    unary_ufuncs,
)
from torch.testing._internal.common_utils import TestGradients


# Ops that fail the inplace requires_grad propagation test for known reasons
test_inplace_ops_propagate_requires_grad_metadata_skips = {
    # Not implemented for floating point types
    skip("bitwise_and"),
    skip("bitwise_left_shift"),
    skip("bitwise_or"),
    skip("bitwise_right_shift"),
    skip("bitwise_xor"),
    skip("gcd"),
    skip("lcm"),
    # out=... arguments don't support automatic differentiation
    skip("ldexp"),
    # Backward not implemented
    skip("floor_divide"),
    skip("heaviside"),
    skip("nextafter"),
    # Output does not require grad (logical ops return bool)
    skip("logical_and"),
    skip("logical_or"),
    skip("logical_xor"),
    skip("resize_as_"),
    # Dtype issues
    skip("float_power"),
    # Numerical gradient mismatch (not a metadata propagation issue)
    skip("igamma"),
    skip("igammac"),
    skip("addcdiv"),
    skip("addcmul"),
}


class TestTensorMetaProp(torch._dynamo.test_case.TestCase):
    """
    Test that inplace operations correctly propagate tensor metadata during Dynamo tracing.
    """

    @ops([op for op in op_db if op.get_inplace() is not None])
    @skipOps(test_inplace_ops_propagate_requires_grad_metadata_skips)
    def test_inplace_ops_propagate_requires_grad_metadata(self, device, dtype, op):
        """
        Test that inplace ops from OpInfo propagate requires_grad correctly.

        This test ensures that when an inplace operation is performed on a tensor
        without requires_grad using an argument with requires_grad=True, the metadata
        is correctly propagated in both eager and compiled modes.

        This is critical because if metadata is traced incorrectly, code that branches
        on requires_grad (like custom autograd functions) will take the wrong path,
        leading to silent incorrectness.
        """

        inplace_op = op.get_inplace()
        if inplace_op is None:
            self.skipTest("No inplace variant for this op")

        class CustomAutograd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                # Return an obviously wrong gradient (fixed value) to detect
                # when composite implicit autograd is used vs custom backward
                (x,) = ctx.saved_tensors
                return torch.full_like(x, 123.0)

        # Iterate directly over sample_inputs to preserve correct tracking
        # (converting to list first breaks the TrackedInputIter tracking)
        for sample in op.sample_inputs(device, dtype, requires_grad=False):
            # Skip samples that are broadcasted or have 0 elements
            if sample.broadcasts_input or sample.input.numel() == 0:
                continue

            # Skip scatter with reduce modes - backward not implemented for these
            if op.name == "scatter" and "reduce" in sample.kwargs:
                continue

            # Reset between samples to avoid exceeding recompile limit
            torch._dynamo.reset()

            # Setup: x starts with requires_grad=False, one arg has requires_grad=True
            x_eager = sample.input.clone().detach()
            args_eager = [
                arg.clone().detach() if isinstance(arg, torch.Tensor) else arg
                for arg in sample.args
            ]

            # Find a floating point tensor arg to set requires_grad=True
            requires_grad_idx = None
            for idx, arg in enumerate(args_eager):
                if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point:
                    arg.requires_grad_(True)
                    requires_grad_idx = idx
                    break

            if requires_grad_idx is None or x_eager.requires_grad:
                continue

            # Apply inplace op in eager mode
            inplace_op(x_eager, *args_eager, **sample.kwargs)
            output_eager = CustomAutograd.apply(x_eager)
            output_eager.sum().backward()

            # Setup compiled version
            x_compiled = sample.input.clone().detach()
            args_compiled = [
                arg.clone().detach() if isinstance(arg, torch.Tensor) else arg
                for arg in sample.args
            ]
            args_compiled[requires_grad_idx].requires_grad_(True)

            # Test 1: Verify that the metadata is propagated after the inplace op in compile time
            def compile_time_check(ctx: ComptimeContext) -> None:
                x = ctx.get_local("x")
                x_fake = x.as_fake()
                # Check requires_grad is propagated
                self.assertTrue(x_fake.requires_grad)
                self.assertTrue(x._ComptimeVar__variable.requires_grad)
                # Check that has_grad_fn is set (not for FakeTensor)
                self.assertTrue(x._ComptimeVar__variable.has_grad_fn)
                # Check dtype is preserved
                self.assertEqual(x_fake.dtype, dtype)
                self.assertEqual(x._ComptimeVar__variable.dtype, dtype)

            def fn(x, *args):
                inplace_op(x, *args, **sample.kwargs)
                comptime(compile_time_check)
                r = CustomAutograd.apply(x)
                return r

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            output_compiled = compiled_fn(x_compiled, *args_compiled)
            output_compiled.sum().backward()

            # Test 2: Verify requires_grad was propagated in runtime
            self.assertEqual(
                x_eager.requires_grad,
                x_compiled.requires_grad,
                msg=f"{op.name}: requires_grad mismatch (eager={x_eager.requires_grad}, compiled={x_compiled.requires_grad})",
            )

            # Test 3: Verify gradients match (with tolerance for float16/bfloat16)
            self.assertEqual(
                args_eager[requires_grad_idx].grad,
                args_compiled[requires_grad_idx].grad,
                msg=f"{op.name}: Gradient mismatch indicates metadata not propagated during tracing",
            )


instantiate_device_type_tests(TestTensorMetaProp, globals())


test_compile_matches_eager_skips: set = set()
test_compile_unary_ufunc_skips: set = set()
test_compile_binary_ufunc_skips: set = set()
test_compile_reduction_skips: set = set()
test_compile_foreach_skips: set = set()
test_compile_linalg_skips: set = set()
test_compile_nn_functional_skips: set = set()
test_compile_grad_skips: set = set()


def _aten_only(ops_list):
    return [op for op in ops_list if not op.name.startswith("_refs")]


_unary_ufunc_ops = _aten_only(unary_ufuncs)
_binary_ufunc_ops = _aten_only(binary_ufuncs)
_reduction_ops = _aten_only(reduction_ops)

_foreach_ops = (
    foreach_unary_op_db
    + foreach_binary_op_db
    + foreach_pointwise_op_db
    + foreach_reduce_op_db
    + foreach_other_op_db
)

_linalg_ops = [op for op in op_db if op.name.startswith("linalg")]
_nn_functional_ops = [op for op in op_db if op.name.startswith("nn.functional")]
_grad_ops = [op for op in op_db if op.supports_autograd]


_FORWARD_DTYPES = (
    torch.float32,
    torch.float64,
    torch.int64,
    torch.bfloat16,
    torch.complex64,
)

_UFUNC_LARGE_DTYPES = (
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
_UFUNC_EXTREMAL_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

_GRAD_DTYPES = (torch.float32, torch.float64, torch.bfloat16)
_grad_ops_dtypes = partial(
    ops, dtypes=OpDTypes.supported_backward, allowed_dtypes=_GRAD_DTYPES
)


def _tol_compare_cpu(op, dtype):
    if dtype is torch.bfloat16:
        return {"atol": 1e-2, "rtol": 1.6e-2}
    if dtype.is_floating_point or dtype.is_complex:
        return {"atol": 1e-3, "rtol": 1e-3}
    return {"atol": 0, "rtol": 0}


def _tol_unary_ufunc(op, dtype):
    if dtype in (torch.uint8, torch.int8, torch.bool):
        return {"atol": 1e-2, "rtol": 1e-3, "equal_nan": True}
    if dtype is torch.bfloat16:
        return {"atol": 1e-5, "rtol": 16e-3, "equal_nan": True}
    if dtype is torch.half:
        return {"atol": 1e-3, "rtol": 1.2e-3, "equal_nan": True}
    return {"equal_nan": True}


def _tol_binary_ufunc(op, dtype):
    if dtype is torch.bfloat16:
        return {"atol": 1e-5, "rtol": 16e-3, "equal_nan": True}
    return {"equal_nan": True}


def _tol_reductions(op, dtype):
    if dtype.is_floating_point:
        return {"atol": 1e-5, "rtol": 1e-3, "equal_nan": True}
    return {"equal_nan": True}


def _tol_default(op, dtype):
    return {}


def _tol_grad(op, dtype):
    if dtype is torch.bfloat16:
        return {"atol": 1e-2, "rtol": 1.6e-2, "equal_nan": True}
    if dtype.is_floating_point or dtype.is_complex:
        return {"atol": 1e-3, "rtol": 1e-3, "equal_nan": True}
    return {"equal_nan": True}


def _opinfo_tolerance_override(op, dtype):
    for di in getattr(op, "decorators", ()):
        if getattr(di, "dtypes", None) is not None and dtype not in di.dtypes:
            continue
        for dec in getattr(di, "decorators", []):
            if isinstance(dec, toleranceOverride) and dtype in dec.d:
                t = dec.d[dtype]
                return {"atol": t.atol, "rtol": t.rtol}
    return None


def _tol_opinfo(op, dtype):
    ov = _opinfo_tolerance_override(op, dtype)
    return ov if ov is not None else {}


class _CompileConsistencyMixin:
    compile_backend = os.environ.get("PYTORCH_TEST_COMPILE_BACKEND", "aot_eager")

    @staticmethod
    def _to_cpu(arg):
        return arg.to(device="cpu") if isinstance(arg, torch.Tensor) else arg

    @classmethod
    def _result_to_cpu(cls, out):
        if isinstance(out, torch.Tensor):
            return out.detach().to("cpu")
        if isinstance(out, (list, tuple)):
            return type(out)(cls._result_to_cpu(o) for o in out)
        if isinstance(out, dict):
            return {k: cls._result_to_cpu(v) for k, v in out.items()}
        return out

    @classmethod
    def _iter_tensors(cls, obj):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                yield from cls._iter_tensors(o)
        elif isinstance(obj, dict):
            for o in obj.values():
                yield from cls._iter_tensors(o)

    @classmethod
    def _sample_has_zero_sized(cls, sample):
        for obj in (sample.input, sample.args, sample.kwargs):
            for t in cls._iter_tensors(obj):
                if t.numel() == 0:
                    return True
        return False

    @classmethod
    def _effective_dtype(cls, expected, dtype):
        if not (dtype.is_floating_point or dtype.is_complex):
            first = next(cls._iter_tensors(expected), None)
            if first is not None and (
                first.is_floating_point() or first.is_complex()
            ):
                return first.dtype
        return dtype

    def _compiled(self, func):
        if self.compile_backend is not None:
            return torch.compile(func, backend=self.compile_backend)
        return torch.compile(func)

    def _compare_forward(self, op, device, dtype, samples, tol_fn):
        func = op.get_op()
        compiled = self._compiled(func)

        tested_any_sample = False
        skipped_zero_sized = False
        skip_zero_sized = not str(device).startswith("cpu")
        for sample in samples:
            torch._dynamo.reset()

            if skip_zero_sized and self._sample_has_zero_sized(sample):
                skipped_zero_sized = True
                continue
            tested_any_sample = True

            cpu_sample = sample.transform(self._to_cpu)
            expected = cpu_sample.output_process_fn_grad(
                func(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)
            )
            actual = sample.output_process_fn_grad(
                compiled(sample.input, *sample.args, **sample.kwargs)
            )
            actual = self._result_to_cpu(actual)

            tol = tol_fn(op, self._effective_dtype(expected, dtype))
            self.assertEqual(
                actual,
                expected,
                msg=f"{op.name}: compiled({self.compile_backend}) on {device} "
                f"differs from CPU reference",
                exact_dtype=False,
                **tol,
            )

        if not tested_any_sample:
            if skipped_zero_sized:
                self.skipTest(
                    f"{op.name}: all samples contain zero-sized tensors "
                    f"(unsupported on {device})"
                )
            self.skipTest(f"{op.name}: no inputs for {dtype} on {device}")

    @staticmethod
    def _to_cpu_leaf(arg):
        if isinstance(arg, torch.Tensor):
            leaf = arg.detach().to("cpu").clone()
            leaf.requires_grad_(arg.requires_grad)
            return leaf
        return arg

    @classmethod
    def _grad_loss(cls, out):
        total = None
        for t in cls._iter_tensors(out):
            if not (t.is_floating_point() or t.is_complex()):
                continue
            s = t.sum()
            if s.is_complex():
                s = s.real + s.imag
            total = s if total is None else total + s
        return total

    def _compare_grad(self, op, device, dtype, tol_fn):
        func = op.get_op()
        compiled = self._compiled(func)
        tol = tol_fn(op, dtype)

        tested_any_sample = False
        skipped_zero_sized = False
        skip_zero_sized = not str(device).startswith("cpu")
        for sample in op.sample_inputs(device, dtype, requires_grad=True):
            torch._dynamo.reset()

            if skip_zero_sized and self._sample_has_zero_sized(sample):
                skipped_zero_sized = True
                continue

            cpu_sample = sample.transform(self._to_cpu_leaf)
            dev_leaves = [
                t
                for t in self._iter_tensors(
                    (sample.input, sample.args, sample.kwargs)
                )
                if t.requires_grad
            ]
            cpu_leaves = [
                t
                for t in self._iter_tensors(
                    (cpu_sample.input, cpu_sample.args, cpu_sample.kwargs)
                )
                if t.requires_grad
            ]
            if not dev_leaves:
                continue

            out = sample.output_process_fn_grad(
                compiled(sample.input, *sample.args, **sample.kwargs)
            )
            out_cpu = cpu_sample.output_process_fn_grad(
                func(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)
            )
            loss = self._grad_loss(out)
            loss_cpu = self._grad_loss(out_cpu)
            if loss is None or not loss.requires_grad or not loss_cpu.requires_grad:
                continue
            tested_any_sample = True

            dev_grads = torch.autograd.grad(loss, dev_leaves, allow_unused=True)
            cpu_grads = torch.autograd.grad(
                loss_cpu, cpu_leaves, allow_unused=True
            )
            dev_grads = [
                self._result_to_cpu(g) if g is not None else None for g in dev_grads
            ]

            for i, (dg, cg) in enumerate(zip(dev_grads, cpu_grads)):
                if dg is None and cg is None:
                    continue
                self.assertEqual(
                    dg,
                    cg,
                    msg=f"{op.name}: grad[{i}] compiled({self.compile_backend}) "
                    f"on {device} differs from CPU reference",
                    exact_dtype=False,
                    **tol,
                )

        if not tested_any_sample:
            if skipped_zero_sized:
                self.skipTest(
                    f"{op.name}: all samples contain zero-sized tensors "
                    f"(unsupported on {device})"
                )
            self.skipTest(
                f"{op.name}: no differentiable inputs for {dtype} on {device}"
            )


class TestCompileOpInfo(_CompileConsistencyMixin, torch._dynamo.test_case.TestCase):
    @ops(op_db, allowed_dtypes=_FORWARD_DTYPES)
    @skipOps(test_compile_matches_eager_skips)
    def test_compile_matches_eager(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        self._compare_forward(op, device, dtype, samples, _tol_compare_cpu)


class TestCompileUnaryUfuncInfo(
    _CompileConsistencyMixin, torch._dynamo.test_case.TestCase
):
    @ops(_unary_ufunc_ops)
    @skipOps(test_compile_unary_ufunc_skips)
    def test_reference_numerics_normal(self, device, dtype, op):
        samples = generate_elementwise_unary_tensors(
            op, device=device, dtype=dtype, requires_grad=False
        )
        self._compare_forward(op, device, dtype, samples, _tol_unary_ufunc)

    @ops(_unary_ufunc_ops)
    @skipOps(test_compile_unary_ufunc_skips)
    def test_reference_numerics_small(self, device, dtype, op):
        if dtype is torch.bool:
            self.skipTest("bool has no small-value regime")
        samples = generate_elementwise_unary_small_value_tensors(
            op, device=device, dtype=dtype
        )
        self._compare_forward(op, device, dtype, samples, _tol_unary_ufunc)

    @ops(_unary_ufunc_ops, allowed_dtypes=_UFUNC_LARGE_DTYPES)
    @skipOps(test_compile_unary_ufunc_skips)
    def test_reference_numerics_large(self, device, dtype, op):
        samples = generate_elementwise_unary_large_value_tensors(
            op, device=device, dtype=dtype
        )
        self._compare_forward(op, device, dtype, samples, _tol_unary_ufunc)

    @ops(_unary_ufunc_ops, allowed_dtypes=_UFUNC_EXTREMAL_DTYPES)
    @skipOps(test_compile_unary_ufunc_skips)
    def test_reference_numerics_extremal(self, device, dtype, op):
        samples = generate_elementwise_unary_extremal_value_tensors(
            op, device=device, dtype=dtype
        )
        self._compare_forward(op, device, dtype, samples, _tol_unary_ufunc)


class TestCompileBinaryUfuncInfo(
    _CompileConsistencyMixin, torch._dynamo.test_case.TestCase
):
    @ops(_binary_ufunc_ops)
    @skipOps(test_compile_binary_ufunc_skips)
    def test_reference_numerics_normal(self, device, dtype, op):
        samples = generate_elementwise_binary_tensors(
            op, device=device, dtype=dtype, requires_grad=False
        )
        self._compare_forward(op, device, dtype, samples, _tol_binary_ufunc)

    @ops(_binary_ufunc_ops)
    @skipOps(test_compile_binary_ufunc_skips)
    def test_reference_numerics_small(self, device, dtype, op):
        if dtype is torch.bool:
            self.skipTest("bool has no small-value regime")
        samples = generate_elementwise_binary_small_value_tensors(
            op, device=device, dtype=dtype
        )
        self._compare_forward(op, device, dtype, samples, _tol_binary_ufunc)

    @ops(_binary_ufunc_ops, allowed_dtypes=_UFUNC_LARGE_DTYPES)
    @skipOps(test_compile_binary_ufunc_skips)
    def test_reference_numerics_large(self, device, dtype, op):
        samples = generate_elementwise_binary_large_value_tensors(
            op, device=device, dtype=dtype
        )
        self._compare_forward(op, device, dtype, samples, _tol_binary_ufunc)

    @ops(_binary_ufunc_ops, allowed_dtypes=_UFUNC_EXTREMAL_DTYPES)
    @skipOps(test_compile_binary_ufunc_skips)
    def test_reference_numerics_extremal(self, device, dtype, op):
        samples = generate_elementwise_binary_extremal_value_tensors(
            op, device=device, dtype=dtype
        )
        self._compare_forward(op, device, dtype, samples, _tol_binary_ufunc)


class TestCompileReductionOpInfo(
    _CompileConsistencyMixin, torch._dynamo.test_case.TestCase
):
    @ops(_reduction_ops, allowed_dtypes=_FORWARD_DTYPES)
    @skipOps(test_compile_reduction_skips)
    def test_compile_reduction(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        self._compare_forward(op, device, dtype, samples, _tol_reductions)


class TestCompileForeachOpInfo(
    _CompileConsistencyMixin, torch._dynamo.test_case.TestCase
):
    @ops(_foreach_ops, allowed_dtypes=_FORWARD_DTYPES)
    @skipOps(test_compile_foreach_skips)
    def test_compile_foreach(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        self._compare_forward(op, device, dtype, samples, _tol_default)


class TestCompileLinalgOpInfo(
    _CompileConsistencyMixin, torch._dynamo.test_case.TestCase
):
    @ops(_linalg_ops, allowed_dtypes=_FORWARD_DTYPES)
    @skipOps(test_compile_linalg_skips)
    def test_compile_linalg(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        self._compare_forward(op, device, dtype, samples, _tol_opinfo)


class TestCompileNNFunctionalOpInfo(
    _CompileConsistencyMixin, torch._dynamo.test_case.TestCase
):
    @ops(_nn_functional_ops, allowed_dtypes=_FORWARD_DTYPES)
    @skipOps(test_compile_nn_functional_skips)
    def test_compile_nn_functional(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        self._compare_forward(op, device, dtype, samples, _tol_opinfo)


class TestCompileOpInfoGrad(_CompileConsistencyMixin, TestGradients):
    @_grad_ops_dtypes(_grad_ops)
    @skipOps(test_compile_grad_skips)
    def test_compile_fn_grad(self, device, dtype, op):
        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest("Skipped! Dtype is not in supported backward dtypes!")

        with _functorch_config.patch(donated_buffer=False):
            self._compare_grad(op, device, dtype, _tol_grad)


instantiate_device_type_tests(TestCompileOpInfo, globals())
instantiate_device_type_tests(TestCompileUnaryUfuncInfo, globals())
instantiate_device_type_tests(TestCompileBinaryUfuncInfo, globals())
instantiate_device_type_tests(TestCompileReductionOpInfo, globals())
instantiate_device_type_tests(TestCompileForeachOpInfo, globals())
instantiate_device_type_tests(TestCompileLinalgOpInfo, globals())
instantiate_device_type_tests(TestCompileNNFunctionalOpInfo, globals())
instantiate_device_type_tests(TestCompileOpInfoGrad, globals())


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
