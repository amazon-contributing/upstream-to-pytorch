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


test_compile_matches_eager_skips = {
    skip("__getitem__"),
    skip("__rmod__"),
    skip("__rpow__"),
    skip("_native_batch_norm_legit"),
    skip("_segment_reduce.lengths"),
    skip("_segment_reduce.offsets"),
    skip("_unsafe_masked_index_put_accumulate"),
    skip("addcdiv"),
    skip("addcmul"),
    skip("addr"),
    skip("alias_copy"),
    skip("all"),
    skip("allclose"),
    skip("amax"),
    skip("amin"),
    skip("any"),
    skip("arange"),  # hang: bfloat16 input hangs torch.compile (never completes)
    skip("argsort"),
    skip("argwhere"),
    skip("as_strided.partial_views"),
    skip("as_strided_copy"),
    skip("as_strided_scatter"),
    skip("bernoulli"),
    skip("block_diag"),
    skip("byte"),
    skip("cat"),
    skip("cauchy"),
    skip("cdist"),
    skip("ceil"),
    skip("char"),
    skip("cholesky"),
    skip("cholesky_inverse"),
    skip("cholesky_solve"),
    skip("clamp_max"),
    skip("clamp_min"),
    skip("clone"),
    skip("constant_pad_nd"),
    skip("cummax"),
    skip("cummin"),
    skip("cumsum"),
    skip("diag"),
    skip("diagflat"),
    skip("diff"),
    skip("dist"),
    skip("div.floor_rounding"),
    skip("div.trunc_rounding"),
    skip("double"),
    skip("empty"),
    skip("empty_like"),
    skip("empty_permuted"),
    skip("empty_strided"),
    skip("erf"),
    skip("expand_copy"),
    skip("exponential"),
    skip("eye"),
    skip("flatten"),
    skip("flip"),
    skip("floor"),
    skip("floor_divide"),
    skip("gather"),
    skip("ge"),
    skip("geometric"),
    skip("gradient"),
    skip("grid_sampler_2d"),
    skip("grid_sampler_3d"),
    skip("histc"),
    skip("index_reduce.amin"),
    skip("index_reduce.mean"),
    skip("index_select"),
    skip("int"),
    skip("isfinite"),
    skip("item"),
    skip("kthvalue"),
    skip("linalg.cholesky"),
    skip("linalg.cholesky_ex"),
    skip("linalg.cond"),
    skip("linalg.det"),
    skip("linalg.eig"),
    skip("linalg.eigh"),
    skip("linalg.eigvals"),
    skip("linalg.eigvalsh"),
    skip("linalg.householder_product"),
    skip("linalg.inv"),
    skip("linalg.inv_ex"),
    skip("linalg.ldl_factor"),
    skip("linalg.ldl_factor_ex"),
    skip("linalg.ldl_solve"),
    skip("linalg.lu"),
    skip("linalg.lu_factor"),
    skip("linalg.lu_factor_ex"),
    skip("linalg.lu_solve"),
    skip("linalg.matrix_norm"),
    skip("linalg.matrix_power"),
    skip("linalg.matrix_rank"),
    skip("linalg.matrix_rank.hermitian"),
    skip("linalg.norm"),
    skip("linalg.norm.subgradients_at_zero"),
    skip("linalg.pinv"),
    skip("linalg.pinv.hermitian"),
    skip("linalg.qr"),
    skip("linalg.slogdet"),
    skip("linalg.solve"),
    skip("linalg.solve_ex"),
    skip("linalg.solve_triangular"),
    skip("linalg.tensorinv"),
    skip("linalg.tensorsolve"),
    skip("linalg.vander"),
    skip("linalg.vecdot"),
    skip("linspace"),
    skip("linspace.tensor_overload"),
    skip("log_normal"),
    skip("logcumsumexp"),
    skip("logdet"),
    skip("logspace"),
    skip("logspace.tensor_overload"),
    skip("long"),
    skip("lt"),
    skip("lu"),
    skip("lu_solve"),
    skip("lu_unpack"),
    skip("masked.amax"),
    skip("masked.amin"),
    skip("masked.cumsum"),
    skip("masked.log_softmax"),
    skip("masked.logsumexp"),
    skip("masked.mean"),
    skip("masked.median"),
    skip("masked.norm"),
    skip("masked.prod"),
    skip("masked.softmax"),
    skip("masked.std"),
    skip("masked.sum"),
    skip("masked_scatter"),
    skip("matmul"),
    skip("matrix_exp"),
    skip("max.reduction_no_dim"),
    skip("max.reduction_with_dim"),
    skip("mean"),
    skip("median"),
    skip("min.reduction_no_dim"),
    skip("min.reduction_with_dim"),
    skip("mode"),
    skip("msort"),
    skip("multinomial"),
    skip("nanmedian"),
    skip("nanquantile"),
    skip("narrow_copy"),
    skip("native_group_norm"),
    skip("neg"),
    skip("new_empty"),
    skip("new_empty_strided"),
    skip("nn.functional.adaptive_max_pool1d"),
    skip("nn.functional.alpha_dropout"),
    skip("nn.functional.batch_norm"),
    skip("nn.functional.bilinear"),
    skip("nn.functional.binary_cross_entropy_with_logits"),
    skip("nn.functional.channel_shuffle"),
    skip("nn.functional.conv2d"),
    skip("nn.functional.conv_transpose1d"),
    skip("nn.functional.conv_transpose2d"),
    skip("nn.functional.conv_transpose3d"),
    skip("nn.functional.cosine_similarity"),
    skip("nn.functional.cross_entropy"),
    skip("nn.functional.dropout"),
    skip("nn.functional.dropout2d"),
    skip("nn.functional.dropout3d"),
    skip("nn.functional.embedding"),
    skip("nn.functional.embedding_bag"),
    skip("nn.functional.feature_alpha_dropout.with_train"),
    skip("nn.functional.feature_alpha_dropout.without_train"),
    skip("nn.functional.grid_sample"),
    skip("nn.functional.group_norm"),
    skip("nn.functional.instance_norm"),
    skip("nn.functional.interpolate.area"),
    skip("nn.functional.interpolate.bicubic"),
    skip("nn.functional.interpolate.bilinear"),
    skip("nn.functional.interpolate.linear"),
    skip("nn.functional.interpolate.trilinear"),
    skip("nn.functional.local_response_norm"),
    skip("nn.functional.max_pool1d"),
    skip("nn.functional.max_pool2d"),
    skip("nn.functional.max_pool3d"),
    skip("nn.functional.multi_head_attention_forward"),
    skip("nn.functional.one_hot"),
    skip("nn.functional.pad.constant"),
    skip("nn.functional.pdist"),
    skip("nn.functional.pixel_shuffle"),
    skip("nn.functional.pixel_unshuffle"),
    skip("nn.functional.relu"),
    skip("nn.functional.scaled_dot_product_attention"),
    skip("nn.functional.upsample_bilinear"),
    skip("nonzero"),
    skip("nonzero_static"),
    skip("norm.nuc"),
    skip("normal"),
    skip("normal.in_place"),
    skip("normal.number_mean"),
    skip("pca_lowrank"),
    skip("permute_copy"),
    skip("pinverse"),
    skip("polar"),
    skip("polygamma.polygamma_n_0"),
    skip("polygamma.polygamma_n_1"),
    skip("polygamma.polygamma_n_2"),
    skip("polygamma.polygamma_n_3"),
    skip("polygamma.polygamma_n_4"),
    skip("pow"),
    skip("prod"),
    skip("qr"),
    skip("quantile"),
    skip("rand_like"),
    skip("randint"),
    skip("randint_like"),
    skip("randn"),
    skip("randn_like"),
    skip("remainder"),
    skip("repeat"),
    skip("resize_"),
    skip("resize_as_"),
    skip("roll"),
    skip("rot90"),
    skip("scatter"),
    skip("scatter_reduce.amax"),
    skip("scatter_reduce.amin"),
    skip("scatter_reduce.mean"),
    skip("scatter_reduce.prod"),
    skip("short"),
    skip("signal.windows.blackman"),
    skip("signal.windows.general_cosine"),
    skip("signal.windows.general_hamming"),
    skip("signal.windows.hamming"),
    skip("signal.windows.hann"),
    skip("signal.windows.kaiser"),
    skip("signal.windows.nuttall"),
    skip("signbit"),
    skip("slice_scatter"),
    skip("sort"),
    skip("sparse.sampled_addmm"),
    skip("special.airy_ai"),
    skip("special.bessel_y0"),
    skip("special.bessel_y1"),
    skip("special.chebyshev_polynomial_t"),
    skip("special.chebyshev_polynomial_u"),
    skip("special.chebyshev_polynomial_v"),
    skip("special.chebyshev_polynomial_w"),
    skip("special.hermite_polynomial_h"),
    skip("special.hermite_polynomial_he"),
    skip("special.legendre_polynomial_p"),
    skip("special.modified_bessel_i1"),
    skip("special.modified_bessel_k0"),
    skip("special.modified_bessel_k1"),
    skip("special.polygamma.special_polygamma_n_0"),
    skip("special.scaled_modified_bessel_k0"),
    skip("special.scaled_modified_bessel_k1"),
    skip("special.shifted_chebyshev_polynomial_t"),
    skip("special.shifted_chebyshev_polynomial_u"),
    skip("special.shifted_chebyshev_polynomial_v"),
    skip("special.shifted_chebyshev_polynomial_w"),
    skip("squeeze"),
    skip("squeeze_copy"),
    skip("stft"),
    skip("sum"),
    skip("sum_to_size"),
    skip("svd"),
    skip("svd_lowrank"),
    skip("t_copy"),
    skip("take_along_dim"),
    skip("tan"),
    skip("tile"),
    skip("to_sparse"),
    skip("topk"),
    skip("transpose"),
    skip("transpose_copy"),
    skip("triangular_solve"),
    skip("triu_indices"),
    skip("trunc"),
    skip("unbind_copy"),
    skip("uniform"),
    skip("unravel_index"),
    skip("unsafe_chunk"),
    skip("unsafe_split"),
    skip("unsqueeze_copy"),
    skip("view_copy"),
}


test_compile_unary_ufunc_skips = {
    skip("abs"),
    skip("acosh"),
    skip("asinh"),
    skip("byte"),
    skip("ceil"),
    skip("char"),
    skip("cos"),
    skip("cosh"),
    skip("erf"),
    skip("exp"),
    skip("floor"),
    skip("frexp"),
    skip("int"),
    skip("isfinite"),
    skip("isinf"),
    skip("log"),
    skip("long"),
    skip("mvlgamma.mvlgamma_p_1"),
    skip("mvlgamma.mvlgamma_p_3"),
    skip("mvlgamma.mvlgamma_p_5"),
    skip("neg"),
    skip("nn.functional.relu"),
    skip("nn.functional.silu"),
    skip("nn.functional.softplus"),
    skip("nn.functional.softsign"),
    skip("nn.functional.threshold"),
    skip("polygamma.polygamma_n_0"),
    skip("polygamma.polygamma_n_1"),
    skip("polygamma.polygamma_n_2"),
    skip("polygamma.polygamma_n_3"),
    skip("polygamma.polygamma_n_4"),
    skip("reciprocal"),
    skip("round.decimals_0"),
    skip("round.decimals_3"),
    skip("round.decimals_neg_3"),
    skip("rsqrt"),
    skip("sgn"),
    skip("short"),
    skip("sign"),
    skip("signbit"),
    skip("sin"),
    skip("sinh"),
    skip("special.airy_ai"),
    skip("special.bessel_y0"),
    skip("special.bessel_y1"),
    skip("special.modified_bessel_i1"),
    skip("special.modified_bessel_k0"),
    skip("special.modified_bessel_k1"),
    skip("special.ndtr"),
    skip("special.polygamma.special_polygamma_n_0"),
    skip("special.scaled_modified_bessel_k0"),
    skip("special.scaled_modified_bessel_k1"),
    skip("sqrt"),
    skip("square"),
    skip("tan"),
    skip("trunc"),
}


test_compile_binary_ufunc_skips = {
    skip("__radd__"),
    skip("__rdiv__"),
    skip("__rmod__"),
    skip("__rmul__"),
    skip("__rpow__"),
    skip("__rsub__"),
    skip("add"),
    skip("bitwise_left_shift"),
    skip("bitwise_xor"),
    skip("clamp_max"),
    skip("clamp_min"),
    skip("div.floor_rounding"),
    skip("div.trunc_rounding"),
    skip("eq"),
    skip("float_power"),
    skip("floor_divide"),
    skip("ge"),
    skip("isclose"),
    skip("jiterator_binary"),
    skip("jiterator_binary_return_by_ref"),
    skip("logical_and"),
    skip("lt"),
    skip("max.binary"),
    skip("maximum"),
    skip("min.binary"),
    skip("minimum"),
    skip("mul"),
    skip("ne"),
    skip("polar"),
    skip("pow"),
    skip("remainder"),
    skip("rsub"),
    skip("special.chebyshev_polynomial_t"),
    skip("special.chebyshev_polynomial_u"),
    skip("special.chebyshev_polynomial_v"),
    skip("special.chebyshev_polynomial_w"),
    skip("special.hermite_polynomial_h"),
    skip("special.hermite_polynomial_he"),
    skip("special.legendre_polynomial_p"),
    skip("special.shifted_chebyshev_polynomial_t"),
    skip("special.shifted_chebyshev_polynomial_u"),
    skip("special.shifted_chebyshev_polynomial_v"),
    skip("special.shifted_chebyshev_polynomial_w"),
    skip("sub"),
}


test_compile_reduction_skips = {
    skip("all"),
    skip("amax"),
    skip("amin"),
    skip("any"),
    skip("linalg.vector_norm"),
    skip("masked.amax"),
    skip("masked.amin"),
    skip("masked.logsumexp"),
    skip("masked.mean"),
    skip("masked.norm"),
    skip("masked.prod"),
    skip("masked.std"),
    skip("masked.sum"),
    skip("masked.var"),
    skip("mean"),
    skip("nanmean"),
    skip("prod"),
    skip("std"),
    skip("sum"),
    skip("var"),
}


test_compile_foreach_skips = {
    skip("_foreach_add"),
    skip("_foreach_addcdiv"),
    skip("_foreach_addcmul"),
    skip("_foreach_clone"),
    skip("_foreach_copy"),
    skip("_foreach_cos"),
    skip("_foreach_div"),
    skip("_foreach_erf"),
    skip("_foreach_exp"),
    skip("_foreach_lerp"),
    skip("_foreach_mm"),
    skip("_foreach_norm"),
    skip("_foreach_pow"),
    skip("_foreach_sin"),
    skip("_foreach_sqrt"),
    skip("_foreach_sub"),
    skip("_foreach_zero"),
}


test_compile_linalg_skips = {
    skip("linalg.cholesky"),
    skip("linalg.cholesky_ex"),
    skip("linalg.cond"),
    skip("linalg.det"),
    skip("linalg.eig"),
    skip("linalg.eigh"),
    skip("linalg.eigvals"),
    skip("linalg.eigvalsh"),
    skip("linalg.householder_product"),
    skip("linalg.inv"),
    skip("linalg.inv_ex"),
    skip("linalg.ldl_factor"),
    skip("linalg.ldl_factor_ex"),
    skip("linalg.ldl_solve"),
    skip("linalg.lu"),
    skip("linalg.lu_factor"),
    skip("linalg.lu_factor_ex"),
    skip("linalg.lu_solve"),
    skip("linalg.matrix_norm"),
    skip("linalg.matrix_power"),
    skip("linalg.matrix_rank"),
    skip("linalg.matrix_rank.hermitian"),
    skip("linalg.multi_dot"),
    skip("linalg.norm"),
    skip("linalg.norm.subgradients_at_zero"),
    skip("linalg.pinv"),
    skip("linalg.pinv.hermitian"),
    skip("linalg.qr"),
    skip("linalg.slogdet"),
    skip("linalg.solve"),
    skip("linalg.solve_ex"),
    skip("linalg.solve_triangular"),
    skip("linalg.tensorinv"),
    skip("linalg.tensorsolve"),
    skip("linalg.vander"),
    skip("linalg.vecdot"),
    skip("linalg.vector_norm"),
}


test_compile_nn_functional_skips = {
    skip("nn.functional.adaptive_max_pool1d"),
    skip("nn.functional.alpha_dropout"),
    skip("nn.functional.batch_norm"),
    skip("nn.functional.bilinear"),
    skip("nn.functional.binary_cross_entropy_with_logits"),
    skip("nn.functional.channel_shuffle"),
    skip("nn.functional.conv2d"),
    skip("nn.functional.conv_transpose1d"),
    skip("nn.functional.conv_transpose2d"),
    skip("nn.functional.conv_transpose3d"),
    skip("nn.functional.cosine_similarity"),
    skip("nn.functional.cross_entropy"),
    skip("nn.functional.dropout"),
    skip("nn.functional.dropout2d"),
    skip("nn.functional.dropout3d"),
    skip("nn.functional.embedding"),
    skip("nn.functional.embedding_bag"),
    skip("nn.functional.feature_alpha_dropout.with_train"),
    skip("nn.functional.feature_alpha_dropout.without_train"),
    skip("nn.functional.gelu"),
    skip("nn.functional.grid_sample"),
    skip("nn.functional.group_norm"),
    skip("nn.functional.instance_norm"),
    skip("nn.functional.interpolate.area"),
    skip("nn.functional.interpolate.bicubic"),
    skip("nn.functional.interpolate.bilinear"),
    skip("nn.functional.interpolate.linear"),
    skip("nn.functional.interpolate.trilinear"),
    skip("nn.functional.max_pool1d"),
    skip("nn.functional.max_pool2d"),
    skip("nn.functional.max_pool3d"),
    skip("nn.functional.max_unpool2d.grad"),
    skip("nn.functional.multi_head_attention_forward"),
    skip("nn.functional.one_hot"),
    skip("nn.functional.pad.constant"),
    skip("nn.functional.pdist"),
    skip("nn.functional.pixel_shuffle"),
    skip("nn.functional.pixel_unshuffle"),
    skip("nn.functional.poisson_nll_loss"),
    skip("nn.functional.scaled_dot_product_attention"),
    skip("nn.functional.triplet_margin_loss"),
    skip("nn.functional.triplet_margin_with_distance_loss"),
    skip("nn.functional.upsample_bilinear"),
}


test_compile_grad_skips = {
    skip("__getitem__"),
    skip("__rmod__"),
    skip("_native_batch_norm_legit"),
    skip("_segment_reduce.lengths"),
    skip("_segment_reduce.offsets"),
    skip("addr"),
    skip("alias_copy"),
    skip("as_strided.partial_views"),
    skip("as_strided_copy"),
    skip("bernoulli"),
    skip("block_diag"),
    skip("cdist"),
    skip("cholesky"),
    skip("cholesky_inverse"),
    skip("cholesky_solve"),
    skip("clamp_max"),
    skip("clamp_min"),
    skip("cummax"),
    skip("cummin"),
    skip("cumprod"),
    skip("cumsum"),
    skip("diag"),
    skip("diagonal_copy"),
    skip("diff"),
    skip("dist"),
    skip("div.no_rounding_mode"),
    skip("dsplit"),
    skip("einsum"),
    skip("erfinv"),
    skip("expand_copy"),
    skip("gather"),
    skip("gradient"),
    skip("grid_sampler_2d"),
    skip("grid_sampler_3d"),
    skip("index_select"),
    skip("kthvalue"),
    skip("linalg.cholesky"),
    skip("linalg.cholesky_ex"),
    skip("linalg.cond"),
    skip("linalg.det"),
    skip("linalg.eig"),
    skip("linalg.eigh"),
    skip("linalg.eigvals"),
    skip("linalg.eigvalsh"),
    skip("linalg.householder_product"),
    skip("linalg.inv"),
    skip("linalg.inv_ex"),
    skip("linalg.lu"),
    skip("linalg.lu_factor"),
    skip("linalg.lu_factor_ex"),
    skip("linalg.lu_solve"),
    skip("linalg.matrix_norm"),
    skip("linalg.matrix_power"),
    skip("linalg.norm"),
    skip("linalg.norm.subgradients_at_zero"),
    skip("linalg.pinv"),
    skip("linalg.pinv.hermitian"),
    skip("linalg.qr"),
    skip("linalg.slogdet"),
    skip("linalg.solve"),
    skip("linalg.solve_ex"),
    skip("linalg.solve_triangular"),
    skip("linalg.tensorinv"),
    skip("linalg.tensorsolve"),
    skip("linalg.vander"),
    skip("linalg.vector_norm"),
    skip("logcumsumexp"),
    skip("logdet"),
    skip("lu"),
    skip("lu_solve"),
    skip("lu_unpack"),
    skip("masked.cumprod"),
    skip("masked.cumsum"),
    skip("masked.mean"),
    skip("masked.median"),
    skip("masked.normalize"),
    skip("masked.prod"),
    skip("matrix_exp"),
    skip("max.reduction_with_dim"),
    skip("mean"),
    skip("median"),
    skip("min.reduction_with_dim"),
    skip("msort"),
    skip("mul"),  # hang: float32 grad hangs torch.compile (never completes)
    skip("mvlgamma.mvlgamma_p_1"),
    skip("mvlgamma.mvlgamma_p_3"),
    skip("mvlgamma.mvlgamma_p_5"),
    skip("nanmedian"),
    skip("nanquantile"),
    skip("narrow_copy"),
    skip("native_group_norm"),
    skip("native_layer_norm"),
    skip("nn.functional.adaptive_max_pool1d"),
    skip("nn.functional.alpha_dropout"),
    skip("nn.functional.batch_norm"),
    skip("nn.functional.bilinear"),
    skip("nn.functional.binary_cross_entropy_with_logits"),
    skip("nn.functional.channel_shuffle"),
    skip("nn.functional.conv1d"),
    skip("nn.functional.conv2d"),
    skip("nn.functional.conv3d"),
    skip("nn.functional.conv_transpose1d"),
    skip("nn.functional.conv_transpose2d"),
    skip("nn.functional.conv_transpose3d"),
    skip("nn.functional.cosine_similarity"),
    skip("nn.functional.cross_entropy"),
    skip("nn.functional.dropout"),
    skip("nn.functional.dropout2d"),
    skip("nn.functional.dropout3d"),
    skip("nn.functional.embedding_bag"),
    skip("nn.functional.feature_alpha_dropout.with_train"),
    skip("nn.functional.feature_alpha_dropout.without_train"),
    skip("nn.functional.grid_sample"),
    skip("nn.functional.group_norm"),
    skip("nn.functional.instance_norm"),
    skip("nn.functional.interpolate.area"),
    skip("nn.functional.interpolate.bilinear"),
    skip("nn.functional.interpolate.linear"),
    skip("nn.functional.interpolate.trilinear"),
    skip("nn.functional.l1_loss"),
    skip("nn.functional.margin_ranking_loss"),
    skip("nn.functional.max_pool1d"),
    skip("nn.functional.max_pool2d"),
    skip("nn.functional.max_pool3d"),
    skip("nn.functional.multi_head_attention_forward"),
    skip("nn.functional.normalize"),
    skip("nn.functional.pad.reflect"),
    skip("nn.functional.pad.replicate"),
    skip("nn.functional.pad.replicate_negative"),
    skip("nn.functional.pdist"),
    skip("nn.functional.rms_norm"),
    skip("nn.functional.scaled_dot_product_attention"),
    skip("nn.functional.triplet_margin_loss"),
    skip("nn.functional.upsample_bilinear"),
    skip("norm"),
    skip("norm.nuc"),
    skip("normal"),
    skip("normal.number_mean"),
    skip("pca_lowrank"),
    skip("permute_copy"),
    skip("pinverse"),
    skip("polar"),
    skip("polygamma.polygamma_n_0"),
    skip("polygamma.polygamma_n_1"),
    skip("polygamma.polygamma_n_2"),
    skip("polygamma.polygamma_n_3"),
    skip("polygamma.polygamma_n_4"),
    skip("prod"),
    skip("qr"),
    skip("quantile"),
    skip("remainder"),
    skip("roll"),
    skip("round.decimals_neg_3"),
    skip("scatter_reduce.amax"),
    skip("scatter_reduce.amin"),
    skip("scatter_reduce.prod"),
    skip("sgn"),
    skip("sin"),
    skip("sort"),
    skip("sparse.sampled_addmm"),
    skip("special.polygamma.special_polygamma_n_0"),
    skip("squeeze_copy"),
    skip("stft"),
    skip("svd"),
    skip("svd_lowrank"),
    skip("t_copy"),
    skip("take_along_dim"),
    skip("tan"),
    skip("to_sparse"),
    skip("topk"),
    skip("transpose"),
    skip("transpose_copy"),
    skip("triangular_solve"),
    skip("unbind_copy"),
    skip("unfold_copy"),
    skip("unsafe_chunk"),
    skip("unsafe_split"),
    skip("unsqueeze_copy"),
    skip("var"),
    skip("var_mean"),
    skip("view_copy"),
}


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
