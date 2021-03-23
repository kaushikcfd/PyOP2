import loopy as lp
from pyop2.configuration import configuration


def transpose_maps(kernel):
    raise NotImplementedError()
    from loopy.kernel.array import FixedStrideArrayDimTag
    from pymbolic import parse

    new_dim_tags = (FixedStrideArrayDimTag(1), FixedStrideArrayDimTag(parse('end-start')))
    new_args = [arg.copy(dim_tags=new_dim_tags) if arg.name[:3] == 'map' else arg for arg in kernel.args]
    kernel = kernel.copy(args=new_args)
    return kernel


def get_loopy_target(target):
    if target == 'opencl':
        return lp.OpenCLTarget()
    elif target == 'cuda':
        return lp.CudaTarget()
    else:
        raise NotImplementedError()


def preprocess_gpu_kernel(program):
    kernel = program.root_kernel

    # changing the address space of temps
    def _change_aspace_tvs(tv):
        if tv.read_only:
            assert tv.initializer is not None
            return tv.copy(address_space=lp.AddressSpace.GLOBAL)
        else:
            return tv.copy(address_space=lp.AddressSpace.PRIVATE)

    new_tvs = {tv_name: _change_aspace_tvs(tv) for tv_name, tv in
               kernel.temporary_variables.items()}
    kernel = kernel.copy(temporary_variables=new_tvs)

    def insn_needs_atomic(insn):
        # updates to global variables are atomic
        import pymbolic
        if isinstance(insn, lp.Assignment):
            if isinstance(insn.assignee, pymbolic.primitives.Subscript):
                assignee_name = insn.assignee.aggregate.name
            else:
                assert isinstance(insn.assignee, pymbolic.primitives.Variable)
                assignee_name = insn.assignee.name

            if assignee_name in kernel.arg_dict:
                return assignee_name in insn.read_dependency_names()
        return False

    new_insns = []
    args_marked_for_atomic = set()
    for insn in kernel.instructions:
        if insn_needs_atomic(insn):
            atomicity = (lp.AtomicUpdate(insn.assignee.aggregate.name), )
            insn = insn.copy(atomicity=atomicity)
            args_marked_for_atomic |= set([insn.assignee.aggregate.name])

        new_insns.append(insn)

    # label args as atomic
    new_args = []
    for arg in kernel.args:
        if arg.name in args_marked_for_atomic:
            new_args.append(arg.copy(for_atomic=True))
        else:
            new_args.append(arg)

    kernel = kernel.copy(instructions=new_insns, args=new_args)

    return program.with_root_kernel(kernel)


def generate_gpu_kernel(program, args=None, argshapes=None, target=None):
    # Kernel transformations
    program = program.copy(target=get_loopy_target(target))
    program = preprocess_gpu_kernel(program)
    kernel = program.root_kernel
    # FIXME: These might not always be true
    # Might need to be removed before going full production
    kernel = lp.assume(kernel, "start=0")
    kernel = lp.assume(kernel, "end>0")

    # choose the preferred algorithm here
    # TODO: Not sure if this is the right way to select different
    # transformation strategies based on kernels
    if program.name in [
            "wrap_form0_cell_integral_otherwise",
            "wrap_form0_exterior_facet_integral_otherwise",
            "wrap_form0_interior_facet_integral_otherwise",
            "wrap_form1_cell_integral_otherwise"]:
        if configuration["gpu_strategy"] == "scpt":
            from pyop2.gpu.snpt import snpt_transform
            kernel, args_to_make_global = snpt_transform(kernel,
                                                         configuration["gpu_cells_per_block"])
        elif configuration["gpu_strategy"] == "user_specified_tile":
            from pyop2.gpu.tile import tiled_transform
            from pyop2.gpu.tile import TilingConfiguration
            kernel, args_to_make_global = tiled_transform(kernel,
                                                          program.callables_table,
                                                          TilingConfiguration(configuration["gpu_cells_per_block"],
                                                                              configuration["gpu_threads_per_cell"],
                                                                              configuration["gpu_op_tile_descriptions"],
                                                                              configuration["gpu_quad_rowtile_lengths"],
                                                                              configuration["gpu_coords_to_shared"],
                                                                              configuration["gpu_input_to_shared"],
                                                                              configuration["gpu_mats_to_shared"],
                                                                              configuration["gpu_quad_weights_to_shared"],
                                                                              configuration["gpu_tiled_prefetch_of_input"],
                                                                              configuration["gpu_tiled_prefetch_of_quad_weights"]))
        elif configuration["gpu_strategy"] == "auto_tile":
            assert args is not None
            assert argshapes is not None
            from pyop2.gpu.tile import AutoTiler
            kernel, args_to_make_global = AutoTiler(program.with_root_kernel(kernel),
                                                    configuration["gpu_planner_kernel_evals"])(args, argshapes)
        elif configuration["gpu_strategy"] == "dq_transform":
            from pyop2.gpu.dq_transform import dq_transform
            kernel, args_to_make_global = dq_transform(kernel, configuration["gpu_cells_per_block"],
                                                               configuration["gpu_threads_per_cell"])
        else:
            raise ValueError("gpu_strategy can be 'scpt', 'user_specified_tile' or 'auto_tile'.")
    elif program.name in [
            "wrap_zero", "wrap_expression_kernel",
            "wrap_expression", "wrap_pyop2_kernel_uniform_extrusion",
            "wrap_form_cell_integral_otherwise",
            "wrap_loopy_kernel_prolong",
            "wrap_loopy_kernel_restrict",
            "wrap_loopy_kernel_inject", "wrap_copy", "wrap_inner"]:
        from pyop2.gpu.snpt import snpt_transform
        kernel, args_to_make_global = snpt_transform(kernel,
                                                     configuration["gpu_cells_per_block"])
    else:
        raise NotImplementedError("Transformation for '%s'." % program.name)

    if False:
        # FIXME
        # optimization for lower order but needs some help from
        # ~firedrake.mesh~ in setting the data layout
        kernel = transpose_maps(kernel)

    program = program.with_root_kernel(kernel)

    code = lp.generate_code_v2(program).device_code()

    if program.name == "wrap_pyop2_kernel_uniform_extrusion":
        code = code.replace("inline void pyop2_kernel_uniform_extrusion", "__device__ inline void pyop2_kernel_uniform_extrusion")

    return code, program, args_to_make_global
