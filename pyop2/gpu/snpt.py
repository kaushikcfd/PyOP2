import loopy as lp


def _make_tv_array_arg(tv):
    assert tv.address_space != lp.AddressSpace.PRIVATE
    arg = lp.ArrayArg(name=tv.name,
                      dtype=tv.dtype,
                      shape=tv.shape,
                      dim_tags=tv.dim_tags,
                      offset=tv.offset,
                      dim_names=tv.dim_names,
                      order=tv.order,
                      alignment=tv.alignment,
                      address_space=tv.address_space,
                      is_output_only=not tv.read_only)
    return arg


def snpt_transform(kernel, block_size):
    """
    SNPT := Single 'n' Per Thread.

    Implements outer-loop parallelization strategy.

    PyOP2 uses 'n' as the outer loop iname. In Firedrake 'n' might denote
    either a cell or a DOF.
    """
    kernel = lp.assume(kernel, "start < end")
    kernel = lp.split_iname(kernel, "n", block_size, outer_tag="g.0", inner_tag="l.0")

    # {{{ making consts as globals: necessary to make the strategy emit valid
    # kernels for all forms

    old_temps = kernel.temporary_variables.copy()
    args_to_make_global = [tv.initializer.flatten()
                           for tv in old_temps.values()
                           if tv.initializer is not None]

    new_temps = {tv.name: tv
                 for tv in old_temps.values()
                 if tv.initializer is None}
    kernel = kernel.copy(args=kernel.args+[_make_tv_array_arg(tv)
                                           for tv in old_temps.values()
                                           if tv.initializer is not None],
                         temporary_variables=new_temps)

    # }}}

    return kernel, args_to_make_global
