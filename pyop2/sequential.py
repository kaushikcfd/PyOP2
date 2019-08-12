# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""OP2 sequential backend."""

import os
import math
from copy import deepcopy as dcopy

import ctypes

from pyop2.datatypes import IntType, as_ctypes
from pyop2 import base
from pyop2 import compilation
from pyop2 import petsc_base
from pyop2.base import par_loop                          # noqa: F401
from pyop2.base import READ, WRITE, RW, INC, MIN, MAX    # noqa: F401
from pyop2.base import ALL
from pyop2.base import Map, MixedMap, Sparsity, Halo      # noqa: F401
from pyop2.base import Set, ExtrudedSet, MixedSet, Subset  # noqa: F401
from pyop2.base import DatView                           # noqa: F401
from pyop2.base import Kernel                            # noqa: F401
from pyop2.base import Arg                               # noqa: F401
from pyop2.petsc_base import DataSet, MixedDataSet       # noqa: F401
from pyop2.petsc_base import Global, GlobalDataSet       # noqa: F401
from pyop2.petsc_base import Dat, MixedDat, Mat          # noqa: F401
from pyop2.exceptions import *  # noqa: F401
from pyop2.mpi import collective
from pyop2.profiling import timed_region
from pyop2.utils import cached_property, get_petsc_dir

import loopy
from loopy.preprocess import realize_reduction_for_single_kernel
import re


class JITModule(base.JITModule):

    _cppargs = []
    _libraries = []
    _system_headers = []

    def __init__(self, kernel, iterset, *args, **kwargs):
        r"""
        A cached compiled function to execute for a specified par_loop.

        See :func:`~.par_loop` for the description of arguments.

        .. warning ::

           Note to implementors.  This object is *cached*, and therefore
           should not hold any long term references to objects that
           you want to be collected.  In particular, after the
           ``args`` have been inspected to produce the compiled code,
           they **must not** remain part of the object's slots,
           otherwise they (and the :class:`~.Dat`\s, :class:`~.Map`\s
           and :class:`~.Mat`\s they reference) will never be collected.
        """
        # Return early if we were in the cache.
        if self._initialized:
            return
        self.comm = iterset.comm
        self._kernel = kernel
        self._fun = None
        self._iterset = iterset
        self._args = args
        self._iteration_region = kwargs.get('iterate', ALL)
        self._pass_layer_arg = kwargs.get('pass_layer_arg', False)
        # Copy the class variables, so we don't overwrite them
        self._cppargs = dcopy(type(self)._cppargs)
        self._libraries = dcopy(type(self)._libraries)
        self._system_headers = dcopy(type(self)._system_headers)
        if not kwargs.get('delay', False):
            self.compile()
            self._initialized = True

    @collective
    def __call__(self, *args):
        return self._fun(*args)

    @cached_property
    def _wrapper_name(self):
        return 'wrap_%s' % self._kernel.name

    @cached_property
    def code_to_compile(self):

        from pyop2.codegen.builder import WrapperBuilder
        from pyop2.codegen.rep2loopy import generate

        builder = WrapperBuilder(iterset=self._iterset, iteration_region=self._iteration_region, pass_layer_to_kernel=self._pass_layer_arg)
        for arg in self._args:
            builder.add_argument(arg)
        builder.set_kernel(self._kernel)

        wrapper = generate(builder)
        code = generate_cuda_kernel(wrapper)

        if self._kernel._cpp:
            from loopy.codegen.result import process_preambles
            preamble = "".join(process_preambles(getattr(code, "device_preambles", [])))
            device_code = "\n\n".join(str(dp.ast) for dp in code.device_programs)
            return preamble + "\nextern \"C\" {\n" + device_code + "\n}\n"
        return code.device_code()

    @collective
    def compile(self):
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        from pyop2.configuration import configuration

        compiler = configuration["compiler"]
        extension = "cpp" if self._kernel._cpp else "c"
        cppargs = self._cppargs
        cppargs += ["-I%s/include" % d for d in get_petsc_dir()] + \
                   ["-I%s" % d for d in self._kernel._include_dirs] + \
                   ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
        ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
                 ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
                 ["-lpetsc", "-lm"] + self._libraries
        ldargs += self._kernel._ldargs

        self._fun = compilation.load(self,
                                     extension,
                                     self._wrapper_name,
                                     cppargs=cppargs,
                                     ldargs=ldargs,
                                     restype=ctypes.c_int,
                                     compiler=compiler,
                                     comm=self.comm)
        # Blow away everything we don't need any more
        del self._args
        del self._kernel
        del self._iterset

    @cached_property
    def argtypes(self):
        index_type = as_ctypes(IntType)
        argtypes = (index_type, index_type)
        argtypes += self._iterset._argtypes_
        for arg in self._args:
            argtypes += arg._argtypes_
        seen = set()
        for arg in self._args:
            maps = arg.map_tuple
            for map_ in maps:
                for k, t in zip(map_._kernel_args_, map_._argtypes_):
                    if k in seen:
                        continue
                    argtypes += (t,)
                    seen.add(k)
        return argtypes


class ParLoop(petsc_base.ParLoop):

    def prepare_arglist(self, iterset, *args):
        arglist = iterset._kernel_args_
        for arg in args:
            arglist += arg._kernel_args_
        seen = set()
        for arg in args:
            maps = arg.map_tuple
            for map_ in maps:
                if map_ is None:
                    continue
                for k in map_._kernel_args_:
                    if k in seen:
                        continue
                    arglist += (k,)
                    seen.add(k)
        return arglist

    @cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.iterset, *self.args,
                         iterate=self.iteration_region,
                         pass_layer_arg=self._pass_layer_arg)

    @collective
    def _compute(self, part, fun, *arglist):
        with timed_region("ParLoop_{0}_{1}".format(self.iterset.name, self._jitmodule._wrapper_name)):
            fun(part.offset, part.offset + part.size, *arglist)


def generate_single_cell_wrapper(iterset, args, forward_args=(), kernel_name=None, wrapper_name=None):
    """Generates wrapper for a single cell. No iteration loop, but cellwise data is extracted.
    Cell is expected as an argument to the wrapper. For extruded, the numbering of the cells
    is columnwise continuous, bottom to top.

    :param iterset: The iteration set
    :param args: :class:`Arg`s
    :param forward_args: To forward unprocessed arguments to the kernel via the wrapper,
                         give an iterable of strings describing their C types.
    :param kernel_name: Kernel function name
    :param wrapper_name: Wrapper function name

    :return: string containing the C code for the single-cell wrapper
    """
    from pyop2.codegen.builder import WrapperBuilder
    from pyop2.codegen.rep2loopy import generate
    from loopy.types import OpaqueType

    forward_arg_types = [OpaqueType(fa) for fa in forward_args]
    builder = WrapperBuilder(iterset=iterset, single_cell=True, forward_arg_types=forward_arg_types)
    for arg in args:
        builder.add_argument(arg)
    builder.set_kernel(Kernel("", kernel_name))
    wrapper = generate(builder, wrapper_name)
    code = loopy.generate_code_v2(wrapper)

    return code.device_code()


def _make_tv_array_arg(tv):
    assert tv.address_space != loopy.AddressSpace.PRIVATE
    arg = loopy.ArrayArg(
            name=tv.name,
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


def transform(kernel, callables_table, ncells_per_block=32,
        nthreads_per_cell=1,
        matvec1_parallelize_across='row', matvec2_parallelize_across='row',
        matvec1_rowtiles=1, matvec1_coltiles=1,
        matvec2_rowtiles=1, matvec2_coltiles=1,
        load_coordinates_to_shared=False,
        load_input_to_shared=False,
        prefetch_tiles=True):

    # {{{ FIXME: Setting names which should be set by TSFC

    quad_iname = 'form_ip'
    output_basis_coeff_temp = 't2'
    input_basis_coeff_temp = 't0'
    scatter_iname = 'i4'
    basis_iname_in_basis_redn = 'form_j'
    quad_iname_in_basis_redn = 'form_ip_basis'
    quad_iname_in_quad_redn = 'form_ip_quad'
    basis_iname_in_quad_redn = 'form_i'
    basis_iname_basis_redn = 'form_j'

    # }}}

    # {{{ sanity checks

    #FIXME: Let's keep on writing this code and visit this later, surely
    # someone will enter failing options.

    assert matvec1_parallelize_across in ['row', 'column']
    assert matvec2_parallelize_across in ['row', 'column']

    # }}}

    # {{{ reading info about the finite element

    nquad = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_ip', constants_only=True).size))
    nbasis = int(loopy.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_j', constants_only=True).size))

    # }}}

    # {{{ tagging the stages of the kernel

    # FIXME: This should be interpreted in TSFC

    new_insns = []

    done_with_jacobi_eval = False
    done_with_quad_init = False
    done_with_quad_reduction = False
    done_with_quad_wrap_up = False
    done_with_basis_reduction = False

    for insn in kernel.instructions:
        if not done_with_jacobi_eval:
            if 'form_ip' in insn.within_inames:
                done_with_jacobi_eval = True

            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["jacobi_eval"])))
                continue
        if not done_with_quad_init:
            if 'form_i' in insn.within_inames:
                done_with_quad_init = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_init"])))
                continue
        if not done_with_quad_reduction:
            if 'form_i' not in insn.within_inames:
                done_with_quad_reduction = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_redn"])))
                continue
        if not done_with_quad_wrap_up:
            if 'basis' in insn.tags:
                done_with_quad_wrap_up = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_wrap_up"])))
                continue
        if not done_with_basis_reduction:
            if 'form_ip' not in insn.within_inames:
                done_with_basis_reduction = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["basis_redn"])))
                continue
        new_insns.append(insn)

    assert done_with_basis_reduction

    kernel = kernel.copy(instructions=new_insns)

    # }}}

    # {{{ privatize temps for function evals and make them LOCAL

    #FIXME: Need these variables from TSFC's metadata
    # This helps to apply transformations separately to the basis part and the
    # quadrature part

    evaluation_variables = (set().union(*[insn.write_dependency_names() for insn in kernel.instructions if 'quad_wrap_up' in insn.tags])
            & set().union(*[insn.read_dependency_names() for insn in kernel.instructions if 'basis' in insn.tags]))

    kernel = loopy.privatize_temporaries_with_inames(kernel, 'form_ip',
            evaluation_variables)
    new_temps = kernel.temporary_variables.copy()
    for eval_var in evaluation_variables:
        new_temps[eval_var] = new_temps[eval_var].copy(
                address_space=loopy.AddressSpace.LOCAL)
    kernel = kernel.copy(temporary_variables=new_temps)

    # }}}

    # {{{ change address space of constants to '__global'

    old_temps = kernel.temporary_variables.copy()
    args_to_make_global = [tv.initializer.flatten() for tv in old_temps.values() if tv.initializer is not None]

    new_temps = dict((tv.name, tv) for tv in old_temps.values() if tv.initializer is None)
    kernel = kernel.copy(
            args=kernel.args+[_make_tv_array_arg(tv) for tv in old_temps.values() if tv.initializer is not None],
            temporary_variables=new_temps)

    # }}}

    #FIXME: Assumes the variable associated with output is 't0'. GENERALIZE THIS!
    kernel = loopy.remove_instructions(kernel, "writes:{} and tag:gather".format(output_basis_coeff_temp))
    kernel = loopy.remove_instructions(kernel, "tag:quad_init")

    from loopy.transform.convert_to_reduction import convert_to_reduction
    kernel = convert_to_reduction(kernel, 'tag:quad_redn', ('form_i', ))
    kernel = convert_to_reduction(kernel, 'tag:basis_redn', ('form_ip', ))

    from loopy.loop import fuse_loop_domains
    kernel = fuse_loop_domains(kernel)

    from loopy.transform.data import remove_unused_axes_in_temporaries
    kernel = remove_unused_axes_in_temporaries(kernel)

    # {{{ remove noops

    noop_insns = set([insn.id for insn in kernel.instructions if
            isinstance(insn, loopy.NoOpInstruction)])
    kernel = loopy.remove_instructions(kernel, noop_insns)

    # }}}

    # Realize CUDA blocks
    kernel = loopy.split_iname(kernel, "n", ncells_per_block*nthreads_per_cell,
            outer_iname="iblock", inner_iname="icell")
    #FIXME: Do not use hard-coded inames, this change should also be in TSFC.
    kernel = loopy.rename_iname(kernel, scatter_iname,
            basis_iname_in_basis_redn, True)

    # Duplicate inames to separate transformation logic for quadrature and basis part
    kernel = loopy.duplicate_inames(kernel, quad_iname, "tag:quadrature",
            quad_iname_in_quad_redn)
    kernel = loopy.duplicate_inames(kernel, quad_iname, "tag:basis",
            quad_iname_in_basis_redn)

    if load_coordinates_to_shared:
        #FIXME: Assumes uses the name 't1' for coordinates
        kernel = loopy.privatize_temporaries_with_inames(kernel, 'icell',
                [coords_temp])
        kernel = loopy.assignment_to_subst(kernel, coords_temp)
        raise NotImplementedError()

    if load_input_to_shared:
        #FIXME: Assumes uses the name 't2' for the input basis coeffs
        kernel = loopy.privatize_temporaries_with_inames(kernel, 'icell',
                [input_basis_coeff_temp])
        kernel = loopy.assignment_to_subst(kernel, input_basis_coeff_temp)
        raise NotImplementedError()

    # compute tile lengths
    matvec1_row_tile_length = math.ceil(nquad // matvec1_rowtiles)
    matvec1_col_tile_length = math.ceil(nbasis // matvec1_coltiles)
    matvec2_row_tile_length = math.ceil(nbasis // matvec2_rowtiles)
    matvec2_col_tile_length = math.ceil(nquad // matvec2_coltiles)

    # Splitting for tiles in matvec1
    kernel = loopy.split_iname(kernel, quad_iname_in_quad_redn, matvec1_row_tile_length, outer_iname='irowtile_matvec1')
    kernel = loopy.split_iname(kernel, basis_iname_in_quad_redn, matvec1_col_tile_length, outer_iname='icoltile_matvec1')

    # Splitting for tiles in matvec2
    kernel = loopy.split_iname(kernel, basis_iname_in_basis_redn, matvec2_row_tile_length, outer_iname='irowtile_matvec2')
    kernel = loopy.split_iname(kernel, quad_iname_in_basis_redn, matvec2_col_tile_length, outer_iname='icoltile_matvec2')

    # {{{ Prefetch wizardry

    if prefetch_tiles:
        from loopy.transform.data import add_prefetch_for_single_kernel
        #FIXME: Assuming that in all the constants the one with single axis is
        # the one corresponding to quadrature weights. fix it by passing some
        # metadata from TSFC.
        # FIXME: Sweep inames depends on the parallelization strategies for
        # both the matvecs, that needs to be taken care of.
        const_matrices_names = set([tv.name for tv in old_temps.values() if tv.initializer is not None and len(tv.shape)>1])
        quad_weights, = [tv.name for tv in old_temps.values() if tv.initializer is not None and len(tv.shape) == 1]

        # {{{ Prefetching: QUAD PART

        quad_const_matrices = const_matrices_names & frozenset().union(*[insn.read_dependency_names() for insn in
            kernel.instructions if 'quad_redn' in insn.tags])
        sweep_inames = (quad_iname_in_quad_redn+'_inner',
                basis_iname_in_quad_redn+'_inner')
        fetch_outer_inames = 'iblock,icoltile_matvec1,irowtile_matvec1'

        quad_prefetch_insns = []

        vng = kernel.get_var_name_generator()
        ing = kernel.get_instruction_id_generator()
        quad_temp_names = [vng('quad_cnst_mtrix_prftch') for _ in quad_const_matrices]
        prefetch_inames = [vng("iprftch") for _ in range(2)]
        for temp_name, var_name in zip(quad_temp_names, quad_const_matrices):
            quad_prefetch_insns.append(ing("quad_prftch_insn"))

            kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                    var_name=var_name,
                    sweep_inames=sweep_inames,
                    temporary_address_space=loopy.AddressSpace.LOCAL,
                    dim_arg_names=prefetch_inames,
                    temporary_name=temp_name,
                    compute_insn_id=quad_prefetch_insns[-1],
                    fetch_outer_inames=fetch_outer_inames,
                    default_tag=None,
                    within="tag:quad_redn")

        #FIXME: In order to save on compilation time we are not sticking to
        # coalesced accesses Otherwise we should join the following inames and
        # then split into nthreads_per_cell

        kernel = loopy.split_iname(kernel, prefetch_inames[1], nthreads_per_cell)
        kernel = loopy.tag_inames(kernel, {prefetch_inames[0]: "l.1", prefetch_inames[1]+"_inner": "l.0"})

        # }}}

        # {{{ Prefetching: BASIS PART

        basis_const_matrices = const_matrices_names & frozenset().union(*[insn.read_dependency_names() for insn in
            kernel.instructions if 'basis_redn' in insn.tags])
        basis_temp_names = [vng('basis_cnst_mtrix_prftch') for _ in basis_const_matrices]

        sweep_inames = (basis_iname_in_basis_redn+'_inner',
                quad_iname_in_basis_redn+'_inner')
        fetch_outer_inames = 'iblock,icoltile_matvec2,irowtile_matvec2'

        basis_prefetch_insns = []
        prefetch_inames = [vng("iprftch") for _ in range(2)]
        for temp_name, var_name in zip(basis_temp_names, basis_const_matrices):
            basis_prefetch_insns.append(ing("basis_prftch_insn"))

            kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                    var_name=var_name,
                    sweep_inames=sweep_inames,
                    temporary_address_space=loopy.AddressSpace.LOCAL,
                    dim_arg_names=prefetch_inames,
                    temporary_name=temp_name,
                    compute_insn_id=basis_prefetch_insns[-1],
                    fetch_outer_inames=fetch_outer_inames,
                    default_tag=None,
                    within="tag:basis_redn")

        # See FIXME for the quad part at this point
        kernel = loopy.split_iname(kernel, prefetch_inames[1], nthreads_per_cell)
        kernel = loopy.tag_inames(kernel, {prefetch_inames[0]: "l.1", prefetch_inames[1]+"_inner": "l.0"})

        # }}}

        # {{{ Prefetch: Quad Weights(Set to false now)

        # Unless we load this into the shared memory and do a collective read
        # in a block, this is no good. As the quad weights are accessed only
        # once. So the only way prefetching would help is through a
        # parallelized read.

        prefetch_quad_weights = False

        if prefetch_quad_weights:
            quad_weight_prefetch_insns = []

            if matvec1_parallelize_across == 'row':
                sweep_inames = (quad_iname_in_quad_redn+'_inner_outer', quad_iname_in_quad_redn+'_inner_inner',)
                fetch_outer_inames = 'irowtile_matvec1, icell, iblock'
            else:
                raise NotImplementedError()
            quad_weight_prefetch_insns.append(ing("basis_prftch_insn"))

            kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                    var_name=quad_weights,
                    sweep_inames=sweep_inames,
                    temporary_address_space=loopy.AddressSpace.PRIVATE,
                    temporary_name='cnst_quad_weight_prftch',
                    compute_insn_id=quad_weight_prefetch_insns[-1],
                    fetch_outer_inames=fetch_outer_inames,
                    within="tag:quad_wrap_up")
        # }}}

        # {{{ Adding dependency between the prefetch instructions

        kernel = loopy.add_dependency(kernel,
                " or ".join("id:{}".format(insn_id) for insn_id in
                    basis_prefetch_insns), "tag:quadrature")

        # }}}

        from loopy.transform.data import flatten_variable, absorb_temporary_into
        for var_name in quad_temp_names+basis_temp_names:
            kernel = flatten_variable(kernel, var_name)
        for quad_temp_name, basis_temp_name in zip(quad_temp_names,
                basis_temp_names):
            if (matvec2_row_tile_length*matvec2_col_tile_length >= matvec1_row_tile_length*matvec1_col_tile_length):
                kernel = absorb_temporary_into(kernel, basis_temp_name, quad_temp_name)
            else:
                kernel = absorb_temporary_into(kernel, quad_temp_name, basis_temp_name)

        kernel = loopy.add_dependency(kernel, 'tag:quad_redn', 'id:quad_prftch_insn*')
        kernel = loopy.add_dependency(kernel, 'tag:basis_redn', 'id:basis_prftch_insn*')

        # do not enforce any dependency between the basis reductions and the
        # quadrature reductions.

        kernel = loopy.remove_dependency(kernel, 'tag:quad_redn', 'tag:quad_redn')
        kernel = loopy.remove_dependency(kernel, 'tag:basis_redn', 'tag:basis_redn')
        kernel = loopy.add_dependency(kernel, 'tag:quad_wrap_up', 'tag:quad_redn')

    # }}}

    # {{{ divide matvec1-tile's work across threads

    if matvec1_parallelize_across == 'row':
        kernel = loopy.split_iname(kernel, quad_iname_in_quad_redn+'_inner', nthreads_per_cell, inner_tag="l.0")
    else:
        kernel = loopy.split_iname(kernel, basis_iname_in_quad_redn+'_inner', nthreads_per_cell, inner_tag="l.0")
        kernel = loopy.split_reduction_inward(kernel, basis_iname_in_quad_redn+'_inner_outer')
        kernel = loopy.split_reduction_inward(kernel, basis_iname_in_quad_redn+'_inner_inner')

    # }}}

    # {{{ diving matvec2-tile's work across threads

    if matvec2_parallelize_across == 'row':
        kernel = loopy.split_iname(kernel, basis_iname_in_basis_redn+'_inner', nthreads_per_cell, inner_tag="l.0")
    else:
        kernel = loopy.split_iname(kernel, quad_iname_in_basis_redn+'_inner', nthreads_per_cell, inner_tag="l.0")
        kernel = loopy.split_reduction_inward(kernel, quad_iname_in_basis_redn+'_inner_outer')
        kernel = loopy.split_reduction_inward(kernel, quad_iname_in_basis_redn+'_inner_inner')

    # }}}

    # {{{ mico-optimizations(None implemented yet)

    #FIXME: Need to set the variables 'remove_func_eval_arrays' depending on
    # the input parameters to 'transform'
    # So, currently we don't support this
    # If 'remove_func_eval_arrays' is set True then the following transformations must be performed
    # 1. Use scalars instead of arrays for the variables produced in
    #    quad_wrap_up.
    # 2. Use the same iname for 'form_ip_basis', 'form_ip_quad'
    #
    # These would be the micro-optimization to use less register space for
    # SCPT.

    remove_func_eval_arrays = False
    if remove_func_eval_arrays:
        raise NotImplementedError()

    # Should trigger when matvec1_parallelize_across = 'col'
    # Then no need to put the LHS into shared memory.
    do_not_prefetch_lhs = False
    if do_not_prefetch_lhs:
        raise NotImplementedError()

    # Again for SCPT we need the mico-optimization that we put the constant
    # matrices into the constant memory for broadcasting purposes.

    # }}}

    #FIXME: Need to fix the shape of t0 to whatever portion we are editing.
    # the address space of t0 depends on the parallelization strategy.
    kernel = realize_reduction_for_single_kernel(kernel, callables_table)

    # Just translate all the dependencies of form_insn_14 to form_insn_15
    for insn in kernel.instructions:
        if re.match(".*form_insn_14.*", insn.id):
            insn_15_eq = re.sub("(.*)form_insn_14(.*)",
                    "\g<1>form_insn_15\g<2>",
                    insn.id)
            new_depends_on = []
            for depends in insn.depends_on:
                if re.match(".*form_insn_14.*", insn.id):
                    kernel = loopy.add_dependency(kernel,
                            "id:{}".format(insn_15_eq),
                            "id:{}".format(depends))
                    kernel = loopy.add_dependency(kernel,
                            "id:{}".format(insn.id),
                            "id:{}".format(re.sub(
                                "(.*)form_insn_14(.*)",
                                "\g<1>form_insn_15\g<2>",
                                depends)))


    if matvec1_parallelize_across == 'row':

        kernel = loopy.privatize_temporaries_with_inames(kernel,
                'form_ip_quad_inner_outer',
                only_var_names=['acc_icoltile_matvec1_form_i_inner',
                    'acc_icoltile_matvec1_form_i_inner_0', 'form_t16', 'form_t17'])
        kernel = loopy.duplicate_inames(kernel, ['form_ip_quad_inner_outer', ],
                within='tag:quad_wrap_up or'
                ' id:red_assign_form_insn_14 or id:red_assign_form_insn_15')

        kernel = loopy.duplicate_inames(kernel,
                ['form_ip_quad_inner_outer'],
                'id:form_insn_14_icoltile_matvec1_form_i_inner_init or id:form_insn_15_icoltile_matvec1_form_i_inner_init')
    else:

        from loopy.transform.batch import save_temporaries_in_loop

        kernel = save_temporaries_in_loop(kernel, 'form_ip_quad_inner', [
            'acc_icoltile_matvec1', 'acc_icoltile_matvec1_0',
            'acc_form_i_inner_inner_0', 'acc_form_i_inner_inner',
            'acc_form_i_inner_outer', 'acc_form_i_inner_outer_0', ],
            within="iname:form_ip_quad_inner")

        reduction_assignees = tuple(insn.assignee for insn in kernel.instructions
                if 'quad_redn' in insn.tags)

        # FIXME: These variables should be named using transform addresssing.
        kernel = loopy.assignment_to_subst(kernel, 'neutral_form_i_inner_inner')
        kernel = loopy.assignment_to_subst(kernel, 'neutral_form_i_inner_inner_0')

        kernel = loopy.assignment_to_subst(kernel, "form_t18")
        kernel = loopy.assignment_to_subst(kernel, "form_t19")
        kernel = loopy.assignment_to_subst(kernel, "form_t20")

        for assignee in reduction_assignees:
            kernel = loopy.assignment_to_subst(kernel, assignee.name)

        # {{{ duplicate form_ip_quad_inner in a bunch of equations

        for i in range(int(math.ceil(math.log2(nthreads_per_cell)))):
            kernel = loopy.duplicate_inames(kernel, "form_ip_quad_inner",
                    within="id:red_stage_{0}_form_i_inner_inner_form_insn_14_icoltile_matvec1_update or "
                    "id:red_stage_{0}_form_i_inner_inner_form_insn_15_icoltile_matvec1_update".format(i),)

        kernel = loopy.duplicate_inames(kernel, "form_ip_quad_inner",
                within="id:form_insn_14_icoltile_matvec1_init or id:form_insn_15_icoltile_matvec1_init")
        kernel = loopy.duplicate_inames(kernel, "form_ip_quad_inner",
                within="id:red_assign_form_insn_14_icoltile_matvec1_update or id:red_assign_form_insn_15_icoltile_matvec1_update")
        kernel = loopy.duplicate_inames(kernel, "form_ip_quad_inner", within="tag:quad_wrap_up")
        # }}}

        # {{{ adding l.0 hw axes to some axes.

        kernel = loopy.add_inames_to_insn(kernel,
                inames="red_form_i_inner_inner_s{0}_1".format(int(math.ceil(math.log2(nthreads_per_cell)))-1),
                insn_match="id:form_insn_14_icoltile_matvec1_init")
        kernel = loopy.add_inames_to_insn(kernel,
                inames="red_form_i_inner_inner_s{0}_1".format(int(math.ceil(math.log2(nthreads_per_cell)))-1),
                insn_match="id:form_insn_15_icoltile_matvec1_init")
        kernel = loopy.add_inames_to_insn(kernel,
                inames="red_form_i_inner_inner_s{0}_1".format(int(math.ceil(math.log2(nthreads_per_cell)))-1),
                insn_match="tag:quad_wrap_up")

        # }}}

    if matvec2_parallelize_across == 'row':
        kernel = loopy.privatize_temporaries_with_inames(kernel, 'form_j_inner_outer',
                only_var_names=['acc_icoltile_matvec2_form_ip_basis_inner'])
        kernel = loopy.duplicate_inames(kernel, ['form_j_inner_outer'], within='tag:scatter or'
                ' id:red_assign_form_insn_21')
        kernel = loopy.duplicate_inames(kernel,
                ['form_j_inner_outer'],
                'id:form_insn_21_icoltile_matvec2_form_ip_basis_inner_init')
    else:
        kernel = loopy.assignment_to_subst(kernel, 'neutral_form_ip_basis_inner_inner')
        from loopy.transform.batch import save_temporaries_in_loop
        # FIXME: Maybe we do not need to privatize "acc_icoltile_matvec2"?
        kernel = loopy.save_temporaries_in_loop(kernel,
                'form_j_inner',
                [
                    "acc_icoltile_matvec2",
                    "acc_form_ip_basis_inner_inner"
                    ],
                within="iname:form_j_inner")

        for i in range(int(math.ceil(math.log2(nthreads_per_cell)))):
            kernel = loopy.duplicate_inames(kernel,
                    "form_j_inner",
                    "id:red_stage_{0}_form_ip_basis_inner_inner_form_insn_21_icoltile_matvec2_update".format(i))
        
        kernel = loopy.duplicate_inames(kernel,
                "form_j_inner",
                within="id:form_insn_21_icoltile_matvec2_init")

        kernel = loopy.duplicate_inames(kernel,
                "form_j_inner",
                within="id:red_assign_form_insn_21_icoltile_matvec2_update")

        kernel = loopy.duplicate_inames(kernel,
                "form_j_inner",
                "tag:scatter or id:red_assign_form_insn_21")

        kernel = loopy.add_inames_to_insn(kernel,
                inames="red_form_ip_basis_inner_inner_s{0}_0".format(int(math.ceil(math.log2(nthreads_per_cell)))-1),
                insn_match="id:form_insn_21_icoltile_matvec2_init")


        kernel = loopy.add_inames_to_insn(kernel,
                inames="red_form_ip_basis_inner_inner_s{0}_0".format(int(math.ceil(math.log2(nthreads_per_cell)))-1),
                insn_match="id:red_assign_form_insn_21 or tag:scatter")

    kernel = loopy.tag_inames(kernel, "icell:l.1, iblock:g.0")
    kernel = loopy.prioritize_loops(kernel, ("irowtile_matvec1", "icoltile_matvec1"))
    kernel = loopy.prioritize_loops(kernel, ("irowtile_matvec2", "icoltile_matvec2"))

    return kernel, args_to_make_global


def generate_cuda_kernel(program, extruded=False):
    # Kernel transformations
    args_to_make_global = []
    program = program.copy(target=loopy.CudaTarget())
    kernel = program.root_kernel

    def insn_needs_atomic(insn):
        # updates to global variables are atomic
        assignee_name = insn.assignee.aggregate.name
        return assignee_name in insn.read_dependency_names() and assignee_name not in kernel.temporary_variables

    new_insns = []
    args_marked_for_atomic = set()
    for insn in kernel.instructions:
        if ('scatter' in insn.tags):
            if insn_needs_atomic(insn):
                atomicity = (loopy.AtomicUpdate(insn.assignee.aggregate.name), )
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

    if kernel.name == "wrap_form0_cell_integral_otherwise":
        kernel = loopy.fix_parameters(kernel, start=0)
        kernel = loopy.assume(kernel, "end > 0")

        # choose the preferred algorithm here
        kernel, args_to_make_global = transform(kernel, program.callables_table,
                nthreads_per_cell=3,
                matvec1_parallelize_across='row',
                matvec2_parallelize_across='column',
                matvec1_rowtiles=1, matvec1_coltiles=2,
                matvec2_rowtiles=2, matvec2_coltiles=1,
                prefetch_tiles=True,)

        # FIXME: Once everything around this generalized transformation is
        # handled, make this guy relive.
        # If this does not hold up currently then we will have a kernel which
        # would give us a wrong answer.
        # kernel = transpose_maps(kernel)
    else:
        # batch cells into groups
        # essentially, each thread computes unroll_size elements, each block computes unroll_size*block_size elements
        batch_size = 32  # configuration["cuda_block_size"]
        unroll_size = 1  # configuration["cuda_unroll_size"]

        kernel = loopy.assume(kernel, "{0} mod {1} = 0".format("end", batch_size*unroll_size))
        kernel = loopy.assume(kernel, "exists zz: zz > 0 and {0} = {1}*zz + {2}".format("end", batch_size*unroll_size, "start"))

        if unroll_size > 1:
            kernel = loopy.split_iname(kernel, "n", unroll_size, inner_tag="ilp")
            kernel = loopy.split_iname(kernel, "n_outer", batch_size, inner_tag="l.0", outer_tag="g.0")
        else:
            kernel = loopy.split_iname(kernel, "n", batch_size, inner_tag="l.0", outer_tag="g.0")

    program = program.with_root_kernel(kernel)

    code = loopy.generate_code_v2(program).device_code()
    if program.name == "wrap_pyop2_kernel_uniform_extrusion":
        code = code.replace("inline void pyop2_kernel_uniform_extrusion", "__device__ inline void pyop2_kernel_uniform_extrusion")

    if program.name == "wrap_form0_cell_integral_otherwise":
        print("Generated code")
        print(code)
        1/0
        pass
        # with open('current_kernel.cu', 'w') as f:
        #     # code = f.read()
        #     f.write(code)

    return code

# vim:fdm=marker
