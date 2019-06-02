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

"""OP2 CUDA backend."""

import os
import ctypes
from copy import deepcopy as dcopy

from contextlib import contextmanager

from pyop2.datatypes import IntType, as_ctypes
from pyop2 import base
from pyop2 import compilation
from pyop2 import petsc_base
from pyop2.base import par_loop                          # noqa: F401
from pyop2.base import READ, WRITE, RW, INC, MIN, MAX    # noqa: F401
from pyop2.base import ALL
from pyop2.base import Map, MixedMap, DecoratedMap, Sparsity, Halo  # noqa: F401
from pyop2.base import Set, ExtrudedSet, MixedSet, Subset  # noqa: F401
from pyop2.base import DatView                           # noqa: F401
from pyop2.base import Kernel                            # noqa: F401
from pyop2.base import Arg                               # noqa: F401
from pyop2.petsc_base import DataSet, MixedDataSet       # noqa: F401
from pyop2.petsc_base import GlobalDataSet       # noqa: F401
from pyop2.petsc_base import Dat as petsc_Dat
from pyop2.petsc_base import Global as petsc_Global
from pyop2.petsc_base import PETSc, MixedDat, Mat          # noqa: F401
from pyop2.exceptions import *  # noqa: F401
from pyop2.mpi import collective
from pyop2.profiling import timed_region, timed_function
from pyop2.utils import cached_property, get_petsc_dir
from pyop2.configuration import configuration
from pyop2.logger import ExecTimeNoter

import loopy
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda_driver
import numpy as np
import islpy
from collections import OrderedDict
from pytools import memoize_method


class Map(Map):
    @cached_property
    def device_handle(self):
        m_gpu = cuda_driver.mem_alloc(int(self.values.nbytes))
        cuda_driver.memcpy_htod(m_gpu, self.values.flatten())
        return m_gpu

    @cached_property
    def _kernel_args_(self):
        return (self.device_handle, )


class Arg(Arg):

    pass


class Dat(petsc_Dat):
    """
    Dat for GPU.
    """

    @contextmanager
    def vec_context(self, access):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Dat`.

        :param access: Access descriptor: READ, WRITE, or RW."""

        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # Getting the Vec needs to ensure we've done all current
        # necessary computation.
        self._force_evaluation(read=access is not base.WRITE,
                               write=access is not base.READ)
        if not hasattr(self, '_vec'):
            # Can't duplicate layout_vec of dataset, because we then
            # carry around extra unnecessary data.
            # But use getSizes to save an Allreduce in computing the
            # global size.
            size = self.dataset.layout_vec.getSizes()
            data = self._data[:size[0]]
            self._vec = PETSc.Vec().create(self.comm)
            self._vec.setSizes(size=size, bsize=self.cdim)
            self._vec.setType('cuda')
            self._vec.setArray(data)
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()
        yield self._vec
        if access is not base.READ:
            self.halo_valid = False

    @cached_property
    def device_handle(self):
        with self.vec as v:
            return v.getCUDAHandle()

    @cached_property
    def _kernel_args_(self):
        return (self.device_handle, )

    @collective
    @property
    def data(self):

        from pyop2.base import _trace

        _trace.evaluate(set([self]), set([self]))

        with self.vec as v:
            v.restoreCUDAHandle(self.device_handle)
            return v.array

        # cuda_driver.memcpy_dtoh(self.data, self.device_handle)
        # return self.data


class Global(petsc_Global):
    @contextmanager
    def vec_context(self, access):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Global`.

        :param access: Access descriptor: READ, WRITE, or RW."""

        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # Getting the Vec needs to ensure we've done all current
        # necessary computation.
        self._force_evaluation(read=access is not base.WRITE,
                               write=access is not base.READ)
        data = self._data
        if not hasattr(self, '_vec'):
            # Can't duplicate layout_vec of dataset, because we then
            # carry around extra unnecessary data.
            # But use getSizes to save an Allreduce in computing the
            # global size.
            size = self.dataset.layout_vec.getSizes()
            self._vec = PETSc.Vec().create(self.comm)
            self._vec.setSizes(size=size, bsize=self.cdim)
            self._vec.setType('cuda')
            self._vec.setArray(data)
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()
        yield self._vec

    @cached_property
    def device_handle(self):
        with self.vec as v:
            return v.getCUDAHandle()

    @cached_property
    def _kernel_args_(self):
        return (self.device_handle, )


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
        self.processed_program = None
        self.args_to_make_global = []
        self.extruded = self._iterset._extruded

        if not kwargs.get('delay', False):
            self.compile()
            self._initialized = True

    @memoize_method
    def grid_size(self, start, end):
        from pymbolic import evaluate

        parameters = {'start': start, 'end': end}
        glens, llens = self.processed_program.get_grid_size_upper_bounds_as_exprs()
        grid_y = 1
        if self.extruded:
            grid_y = glens[1]

        grid_y = 1
        if self.extruded:
            grid_y = glens[1]
        grid = (int(evaluate(glens, parameters)[0]), grid_y)

        block = tuple(int(k) for k in evaluate(llens, parameters))

        block += (3-len(block))*(1, )
        assert len(block) == 3

        return grid, block

    @cached_property
    def get_args_marked_for_globals(self):
        const_args_as_globals = tuple(cuda_driver.mem_alloc(arg.nbytes) for arg in
            self.args_to_make_global)
        for arg_gpu, arg in zip(const_args_as_globals,
                self.args_to_make_global):
            cuda_driver.memcpy_htod(arg_gpu, arg)

        evt = cuda_driver.Event()
        evt.record()
        evt.synchronize()

        return const_args_as_globals

    @memoize_method
    def print_data_layout_info(self, args):
        print(75*"=")
        print("WARNING: These numbers should be only trusted for a P2 element")
        print(75*"=")

        import pycuda.driver as cuda
        dev_map_basis, dev_map_coords = args[-2:]
        map_basis = np.empty(dtype=self.processed_program.args[-2].dtype, shape=(6, args[1]-args[0]))
        map_coords = np.empty(dtype=self.processed_program.args[-1].dtype, shape=(3, args[1]-args[0]))
        cuda.memcpy_dtoh(dest=map_basis, src=dev_map_basis)
        cuda.memcpy_dtoh(dest=map_coords, src=dev_map_coords)

        print(map_coords)
        print(map_basis)
        print(75*"=")

        print('Basis max:', np.max(map_basis))
        print('Coords max:', np.max(map_coords))

        # map_basis = map_basis.T
        map_basis = map_basis.flatten()
        map_basis = map_basis.reshape((args[1]*6) // 32, 32)
        map_basis = map_basis // 4
        accesses = []
        for warp in map_basis:
            accesses.append(len(np.unique(warp)))


        accesses = np.array(accesses)
        print('Transactions per access for DOFs:', np.average(accesses))

        # map_coords = map_coords.T
        map_coords = map_coords.flatten()
        map_coords = map_coords.reshape((args[1]*3) // 32, 32)
        map_coords = map_coords // 4
        accesses = []
        for warp in map_coords:
            accesses.append(len(np.unique(warp)))

        accesses = np.array(accesses)
        print('Transactions per access for Coords:', np.average(accesses))

    @collective
    def __call__(self, *args):
        grid, block = self.grid_size(args[0], args[1])
        extra_global_args = self.get_args_marked_for_globals

        # self.print_data_layout_info(args)

        return self._fun.prepared_call(grid, block, *(args[1:]+extra_global_args))

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
        code, self.processed_program, self.args_to_make_global = generate_cuda_kernel(wrapper, self.extruded)

        if self._wrapper_name == configuration["cuda_jitmodule_name"]:
            if configuration["load_cuda_kernel"]:
                f = open(configuration["cuda_kernel_name"], "r")
                code = f.read()
                f.close()
            if configuration["dump_cuda_kernel"]:
                f = open(configuration["cuda_kernel_name"], "w")
                f.write(code)
                f.close()

        return code

    @collective
    def compile(self):
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        from pycuda.compiler import SourceModule

        options = ["-use_fast_math", "-Xptxas", "-dlcm=ca"]
        if True or configuration["cuda_timer_profile"]:
            options.append("-lineinfo")
        func = SourceModule(self.code_to_compile, options=options)
        self._fun = func.get_function(self._wrapper_name)
        self._fun.prepare(self.argtypes[1:]+"P"*len(self.args_to_make_global))

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
                    argtypes += (ctypes.c_void_p,)
                    seen.add(k)

        type_map = dict([(ctypes.c_void_p, "P"), (ctypes.c_int, "i")])
        argtypes = "".join(type_map[t] for t in argtypes)

        return argtypes


class ParLoop(petsc_base.ParLoop):

    printed = set()

    def __init__(self, *args, **kwargs):
        super(ParLoop, self).__init__(*args, **kwargs)
        self.kernel.cpp = True

    def prepare_arglist(self, iterset, *args):
        nbytes = 0

        arglist = iterset._kernel_args_
        for arg in args:
            arglist += arg._kernel_args_
            if arg.access is INC:
                nbytes += arg.data.nbytes * 2
            else:
                nbytes += arg.data.nbytes
        seen = set()
        for arg in args:
            maps = arg.map_tuple
            for map_ in maps:
                for k in map_._kernel_args_:
                    if k in seen:
                        continue
                    arglist += map_._kernel_args_
                    seen.add(k)
                    nbytes += map_.values.nbytes

        self.nbytes = nbytes
        wrapper_name = "wrap_" + self._kernel.name
        if wrapper_name not in ParLoop.printed:
            # print("{0}_BYTES= {1}".format("wrap_" + self._kernel.name, self.nbytes))
            ParLoop.printed.add(wrapper_name)

        return arglist

    @collective
    @timed_function("ParLoopRednEnd")
    def reduction_end(self):
        """End reductions"""
        for arg in self.global_reduction_args:
            arg.reduction_end(self.comm)
        # Finalise global increments
        for tmp, glob in self._reduced_globals.items():
            # These can safely access the _data member directly
            # because lazy evaluation has ensured that any pending
            # updates to glob happened before this par_loop started
            # and the reduction_end on the temporary global pulled
            # data back from the device if necessary.
            # In fact we can't access the properties directly because
            # that forces an infinite loop.
            with tmp.vec as v:
                glob._data += v.array_r

    @cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.iterset, *self.args,
                         iterate=self.iteration_region,
                         pass_layer_arg=self._pass_layer_arg)

    @collective
    def _compute(self, part, fun, *arglist):
        if part.size == 0:
            return

        if configuration["cuda_timer"]:
            from time import time
            start = cuda_driver.Event()
            end = cuda_driver.Event()
            py_start_time = time()
            start.record()
            # start.synchronize()
            fun(part.offset, part.offset + part.size, *arglist)
            end.record()
            end.synchronize()
            py_end_time = time()
            # print("Python time:", py_end_time-py_start_time)
            ExecTimeNoter.note(start.time_till(end)/1000)
            return

        with timed_region("ParLoop_{0}_{1}".format(self.iterset.name, self._jitmodule._wrapper_name)):
            fun(part.offset, part.offset + part.size, *arglist)


def generate_single_cell_wrapper(iterset, args, forward_args=(), kernel_name=None, wrapper_name=None, restart_counter=True):
    """Generates wrapper for a single cell. No iteration loop, but cellwise data is extracted.
    Cell is expected as an argument to the wrapper. For extruded, the numbering of the cells
    is columnwise continuous, bottom to top.

    :param iterset: The iteration set
    :param args: :class:`Arg`s
    :param forward_args: To forward unprocessed arguments to the kernel via the wrapper,
                         give an iterable of strings describing their C types.
    :param kernel_name: Kernel function name
    :param wrapper_name: Wrapper function name
    :param restart_counter: Whether to restart counter in naming variables and indices
                            in code generation.

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
    wrapper = generate(builder, wrapper_name, restart_counter)
    code = loopy.generate_code_v2(wrapper)

    return code.device_code()


def transform(kernel, callables_table, ncells_per_group=32,
        nthreads_per_cell=1,
        matvec1_parallelize_across='row', matvec2_parallelize_across='row',
        matvec1_rowtiles=1, matvec1_coltiles=1,
        matvec2_rowtiles=1, matvec2_coltiles=1,
        n_tilecomputes_to_store_after=1,
        load_coordinates_to_shared=False,
        load_input_to_shared=False,):

    # {{{ sanity checks

    #FIXME: Let's keep on writing this code and visit this later, surely
    # someone will enter failing options.

    assert n_tilecomputes_to_store_after <= matvec1_rowtiles

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

    #FIXME: Assumes the variable associated with output is 't0'. GENERALIZE THIS!
    kernel = loopy.remove_instructions(kernel, "writes:t0 and tag:gather")
    kernel = loopy.remove_instructions(kernel, "tag:quad_init")

    from loopy.transform.convert_to_reduction import convert_to_reduction
    kernel = convert_to_reduction(kernel, 'tag:quad_redn', ('form_i', ))
    kernel = convert_to_reduction(kernel, 'tag:basis_redn', ('form_ip', ))

    from loopy.loop import fuse_loop_domains
    kernel = fuse_loop_domains(kernel)
    kernel = loopy.fold_constants(kernel)

    from loopy.transform.data import remove_unused_axes_in_temporaries
    kernel = remove_unused_axes_in_temporaries(kernel)

    if load_coordinates_to_shared:
        raise NotImplementedError()

    if load_input_to_shared:
        raise NotImplementedError()

    print(kernel)
    1/0


def transpose_maps(kernel):
    print("Caution: The map representation in the kernel is transposed")
    from loopy.kernel.array import FixedStrideArrayDimTag
    from pymbolic import parse

    new_dim_tags = (FixedStrideArrayDimTag(1), FixedStrideArrayDimTag(parse('end')))
    new_args = [arg.copy(dim_tags=new_dim_tags) if arg.name[:3] == 'map' else arg for arg in kernel.args]
    kernel = kernel.copy(args=new_args)
    return kernel


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

    if kernel.name == configuration["cuda_jitmodule_name"]:
        kernel = loopy.fix_parameters(kernel, start=0)
        kernel = loopy.assume(kernel, "end > 0")

        # choose the preferred algorithm here
        kernel, args_to_make_global = transform(kernel, program.callables_table)

        # kernel = transpose_maps(kernel)
    else:
        # batch cells into groups
        # essentially, each thread computes unroll_size elements, each block computes unroll_size*block_size elements
        batch_size = configuration["cuda_block_size"]
        unroll_size = configuration["cuda_unroll_size"]

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

    if program.name == configuration["cuda_jitmodule_name"]:
        print("Generated code")
        # print(code)
        # 1/0
        pass
        # with open('current_kernel.cu', 'w') as f:
        #     # code = f.read()
        #     f.write(code)

    return code, program, args_to_make_global
