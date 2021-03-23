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

from hashlib import md5
from contextlib import contextmanager

from pyop2.datatypes import IntType, as_ctypes
from pyop2 import base
from pyop2 import compilation
from pyop2 import petsc_base
from pyop2 import sequential
from pyop2.exceptions import *  # noqa: F401
from pyop2.mpi import collective
from pyop2.profiling import timed_region
from pyop2.utils import *
from pyop2.configuration import configuration

import numpy
import pycuda.driver as cuda
from pytools import memoize_method
from pyop2.petsc_base import PETSc, AbstractPETScBackend
from pyop2.logger import ExecTimeNoter


def cudamem_from_numpy_array(ary):
    ary_on_gpu = cuda.mem_alloc(int(ary.nbytes))
    cuda.memcpy_htod(ary_on_gpu, ary)
    return ary_on_gpu


class Map(base.Map):

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        self._availability_flag = base.AVAILABLE_ON_HOST_ONLY

    @cached_property
    def _cuda_values(self):
        self._availability_flag = base.AVAILABLE_ON_BOTH
        return cudamem_from_numpy_array(self._values)

    def get_availability(self):
        return self._availability_flag

    def ensure_availability_on_device(self):
        self._cuda_values

    def ensure_availability_on_host(self):
        # Map once initialized is not over-written so always available
        # on host.
        pass

    @property
    def _kernel_args_(self):
        if cuda_backend.offloading:
            if not self.is_available_on_device():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("Map unavailable on device. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_device()

            return (self._cuda_values,)
        else:
            return super(Map, self)._kernel_args_


class ExtrudedSet(base.ExtrudedSet):
    """
    ExtrudedSet for CUDA.
    """

    def __init__(self, *args, **kwargs):
        super(ExtrudedSet, self).__init__(*args, **kwargs)
        self._availability_flag = base.AVAILABLE_ON_HOST_ONLY

    @cached_property
    def cuda_layers_array(self):
        self._availability_flag = base.AVAILABLE_ON_BOTH
        return cudamem_from_numpy_array(self.layers_array)

    def get_availability(self):
        return self._availability_flag

    def ensure_availability_on_device(self):
        self.cuda_layers_array

    def ensure_availability_on_host(self):
        # ExtrudedSet once initialized is not over-written so always available
        # on host.
        pass

    @property
    def _kernel_args_(self):
        if cuda_backend.offloading:
            if not self.is_available_on_device():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("ExtrudedSet unavailable on device. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_device()

            return (self.cuda_layers_array,)
        else:
            return super(ExtrudedSet, self)._kernel_args_


class Subset(base.Subset):
    """
    Subset for CUDA.
    """
    def __init__(self, *args, **kwargs):
        super(Subset, self).__init__(*args, **kwargs)
        self._availability_flag = base.AVAILABLE_ON_HOST_ONLY

    def get_availability(self):
        return self._availability_flag

    @cached_property
    def _cuda_indices(self):
        self._availability_flag = base.AVAILABLE_ON_BOTH
        return cudamem_from_numpy_array(self._indices)

    def ensure_availability_on_device(self):
        self._cuda_indices

    def ensure_availability_on_host(self):
        # Subset once initialized is not over-written so always available
        # on host.
        pass

    @property
    def _kernel_args_(self):
        if cuda_backend.offloading:
            if not self.is_available_on_device():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("Subset unavailable on device. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_device()

            return (self._cuda_indices,)
        else:
            return super(Subset, self)._kernel_args_


class Dat(petsc_base.Dat):
    """
    Dat for CUDA.
    """
    def __init__(self, *args, **kwargs):
        super(Dat, self).__init__(*args, **kwargs)
        # _availability_flag: only used when Dat cannot be represented as a
        # petscvec; when Dat can be represented as a petscvec the availability
        # flag is directly read from the petsc vec.
        self._availability_flag = base.AVAILABLE_ON_HOST_ONLY

    @cached_property
    def _cuda_data(self):
        """
        Only used when the Dat's data cannot be represented as a petsc Vec.
        """
        self._availability_flag = base.AVAILABLE_ON_BOTH
        return cudamem_from_numpy_array(self._data)

    @cached_property
    def _vec(self):
        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # Can't duplicate layout_vec of dataset, because we then
        # carry around extra unnecessary data.
        # But use getSizes to save an Allreduce in computing the
        # global size.
        size = self.dataset.layout_vec.getSizes()
        data = self._data[:size[0]]
        return PETSc.Vec().createCUDAWithArrays(data, size=size, bsize=self.cdim, comm=self.comm)

    def get_availability(self):
        if self.can_be_represented_as_petscvec():
            return base.DataAvailability(self._vec.getOffloadMask())
        else:
            return self._availability_flag

    def ensure_availability_on_device(self):
        if self.can_be_represented_as_petscvec():
            if not cuda_backend.offloading:
                raise NotImplementedError("PETSc limitation: can ensure availaibility"
                                          " on GPU only within an offloading context.")

            self._vec.getCUDAHandle('r')  # performs a host->device transfer if needed
        else:
            if not self.is_available_on_device():
                cuda.memcpy_htod(self._cuda_data, self._data)
            self._availability_flag = AVAILABLE_ON_BOTH

    def ensure_availability_on_host(self):
        if self.can_be_represented_as_petscvec():
            self._vec.getArray(readonly=True)  # performs a device->host transfer if needed
        else:
            if not self.is_available_on_host():
                cuda.memcpy_dtoh(self._data, self._cuda_data)
            self._availability_flag = AVAILABLE_ON_BOTH

    @contextmanager
    def vec_context(self, access):
        r"""A context manager for a :class:`PETSc.Vec` from a :class:`Dat`.

        :param access: Access descriptor: READ, WRITE, or RW."""
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()

        if cuda_backend.offloading:
            if not self.is_available_on_device():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("Dat unavailable on device. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_device()
            self._vec.bindToCPU(False)
        else:
            if not self.is_available_on_host():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("Dat unavailable on host. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_host()
            self._vec.bindToCPU(True)

        yield self._vec

        if access is not base.READ:
            self.halo_valid = False

    @property
    def _kernel_args_(self):
        if self.can_be_represented_as_petscvec():
            with self.vec as v:
                if cuda_backend.offloading:
                    v.restoreCUDAHandle(v.getCUDAHandle())  # convey to petsc that we have updated the data in the CL buffer
                    return (v.getCUDAHandle(),)
                else:
                    return (self._data.ctypes.data, )
        else:
            if cuda_backend.offloading:
                if not self.is_available_on_device():
                    if configuration['only_explicit_host_device_data_transfers']:
                        raise RuntimeError("Dat unavailable on device. Call"
                                           " ensure_availability_on_device()")

                    self.ensure_availability_on_device()

                self._availability_flag = AVAILABLE_ON_DEVICE_ONLY
                return (self._cuda_data, )
            else:
                if not self.is_available_on_host():
                    if configuration['only_explicit_host_device_data_transfers']:
                        raise RuntimeError("Dat unavailable on host. Call"
                                           " ensure_availability_on_device()")

                    self.ensure_availability_on_host()

                self._availability_flag = AVAILABLE_ON_HOST_ONLY
                return (self._data.ctypes.data, )

    @collective
    @property
    def data(self):
        if self.dataset.total_size > 0 and self._data.size == 0 and self.cdim > 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")

        self.halo_valid = False

        # {{{ ensure availability on host

        if not self.is_available_on_host():
            if configuration['only_explicit_host_device_data_transfers']:
                raise RuntimeError("Dat unavailable on host. Call"
                                   " ensure_availability_on_device()")
            self.ensure_availability_on_host()

        # }}}

        v = self._data[:self.dataset.size].view()
        v.setflags(write=True)

        # {{{ marking data on the device as invalid

        if self.can_be_represented_as_petscvec():
            self._vec.array_w  # let petsc know that we are altering data on the CPU
        else:
            self._availability_flag = AVAILABLE_ON_HOST_ONLY

        # }}}

        return v

    @property
    @collective
    def data_with_halos(self):
        self.global_to_local_begin(RW)
        self.global_to_local_end(RW)
        self.halo_valid = False

        # {{{ ensure availability on host

        if not self.is_available_on_host():
            if configuration['only_explicit_host_device_data_transfers']:
                raise RuntimeError("Dat unavailable on host. Call"
                                   " ensure_availability_on_device()")
            self.ensure_availability_on_host()

        # }}}

        v = self._data.view()
        v.setflags(write=True)

        # {{{ marking data on the device as invalid

        if self.can_be_represented_as_petsvec():
            self._vec.array_w  # let petsc know that we are altering data on the CPU
        else:
            self._availability_flag = AVAILABLE_ON_HOST_ONLY

        # }}}

        return v

    @property
    @collective
    def data_ro(self):
        if self.dataset.total_size > 0 and self._data.size == 0 and self.cdim > 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")

        # {{{ ensure availability on host

        if not self.is_available_on_host():
            if configuration['only_explicit_host_device_data_transfers']:
                raise RuntimeError("Dat unavailable on host. Call"
                                   " ensure_availability_on_device()")
            self.ensure_availability_on_host()

        # }}}

        v = self._data[:self.dataset.size].view()
        v.setflags(write=False)
        return v

    @property
    @collective
    def data_ro_with_halos(self):
        self.global_to_local_begin(READ)
        self.global_to_local_end(READ)
        v = self._data.view()

        # {{{ ensure availability on host

        if not self.is_available_on_host():
            if configuration['only_explicit_host_device_data_transfers']:
                raise RuntimeError("Dat unavailable on host. Call"
                                   " ensure_availability_on_device()")
            self.ensure_availability_on_host()

        # }}}

        v.setflags(write=False)
        return v


class Global(petsc_base.Global):
    """
    Global for CUDA.
    """

    def __init__(self, *args, **kwargs):
        super(Global, self).__init__(*args, **kwargs)
        self._availability_flag = base.AVAILABLE_ON_HOST_ONLY

    @cached_property
    def _cuda_data(self):
        self._availability_flag = base.AVAILABLE_ON_BOTH
        return cudamem_from_numpy_array(self._data)

    def get_availability(self):
        return self._availability_flag

    def ensure_availability_on_device(self):
        if not self.is_available_on_device():
            cuda.memcpy_htod(self._cuda_data, self._data)
            self._availability_flag = base.AVAILABLE_ON_BOTH

    def ensure_availability_on_host(self):
        if not self.is_available_on_host():
            cuda.memcpy_dtoh(self._data, self._cuda_data)
            self._availability_flag = base.AVAILABLE_ON_BOTH

    @property
    def _kernel_args_(self):
        if cuda_backend.offloading:
            if not self.is_available_on_device():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("Global unavailable on device. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_device()

            self._availability_flag = base.AVAILABLE_ON_DEVICE_ONLY
            return (self._cuda_data,)
        else:
            self._availability_flag = base.AVAILABLE_ON_HOST_ONLY
            return super(Global, self)._kernel_args_


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
        self._iteration_region = kwargs.get('iterate', base.ALL)
        self._pass_layer_arg = kwargs.get('pass_layer_arg', False)
        # Copy the class variables, so we don't overwrite them
        self._cppargs = dcopy(type(self)._cppargs)
        self._libraries = dcopy(type(self)._libraries)
        self._system_headers = dcopy(type(self)._system_headers)
        if not kwargs.get('delay', False):
            self.compile()
            self._initialized = True

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        key = super(JITModule, cls)._cache_key(*args, **kwargs)
        key += (configuration["gpu_strategy"],)
        if configuration["gpu_strategy"] == "scpt":
            key += (configuration["gpu_cells_per_block"],
                    configuration["gpu_threads_per_cell"])
        elif configuration["gpu_strategy"] == "user_specified_tile":
            key += (configuration["gpu_cells_per_block"],
                    configuration["gpu_threads_per_cell"],
                    configuration["gpu_op_tile_descriptions"],
                    configuration["gpu_quad_rowtile_lengths"],
                    configuration["gpu_input_to_shared"],
                    configuration["gpu_quad_weights_to_shared"],
                    configuration["gpu_mats_to_shared"],
                    configuration["gpu_tiled_prefetch_of_input"],
                    configuration["gpu_tiled_prefetch_of_quad_weights"],)
        elif configuration["gpu_strategy"] == "auto_tile":
            key += (configuration["gpu_planner_kernel_evals"],)
            assert isinstance(args[1], base.Set)
            problem_size = args[1].size
            # FIXME: is this a good heuristic?
            # perform experiments to verify it.
            # Also this number should not exceed certain number i.e. when the
            # device would be saturated.
            key += (min(int(numpy.log2(problem_size)), 18),)
        elif configuration["gpu_strategy"] == "dq_transform":
            key += (configuration["gpu_cells_per_block"],
                    configuration["gpu_threads_per_cell"])
        else:
            raise NotImplementedError('For strategy: {}'.format(
                configuration["gpu_strategy"]))
        return key

    @memoize_method
    def grid_size(self, start, end):
        with open(self.config_file_path, 'r') as f:
            glens_llens = f.read()

        _, glens, llens = glens_llens.split('\n')

        from pymbolic import parse, evaluate
        glens = parse(glens)
        llens = parse(llens)

        parameters = {'start': start, 'end': end}

        grid = tuple(int(evaluate(glens[i], parameters)) if i < len(glens) else 1
                     for i in range(2))
        block = tuple(int(evaluate(llens[i], parameters)) if i < len(llens) else 1
                      for i in range(3))

        return grid, block

    @cached_property
    def get_args_marked_for_globals(self):
        args_to_make_global = []
        for i in range(len(self._fun.arg_format)-len(self.argtypes)):
            args_to_make_global.append(numpy.load(self.ith_added_global_arg_i(i)))

        const_args_as_globals = tuple(cuda.mem_alloc(arg.nbytes)
                                      for arg in args_to_make_global)
        for arg_gpu, arg in zip(const_args_as_globals, args_to_make_global):
            cuda.memcpy_htod(arg_gpu, arg)

        evt = cuda.Event()
        evt.record()
        evt.synchronize()

        return const_args_as_globals

    @cached_property
    def config_file_path(self):
        cachedir = configuration['cache_dir']
        return os.path.join(cachedir, '{}_num_args_to_load_glens_llens'.format(self.get_encoded_cache_key))

    @memoize_method
    def ith_added_global_arg_i(self, i):
        cachedir = configuration['cache_dir']
        return os.path.join(cachedir, '{}_dat_{}.npy'.format(self.get_encoded_cache_key, i))

    @collective
    def __call__(self, *args):
        if self._initialized:
            grid, block = self.grid_size(args[0], args[1])
            extra_global_args = self.get_args_marked_for_globals
        else:
            self.args = args[:]
            self.compile()
            self._initialized = True
            return self.__call__(*args)

        return self._fun.prepared_call(grid, block, *(args+extra_global_args))

    @cached_property
    def _wrapper_name(self):
        return 'wrap_%s' % self._kernel.name

    @cached_property
    def num_args_to_make_global(self):
        with open(self.config_file_path, 'r') as f:
            return int(f.readline().strip())

    @cached_property
    def get_encoded_cache_key(self):
        a = md5(str(self.cache_key[1:]).encode()).hexdigest()
        return a

    @cached_property
    def code_to_compile(self):
        assert self.args is not None
        from pyop2.codegen.builder import WrapperBuilder
        from pyop2.codegen.rep2loopy import generate
        from pyop2.gpu.generate import generate_gpu_kernel

        builder = WrapperBuilder(kernel=self._kernel,
                                 iterset=self._iterset,
                                 iteration_region=self._iteration_region,
                                 pass_layer_to_kernel=self._pass_layer_arg)

        for arg in self._args:
            builder.add_argument(arg)

        wrapper = generate(builder, include_petsc=False, include_complex=False)

        code, processed_program, args_to_make_global = generate_gpu_kernel(wrapper, self.args, self.argshapes, 'cuda')

        for i, arg_to_make_global in enumerate(args_to_make_global):
            numpy.save(self.ith_added_global_arg_i(i),
                       arg_to_make_global)

        with open(self.config_file_path, 'w') as f:
            glens, llens = processed_program.get_grid_size_upper_bounds_as_exprs()
            f.write(str(len(args_to_make_global)))
            f.write('\n')
            f.write('('+','.join(str(glen) for glen in glens)+',)')
            f.write('\n')
            f.write('('+','.join(str(llen) for llen in llens)+',)')

        return code

    @collective
    def compile(self):

        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        compiler = "nvcc"
        extension = "cu"
        self._fun = compilation.load(self,
                                     extension,
                                     self._wrapper_name,
                                     cppargs=[],
                                     ldargs=[],
                                     compiler=compiler,
                                     comm=self.comm)

        type_map = dict([(ctypes.c_void_p, "P"), (ctypes.c_int, "i")])
        argtypes = "".join(type_map[t] for t in self.argtypes)

        self._fun.prepare(argtypes+"P"*self.num_args_to_make_global)

        # Blow away everything we don't need any more
        del self.args
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

        return argtypes

    @cached_property
    def argshapes(self):
        argshapes = ((), ())
        if self._iterset._argtypes_:
            # FIXME: Do not put in a bogus value
            argshapes += ((), )

        for arg in self._args:
            argshapes += (arg.data.shape, )
        seen = set()
        for arg in self._args:
            maps = arg.map_tuple
            for map_ in maps:
                for k, t in zip(map_._kernel_args_, map_._argtypes_):
                    if k in seen:
                        continue
                    argshapes += (map_.shape, )
                    seen.add(k)

        return argshapes


class ParLoop(petsc_base.ParLoop):

    printed = set()

    def prepare_arglist(self, iterset, *args):
        nbytes = 0

        arglist = iterset._kernel_args_
        for arg in args:
            arglist += arg._kernel_args_
            if arg.access is base.INC:
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

        return arglist

    @collective
    def reduction_end(self):
        """End reductions"""
        if not self._has_reduction:
            return
        with self._reduction_event_end:
            for arg in self.global_reduction_args:
                arg.reduction_end(self.comm)
            # Finalise global increments
            for tmp, glob in self._reduced_globals.items():
                # copy results to the host
                tmp.ensure_availability_on_host()
                glob.ensure_availability_on_host()
                glob._data += tmp._data
                glob._availability_flag = base.AVAILABLE_ON_HOST_ONLY

    @cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.iterset, *self.args,
                         iterate=self.iteration_region,
                         pass_layer_arg=self._pass_layer_arg,
                         delay=True)

    @collective
    def _compute(self, part, fun, *arglist):
        if part.size == 0:
            return

        # how about over here we decide what should the strategy be..

        if configuration["gpu_timer"] and self._kernel.name == "form0_cell_integral_otherwise":
            start = cuda.Event()
            end = cuda.Event()
            start.record()
            start.synchronize()
            fun(part.offset, part.offset + part.size, *arglist)
            end.record()
            end.synchronize()
            ExecTimeNoter.note(start.time_till(end)/1000)
            # print("{0}_TIME= {1}".format(self._jitmodule._wrapper_name, start.time_till(end)/1000))
            return

        with timed_region("ParLoop_{0}_{1}".format(self.iterset.name, self._jitmodule._wrapper_name)):
            fun(part.offset, part.offset + part.size, *arglist)


class CUDABackend(AbstractPETScBackend):
    ParLoop_offloading = ParLoop
    ParLoop_no_offloading = sequential.ParLoop
    ParLoop = sequential.ParLoop
    Set = base.Set
    ExtrudedSet = ExtrudedSet
    MixedSet = base.MixedSet
    Subset = Subset
    DataSet = petsc_base.DataSet
    MixedDataSet = petsc_base.MixedDataSet
    Map = Map
    MixedMap = base.MixedMap
    Dat = Dat
    MixedDat = petsc_base.MixedDat
    DatView = base.DatView
    Mat = petsc_base.Mat
    Global = Global
    GlobalDataSet = petsc_base.GlobalDataSet
    PETScVecType = 'cuda'

    def __init__(self):
        self.offloading = False

    def turn_on_offloading(self):
        self.offloading = True
        self.ParLoop = self.ParLoop_offloading

    def turn_off_offloading(self):
        self.offloading = False
        self.ParLoop = self.ParLoop_no_offloading


cuda_backend = CUDABackend()
