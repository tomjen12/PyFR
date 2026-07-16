from ctypes import (POINTER, byref, c_int, c_int64, c_double, c_float,
                    c_uint32, c_void_p)
import os
import time

import numpy as np

from pyfr.backends.hip.provider import HIPKernel, HIPKernelProvider
from pyfr.ctypesutil import LibWrapper


# Possible RocBLAS exception types
class RocBLASError(Exception): pass
class RocBLASInvalidHandle(RocBLASError): pass
class RocBLASNotImplemented(RocBLASError): pass
class RocBLASInvalidPointer(RocBLASError): pass
class RocBLASInvalidSize(RocBLASError): pass
class RocBLASInternalError(RocBLASError): pass
class RocBLASInvalidValue(RocBLASError): pass


class RocBLASWrappers(LibWrapper):
    _libname = 'rocblas'

    # Error codes
    _statuses = {
        1: RocBLASInvalidHandle,
        2: RocBLASNotImplemented,
        3: RocBLASInvalidPointer,
        4: RocBLASInvalidSize,
        6: RocBLASInternalError,
        11: RocBLASInvalidValue
    }

    # Constants
    OPERATION_NONE = 111
    OPERATION_TRANSPOSE = 112
    DATATYPE_F32_R = 151
    DATATYPE_F64_R = 152
    GEMM_ALGO_SOLUTION_INDEX = 1

    # Functions
    _functions = [
        (c_int, 'rocblas_create_handle', POINTER(c_void_p)),
        (c_int, 'rocblas_destroy_handle', c_void_p),
        (c_int, 'rocblas_set_stream', c_void_p, c_void_p),
        (c_int, 'rocblas_dgemm_64', c_void_p, c_int, c_int, c_int64, c_int64,
         c_int64, POINTER(c_double), c_void_p, c_int64, c_void_p, c_int64,
         POINTER(c_double), c_void_p, c_int64),
        (c_int, 'rocblas_sgemm_64', c_void_p, c_int, c_int, c_int64, c_int64,
         c_int64, POINTER(c_float), c_void_p, c_int64, c_void_p, c_int64,
         POINTER(c_float), c_void_p, c_int64),
        (c_int, 'rocblas_gemm_ex_get_solutions', c_void_p, c_int, c_int, c_int,
         c_int, c_int, c_void_p, c_void_p, c_int, c_int, c_void_p,
         c_int, c_int, c_void_p, c_void_p, c_int, c_int, c_void_p,
         c_int, c_int, c_int, c_int, c_uint32, POINTER(c_int), POINTER(c_int)),
        (c_int, 'rocblas_gemm_ex', c_void_p, c_int, c_int, c_int,
         c_int, c_int, c_void_p, c_void_p, c_int, c_int, c_void_p,
         c_int, c_int, c_void_p, c_void_p, c_int, c_int, c_void_p,
         c_int, c_int, c_int, c_int, c_int, c_uint32)
    ]


class HIPRocBLASKernels(HIPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self._cstream = backend.hip.create_stream()

        # Ensure memory can be allocated in captured streams
        os.environ['ROCBLAS_STREAM_ORDER_ALLOC'] = '1'

        # Load and wrap rocBLAS
        self._wrappers = RocBLASWrappers()

        # Init
        self._handle = c_void_p()
        self._wrappers.rocblas_create_handle(self._handle)

        # GEMM cache
        self._mul_cache = {}

        # Maximum number of solution indices to try
        self.nkerns = backend.cfg.getint('backend-hip', 'rocblas-nkerns', 2048)

    def __del__(self):
        try:
            if self._handle:
                self._wrappers.rocblas_set_stream(self._handle, self._cstream)
                self._wrappers.rocblas_destroy_handle(self._handle)
        except AttributeError:
            pass

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        tstart = time.perf_counter()
        try:
            return self._mul(a, b, out, alpha, beta)
        finally:
            total_s = time.perf_counter() - tstart
            times = getattr(self.backend, '_kernel_create_times', None)
            if times is None:
                self.backend._kernel_create_times = times = {}

            times['rocblas'] = total_s

            details = getattr(self.backend, '_kernel_create_details', None)
            if details is not None:
                for detail in reversed(details):
                    if (detail.get('provider') == 'rocblas' and
                        'total_s' not in detail):
                        detail['total_s'] = total_s
                        break

    def _mul(self, a, b, out, alpha=1.0, beta=0.0):
        detail = {
            'provider': 'rocblas',
            'cache_hit': '',
            'candidate_count': 0,
            'selected': '',
            'output_save_s': 0.0,
            'autotuning_s': 0.0,
            'restore_s': 0.0
        }
        details = getattr(self.backend, '_kernel_create_details', None)
        if details is not None:
            details.append(detail)

        h, w = self._handle, self._wrappers
        cstream = self._cstream
        force_algo = os.environ.get('PYFR_ROCBLAS_FORCE_ALGO')

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # RocBLAS expects inputs to be column-major (or Fortran order in
        # NumPy parlance).  However as C = A*B => C^T = (A*B)^T
        # = (B^T)*(A^T) with a little trickery we can multiply our
        # row-major matrices directly.
        m, n, k = b.ncol, a.nrow, a.ncol
        A, B, C = b, a, out

        # Cache key
        ckey = (A.dtype, alpha, beta, m, n, k, A.leaddim, B.leaddim,
                C.leaddim, force_algo)

        # Do not transpose either A or B
        opA = opB = w.OPERATION_NONE

        # α and β factors for C = α*(A*B) + β*C
        if a.dtype == np.float64:
            rtype = w.DATATYPE_F64_R
            gemm_fn = w.rocblas_dgemm_64
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
        else:
            rtype = w.DATATYPE_F32_R
            gemm_fn = w.rocblas_sgemm_64
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)

        def gemm(stream, algo):
            w.rocblas_set_stream(h, stream)
            if algo is None:
                gemm_fn(h, opA, opB, m, n, k, byref(alpha_ct), A, A.leaddim, B,
                        B.leaddim, byref(beta_ct), C, C.leaddim)
            else:
                w.rocblas_gemm_ex(
                    h, opA, opB, m, n, k, byref(alpha_ct), A, rtype, A.leaddim,
                    B, rtype, B.leaddim, byref(beta_ct), C, rtype, C.leaddim,
                    C, rtype, C.leaddim, rtype, w.GEMM_ALGO_SOLUTION_INDEX,
                    algo, 0
                )

        try:
            algo, dt = self._mul_cache[ckey]
            detail['cache_hit'] = True
            detail['selected'] = f'rocblas-algo-{algo if algo is not None else "default"}'
        except KeyError:
            detail['cache_hit'] = False
            ifac = self.backend.autotune_ifac

            def get_solution_indices():
                def get_solutions(sidx):
                    size_ct = c_int(len(sidx) if sidx is not None else 0)
                    w.rocblas_gemm_ex_get_solutions(
                        h, opA, opB, m, n, k, byref(alpha_ct), A, rtype,
                        A.leaddim, B, rtype, B.leaddim, byref(beta_ct), C,
                        rtype, C.leaddim, C, rtype, C.leaddim, rtype,
                        w.GEMM_ALGO_SOLUTION_INDEX, 0, sidx, byref(size_ct)
                    )
                    return size_ct.value

                sidx = (c_int * min(get_solutions(None), self.nkerns - 1))()
                get_solutions(sidx)

                return list(sidx)

            if force_algo is not None:
                if force_algo.lower() in ('default', 'none'):
                    candidates = [None]
                else:
                    try:
                        falgo = int(force_algo)
                    except ValueError:
                        raise ValueError(
                            'PYFR_ROCBLAS_FORCE_ALGO must be an integer '
                            "solution index, 'default', or 'none'"
                        )

                    # Query solutions in this process before forcing an index.
                    # Some rocBLAS solution IDs are not useful unless the
                    # solution table has been materialized for this GEMM.
                    if all(sz <= 2**31 - 1 for sz in ckey[3:-1]):
                        solutions = get_solution_indices()
                        if falgo not in solutions:
                            print(
                                f'rocBLAS force warning: algo={falgo} is not '
                                f'in the current solution list',
                                flush=True
                            )

                    candidates = [falgo]
            # Check if sizes fit in 32-bit for gemm_ex autotuning
            elif all(sz <= 2**31 - 1 for sz in ckey[3:-1]):
                candidates = [None, *get_solution_indices()]
            else:
                candidates = [None]

            # Save a copy of the contents of the output matrix
            tout = time.perf_counter()
            out_np = getattr(out, 'parent', out).get()
            detail['output_save_s'] += time.perf_counter() - tout

            tautotune = time.perf_counter()
            best_kern = None
            for algo in candidates:
                try:
                    dt = self._benchmark(lambda s: gemm(s, algo))
                    detail['candidate_count'] += 1
                    print(
                        f'rocBLAS autotune: algo='
                        f'{algo if algo is not None else "default"} '
                        f'time={dt:.6e}s',
                        flush=True
                    )
                    if best_kern is None or dt < ifac*best_kern[-1]:
                        best_kern = algo, dt
                # In the case of invalid values raised by rocblas
                except RocBLASError as exc:
                    print(
                        f'rocBLAS autotune: algo='
                        f'{algo if algo is not None else "default"} '
                        f'failed ({type(exc).__name__})',
                        flush=True
                    )

                    if force_algo is not None:
                        raise

            if best_kern is None:
                raise RocBLASInternalError
            detail['autotuning_s'] += time.perf_counter() - tautotune

            # Restore the output matrix
            trestore = time.perf_counter()
            getattr(out, 'parent', out).set(out_np)
            detail['restore_s'] += time.perf_counter() - trestore

            # Update the cache
            self._mul_cache[ckey] = algo, dt = best_kern

            print(
                f'rocBLAS autotune selected: algo='
                f'{algo if algo is not None else "default"} '
                f'time={dt:.6e}s',
                flush=True
            )
            detail['selected'] = f'rocblas-algo-{algo if algo is not None else "default"}'

        class MulKernel(HIPKernel):
            def add_to_graph(self, graph, deps):
                # Capture the execution of rocBLAS to obtain a graph
                cstream.begin_capture()
                self.run(cstream)
                gnode = cstream.end_capture()

                # Embed this graph in our main graph
                return graph.graph.add_graph(gnode, deps)

            def run(self, stream):
                gemm(stream, algo)

        mk = MulKernel(mats=[a, b, out], dt=dt)
        mk.kernel_variant = f'rocblas-algo-{algo if algo is not None else "default"}'

        return mk
