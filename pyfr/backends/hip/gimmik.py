from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import time
from weakref import finalize

from gimmik import HIPMatMul

from pyfr.backends.base import NotSuitableError
from pyfr.backends.hip.compiler import HIPRTC
from pyfr.backends.hip.provider import HIPKernel, HIPKernelProvider
from pyfr.cache import ObjectCache
from pyfr.util import digest


_worker_cache = None
_worker_hiprtc = None


def _kernel_cache_entry(src, gcn_arch, compiler_version):
    src = f'extern "C"\n{{\n{src}\n}}'
    flags = [f'--gpu-architecture={gcn_arch}', '-munsafe-fp-atomics']
    ckey = digest(compiler_version, 'kernel', src, flags)

    return ckey, src, flags


def _warm_kernel_cache(src, gcn_arch, compiler_version):
    global _worker_cache, _worker_hiprtc

    if _worker_cache is None:
        _worker_cache = ObjectCache('hip')
    if _worker_hiprtc is None:
        _worker_hiprtc = HIPRTC()

    ckey, src, flags = _kernel_cache_entry(src, gcn_arch, compiler_version)

    if _worker_cache.get_bytes(ckey) is None:
        code = _worker_hiprtc.compile('kernel', src, flags)
        _worker_cache.set_with_bytes(ckey, code)


def _warmup_worker(_):
    global _worker_cache, _worker_hiprtc

    if _worker_cache is None:
        _worker_cache = ObjectCache('hip')
    if _worker_hiprtc is None:
        _worker_hiprtc = HIPRTC()

    return True


def _render_compile_candidate(args):
    (idx, arr, dtype, alpha, beta, aligne, kname, gcn_arch, warp_size,
     force_variant, compiler_version) = args

    mm = HIPMatMul(alpha*arr, beta=beta, aligne=aligne)
    src, meta = mm.render_candidate(idx, dtype, kname=kname,
                                    gcn_arch=gcn_arch,
                                    warp_size=warp_size)

    tplname = meta.get('tplname', '')
    variant = meta.get('desc', tplname)

    if force_variant and force_variant != variant:
        return idx, None

    _warm_kernel_cache(src, gcn_arch, compiler_version)

    return idx, (src, meta, variant)


class HIPGiMMiKKernels(HIPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        # Maximum number of kernels to consider
        self.nkerns = backend.cfg.getint('backend-hip', 'gimmik-nkerns', 20)

        # Number of benchmarking runs
        self.nbench = backend.cfg.getint('backend-hip', 'gimmik-nbench', 5)

        # Parallel source rendering and compiler-cache warmup
        self.parallel_compile = backend.cfg.getbool(
            'backend-hip', 'gimmik-parallel-compile', False
        )
        self.parallel_compile_workers = backend.cfg.getint(
            'backend-hip', 'gimmik-parallel-compile-workers', self.nkerns
        )
        self.parallel_compile_pool = None

        if self.parallel_compile:
            workers = max(1, self.parallel_compile_workers)
            mpctx = mp.get_context('spawn')
            self.parallel_compile_pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=mpctx
            )

            # Spawn workers eagerly so the first autotune does not pay for
            # process creation and HIPRTC initialization.
            list(self.parallel_compile_pool.map(_warmup_worker,
                                                range(workers)))

        # Kernel cache
        self._mul_kerns = {}

    def __del__(self):
        try:
            if self.parallel_compile_pool is not None:
                self.parallel_compile_pool.shutdown()
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

            times['gimmik'] = total_s

            details = getattr(self.backend, '_kernel_create_details', None)
            if details is not None:
                for detail in reversed(details):
                    if (detail.get('provider') == 'gimmik' and
                        'total_s' not in detail):
                        detail['total_s'] = total_s
                        break

    def _mul(self, a, b, out, alpha=1.0, beta=0.0):
        detail = {
            'provider': 'gimmik',
            'cache_hit': '',
            'candidate_count': 0,
            'selected': '',
            'output_save_s': 0.0,
            'autotuning_s': 0.0,
            'generate_s': 0.0,
            'parallel_compile_wall_s': 0.0,
            'build_kernel_s': 0.0,
            'benchmark_s': 0.0,
            'restore_s': 0.0,
        }
        details = getattr(self.backend, '_kernel_create_details', None)
        if details is not None:
            details.append(detail)

        force_variant = os.environ.get('PYFR_GIMMIK_FORCE_VARIANT')

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('GiMMiK requires a constant a matrix')

        # Fetch the matrix
        arr = a.get()

        # Dimensions
        n = b.ncol
        ldb, ldc = b.leaddim, out.leaddim

        # Alignment
        if 'align' in b.tags and 'align' in out.tags:
            aligne = self.backend.alignb // b.itemsize
        else:
            aligne = None

        # Cache key
        ckey = (a.mid, alpha, beta, aligne, force_variant)

        # Check the kernel cache
        try:
            kern, block, grid_y, ncolsv, dt, variant = self._mul_kerns[ckey]
            detail['cache_hit'] = True
            detail['selected'] = variant
        except KeyError:
            detail['cache_hit'] = False
            ifac = self.backend.autotune_ifac
            kname = f'gimmik_mm_{arr.shape[0]}x{arr.shape[1]}'
            kdata = None
            best_kern = None

            # Save a copy of the contents of the output matrix
            tout = time.perf_counter()
            out_np = getattr(out, 'parent', out).get()
            detail['output_save_s'] += time.perf_counter() - tout

            def benchmark_candidate(src, meta, variant):
                tbuild = time.perf_counter()
                kern = self._build_kernel(kname, src, 'iPiPi')
                detail['build_kernel_s'] += time.perf_counter() - tbuild
                detail['candidate_count'] += 1

                grid_y, ncolsv = meta['grid_y'], meta['ncolsv']
                grid = (-(-n // ncolsv), grid_y, 1)
                params = kern.make_params(grid, meta['block'])
                params.set_args(n, b, ldb, out, ldc)

                # Obtain the runtime
                tbench = time.perf_counter()
                dt = self._benchmark(
                    lambda stream: kern.exec_async(stream, params),
                    nbench=self.nbench
                )
                detail['benchmark_s'] += time.perf_counter() - tbench

                tplname = meta.get('tplname', '')
                print(
                    f'GiMMiK autotune: tplname={tplname} variant={variant} '
                    f'block={meta["block"]} shared={meta.get("shared", 0)} '
                    f'ncolsv={ncolsv} time={dt:.6e}s regs={kern.nreg} '
                    f'local_mem={kern.local_mem}',
                    flush=True
                )

                kdata = {
                    'runtime': dt,
                    'registers': kern.nreg,
                    'local_mem': kern.local_mem
                }

                return kern, meta['block'], grid_y, ncolsv, dt, variant, kdata

            def update_best(bench):
                nonlocal best_kern

                if best_kern is None or bench[4] < ifac*best_kern[4]:
                    best_kern = bench[:-1]

            mm = HIPMatMul(alpha*arr, beta=beta, aligne=aligne)

            tautotune = time.perf_counter()
            try:
                if self.parallel_compile:
                    gcn_arch = self.backend.props['gcn_arch_name']
                    warp_size = self.backend.props['warp_size']
                    count = min(
                        self.nkerns,
                        mm.candidate_count(a.dtype, gcn_arch=gcn_arch,
                                           warp_size=warp_size)
                    )
                    args = [
                        (
                            idx, arr, a.dtype, alpha, beta, aligne, kname,
                            gcn_arch, warp_size, force_variant,
                            self.backend.compiler.version
                        )
                        for idx in range(count)
                    ]

                    twall = time.perf_counter()
                    candidates = list(
                        self.parallel_compile_pool.map(
                            _render_compile_candidate, args
                        )
                    ) if args else []
                    detail['parallel_compile_wall_s'] = (
                        time.perf_counter() - twall
                    )

                    for idx, candidate in candidates:
                        if candidate is None:
                            continue

                        src, meta, variant = candidate
                        update_best(benchmark_candidate(src, meta, variant))
                else:
                    kgen = mm.kernels(
                        a.dtype, kname=kname,
                        gcn_arch=self.backend.props['gcn_arch_name'],
                        warp_size=self.backend.props['warp_size']
                    )

                    # Benchmark the sequence of kernels generated by GiMMiK
                    try:
                        for i in range(self.nkerns):
                            tgenerate = time.perf_counter()
                            src, meta = kgen.send(kdata)
                            detail['generate_s'] += (
                                time.perf_counter() - tgenerate
                            )

                            tplname = meta.get('tplname', '')
                            variant = meta.get('desc', tplname)

                            if force_variant and force_variant != variant:
                                kdata = None
                                continue

                            bench = benchmark_candidate(src, meta, variant)
                            update_best(bench)
                            kdata = bench[-1]

                            if force_variant:
                                break
                    except StopIteration:
                        pass
            finally:
                detail['autotuning_s'] += time.perf_counter() - tautotune

            # Restore the output matrix
            trestore = time.perf_counter()
            getattr(out, 'parent', out).set(out_np)
            detail['restore_s'] += time.perf_counter() - trestore

            if best_kern is None:
                raise NotSuitableError(
                    f'GiMMiK variant {force_variant!r} is not available'
                )

            print(
                f'GiMMiK autotune selected: variant={best_kern[5]} '
                f'block={best_kern[1]} grid_y={best_kern[2]} '
                f'ncolsv={best_kern[3]} time={best_kern[4]:.6e}s',
                flush=True
            )

            # Update the cache
            self._mul_kerns[ckey] = (
                kern, block, grid_y, ncolsv, dt, variant
            ) = best_kern
            finalize(a, lambda: self._mul_kerns.pop(ckey))
            detail['selected'] = variant

        # Set the parameters
        grid = (-(-n // ncolsv), grid_y, 1)
        params = kern.make_params(grid, block)
        params.set_args(n, b, ldb, out, ldc)

        class MulKernel(HIPKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_kernel(params, deps)

            def run(self, stream):
                kern.exec_async(stream, params)

        mk = MulKernel(mats=[a, b, out], dt=dt)
        mk.kernel_variant = variant

        return mk
