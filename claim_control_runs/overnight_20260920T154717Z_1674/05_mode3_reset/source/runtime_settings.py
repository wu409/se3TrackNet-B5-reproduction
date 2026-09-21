"""Configurable CPU/I/O budgets and paired SAM2 defaults; no heavy imports."""
import os
import sys

SAM2_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_DEFAULT_CHECKPOINT = "sam2.1_hiera_small.pt"
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
_LIMITER = None


def positive(value, name):
    if not str(value).isdigit() or int(value) < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def configure_environment(env=None):
    env = os.environ if env is None else env
    threads = positive(env.get("B5_NUM_THREADS", env.get("OMP_NUM_THREADS", "16")), "B5_NUM_THREADS")
    env["B5_NUM_THREADS"] = str(threads)
    for key in THREAD_KEYS:
        env[key] = str(threads)
    env["B5_IO_WORKERS"] = str(positive(env.get("B5_IO_WORKERS", "8"), "B5_IO_WORKERS"))
    return threads


def execution_config():
    threads = configure_environment()
    return dict(version="cpu_io_budget_v1", cpu_threads=threads,
                io_workers=int(os.environ["B5_IO_WORKERS"]))


def apply_frozen(config):
    if not isinstance(config, dict) or config.get("version") != "cpu_io_budget_v1":
        raise ValueError("Missing/unsupported frozen CPU/I/O budget")
    os.environ["B5_NUM_THREADS"] = str(positive(config['cpu_threads'], 'cpu_threads'))
    os.environ["B5_IO_WORKERS"] = str(positive(config['io_workers'], 'io_workers'))
    configure_libraries()


def configure_libraries():
    """Call before inference; update already-loaded libraries, never import CUDA."""
    global _LIMITER
    threads = configure_environment()
    torch = sys.modules.get('torch')
    if torch is not None and hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != threads:
            # PyTorch permits this only before parallel work has started.
            torch.set_num_interop_threads(threads)
    cv2 = sys.modules.get('cv2')
    if cv2 is not None and hasattr(cv2, 'setNumThreads'):
        cv2.setNumThreads(threads)
    numexpr = sys.modules.get('numexpr')
    if numexpr is not None:
        numexpr.set_num_threads(threads)
    try:
        from threadpoolctl import threadpool_limits
        _LIMITER = threadpool_limits(limits=threads)
    except ImportError:
        pass  # environment settings still apply before NumPy/BLAS import
    return threads


def validate_sam_pair(config, checkpoint):
    """Reject common Small/Large config-checkpoint mismatches before CUDA load."""
    pairs = {'sam2.1_hiera_tiny.pt': 'sam2.1_hiera_t.yaml',
             'sam2.1_hiera_small.pt': 'sam2.1_hiera_s.yaml',
             'sam2.1_hiera_base_plus.pt': 'sam2.1_hiera_b+.yaml',
             'sam2.1_hiera_large.pt': 'sam2.1_hiera_l.yaml'}
    basename = os.path.basename(str(checkpoint))
    if basename in pairs and os.path.basename(str(config)) != pairs[basename]:
        raise ValueError('SAM2 config/checkpoint mismatch: %s vs %s' % (config, checkpoint))


configure_environment()
