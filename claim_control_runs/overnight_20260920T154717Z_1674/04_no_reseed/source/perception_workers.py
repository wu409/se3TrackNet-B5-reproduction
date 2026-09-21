"""CUDA workers for perception_runtime. Launch with the component's own Python.

Stdout is reserved for request-correlated JSON; all third-party output is logged.
No changes are made to SAM2 or FoundationPose installations.
"""
import runtime_settings
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np


def configure_component_imports(kind, repo):
    """Isolate component imports from the SE3 launcher/source directory."""
    repo = Path(repo).resolve()
    launcher = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != launcher]
    os.chdir(str(repo))
    paths = [str(repo)]
    if kind == "foundationpose":
        paths.append(str(repo / "learning" / "models"))
    sys.path[:0] = paths
    if kind == "foundationpose":
        import inspect
        import network_modules
        expected = (repo / "learning" / "models" / "network_modules.py").resolve()
        actual = Path(inspect.getfile(network_modules.ConvBNReLU)).resolve()
        if actual != expected:
            raise RuntimeError("FoundationPose network module collision: " + str(actual))
        if "norm_layer" not in inspect.signature(network_modules.ConvBNReLU.__init__).parameters:
            raise RuntimeError("FoundationPose ConvBNReLU lacks norm_layer: " + str(actual))


class LazyFrames:
    """Official synchronous loader arithmetic, but only one decoded CPU image.

    No frame beyond allowed_index can be decoded. SAM2's CPU temporal state is
    left intact: unvalidated memory pruning could change predictions.
    """
    def __init__(self, paths, image_size, height, width):
        self.paths, self.image_size = paths, image_size
        self.height, self.width = height, width
        self.allowed_index = 0
        self.cached_index, self.cached = None, None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import torch
        from sam2.utils.misc import _load_img_as_tensor
        if not isinstance(index, int) or not 0 <= index < len(self.paths) or index > self.allowed_index:
            raise RuntimeError("SAM2 attempted a non-causal/invalid frame read: " + str(index))
        if index != self.cached_index:
            image, height, width = _load_img_as_tensor(self.paths[index], self.image_size)
            if (height, width) != (self.height, self.width):
                raise ValueError("SAM2 frame dimensions changed within the episode")
            # The legacy synchronous loader assigns float64 PIL output into a
            # float32 batch BEFORE normalization. Preserve that rounding order.
            image = image.to(dtype=torch.float32)
            image -= torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
            image /= torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
            self.cached_index, self.cached = index, image
        return self.cached


class SamWorker:
    def __init__(self, cfg):
        import torch
        from PIL import Image
        from sam2.build_sam import build_sam2_video_predictor
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        self.cfg = cfg
        self.scratch = Path(cfg["scratch"])
        self.predictor = build_sam2_video_predictor(cfg["model_cfg"], cfg["checkpoint"],
            device="cuda", mode="eval", apply_postprocessing=True)
        # Initialize the official state using ONLY frame zero. Then substitute
        # a causal lazy sequence; no full video tensor or future decoding.
        first_dir = self.scratch / "sam2_first_frame"
        first_dir.mkdir()
        import shutil
        destination = first_dir / "00000000.jpg"
        try:
            os.symlink(cfg["rgb_paths"][0], str(destination))
        except OSError:
            shutil.copy2(cfg["rgb_paths"][0], str(destination))
        self.state = self.predictor.init_state(video_path=str(first_dir), offload_video_to_cpu=True,
            offload_state_to_cpu=True, async_loading_frames=False)
        for key in ("images", "num_frames", "video_height", "video_width"):
            if key not in self.state:
                raise RuntimeError("Incompatible SAM2 state API: missing " + key)
        self.frames = LazyFrames(cfg["rgb_paths"], self.predictor.image_size,
                                 self.state["video_height"], self.state["video_width"])
        self.state["images"] = self.frames
        self.state["num_frames"] = len(self.frames)
        mask_path = Path(cfg["initial_mask"])
        if mask_path.suffix.lower() == ".npy":
            mask = np.load(mask_path, allow_pickle=False)
        else:
            with Image.open(mask_path) as image:
                mask = np.asarray(image)
        if mask.ndim == 3:
            mask = np.any(mask > 0, axis=2)
        mask = mask > 0
        if mask.shape != (self.frames.height, self.frames.width) or int(mask.sum()) < 20:
            raise ValueError("Invalid official first-frame mask")
        self.predictor.reset_state(self.state)
        self.predictor.add_new_mask(self.state, frame_idx=0, obj_id=1, mask=mask)
        self.generator = self.predictor.propagate_in_video(self.state, start_frame_idx=0,
            max_frame_num_to_track=len(self.frames), reverse=False)
        self.index = -1

    def handle(self, request):
        index = int(request["frame_index"])
        if (request["operation"] != "advance" or index != self.index + 1
                or request["rgb_path"] != self.cfg["rgb_paths"][index]):
            raise ValueError("SAM2 request order/path mismatch")
        self.frames.allowed_index = index
        frame_index, object_ids, logits = next(self.generator)
        if int(frame_index) != index:
            raise RuntimeError("SAM2 propagation yielded wrong frame")
        ids = [int(x) for x in object_ids]
        if 1 not in ids:
            raise RuntimeError("SAM2 lost the requested object ID")
        mask_logits = logits[ids.index(1)]
        if mask_logits.ndim == 3:
            mask_logits = mask_logits[0]
        mask = mask_logits.detach().float().cpu().numpy() > 0.0
        if mask.shape != (self.frames.height, self.frames.width):
            raise RuntimeError("SAM2 returned a wrong-sized mask")
        np.save(self.scratch / "current_mask.npy", mask)
        self.index = index
        return dict(frame_index=index, mask_pixels=int(mask.sum()), propagated_frames=index + 1)

    def close(self):
        self.generator.close()


class FoundationWorker:
    def __init__(self, cfg):
        import torch
        import trimesh
        import nvdiffrast.torch as dr
        from estimater import FoundationPose
        from learning.training.predict_pose_refine import PoseRefinePredictor
        from learning.training.predict_score import ScorePredictor
        runtime_settings.configure_libraries()
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        self.scratch = Path(cfg["scratch"])
        mesh = trimesh.load(cfg["mesh"], process=False)
        if hasattr(mesh, "geometry"):
            geometries = list(mesh.geometry.values())
            if not geometries:
                raise ValueError("Mesh scene contains no geometry")
            mesh = trimesh.util.concatenate(geometries)
        if not len(mesh.vertices):
            raise ValueError("Mesh has no vertices")
        try:
            mesh.fix_normals()
        except Exception:
            pass
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        official = Path(cfg["repo"]) / "weights/2023-10-28-18-33-37/model_best.pth"
        if Path(cfg["refiner_weight"]).resolve() != official.resolve():
            checkpoint = torch.load(cfg["refiner_weight"], map_location="cuda")
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                checkpoint = checkpoint["model"]
            self.refiner.model.load_state_dict(checkpoint)
            self.refiner.model.cuda().eval()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(model_pts=np.asarray(mesh.vertices, dtype=np.float64),
            model_normals=np.asarray(mesh.vertex_normals, dtype=np.float64), mesh=mesh,
            scorer=self.scorer, refiner=self.refiner, glctx=self.glctx, debug=0,
            debug_dir=str(self.scratch / "foundationpose_debug"))
        exported = getattr(self.est, "mesh_path", None)
        if exported is not None:
            (self.scratch / "owned_mesh.json").write_text(json.dumps({"path": str(exported)}), encoding="utf-8")
        self.calls = 0

    def handle(self, request):
        import torch
        if request["operation"] != "register" or int(request["iteration"]) < 1:
            raise ValueError("Expected independent register request with positive iterations")
        with np.load(self.scratch / "register_input.npz", allow_pickle=False) as data:
            rgb, depth, mask, K = (data[k] for k in ("rgb", "depth", "mask", "K"))
        # Early-return/error must never expose previous request's score/pose.
        for attr in ("pose_last", "poses", "scores", "best_id", "H", "W", "K", "ob_id", "ob_mask"):
            if hasattr(self.est, attr):
                delattr(self.est, attr)
        self.est.pose_last = None
        pose = self.est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                 iteration=int(request["iteration"]))
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(pose).all():
            raise RuntimeError("FoundationPose returned a non-finite pose")
        scores = getattr(self.est, "scores", None)
        if scores is None:
            # register can early-return without scoring when depth support is
            # insufficient. Report proposal absence rather than stale scores.
            return dict(frame_index=request["frame_index"], pose=pose.tolist(),
                diagnostics={"foundationpose_error": "register_returned_without_scores",
                             "foundationpose_score_count": 0, "raw_valid": False})
        if torch.is_tensor(scores):
            scores = scores.detach().float().cpu().numpy()
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        scores = np.sort(scores[np.isfinite(scores)])[::-1]
        self.calls += 1
        diagnostics = dict(foundationpose_error=None, raw_valid=True,
            foundationpose_score_count=int(len(scores)),
            foundationpose_score_top1=float(scores[0]) if len(scores) else None,
            foundationpose_score_top2=float(scores[1]) if len(scores) > 1 else None,
            foundationpose_score_margin=float(scores[0]-scores[1]) if len(scores) > 1 else None)
        return dict(frame_index=request["frame_index"], pose=pose.tolist(), diagnostics=diagnostics)

    def close(self):
        pass  # CUDA objects are released when this episode worker exits.


def main():
    # Preserve a private copy of the IPC pipe, then redirect OS-level stdout as
    # well as Python prints. C++/CUDA/Open3D messages must not corrupt JSON IPC.
    channel = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1, encoding='utf-8')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    kind, path = sys.argv[1:]
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    worker = None
    try:
        # Initialization is inside the first request: errors reach the parent
        # with the request ID, and model startup is included in measured cost.
        for line in sys.stdin:
            request = json.loads(line)
            try:
                import torch
                if worker is None:
                    configure_component_imports(kind, cfg["repo"])
                    runtime_settings.configure_libraries()
                start = time.perf_counter()
                with torch.inference_mode():
                    if worker is None:
                        random.seed(cfg["seed"])
                        np.random.seed(cfg["seed"])
                        torch.manual_seed(cfg["seed"])
                        torch.backends.cudnn.benchmark = False
                        worker = {"sam2": SamWorker, "foundationpose": FoundationWorker}[kind](cfg)
                    result = worker.handle(request)
                torch.cuda.synchronize()
                result.update(worker_wall_ms=(time.perf_counter()-start)*1000.,
                    peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                    peak_reserved_mb=torch.cuda.max_memory_reserved()/1024**2)
                # Release unused allocator blocks while keeping live model/state
                # tensors. This helps co-resident SE3/FP without dropping state.
                torch.cuda.empty_cache()
                result["request_id"] = request["request_id"]
                channel.write(json.dumps(result, allow_nan=False) + "\n")
                channel.flush()
            except Exception as exc:
                traceback.print_exc()
                channel.write(json.dumps(dict(request_id=request["request_id"], error=repr(exc))) + "\n")
                channel.flush()
                return 1
    finally:
        if worker is not None:
            worker.close()
    return 0


def check_environment(kind, repo):
    """Import/API check only: no model loading, inference or CUDA allocation."""
    import inspect
    configure_component_imports(kind, repo)
    if kind == "sam2":
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        from sam2.utils.misc import _load_img_as_tensor
        expected = {"init_state": ("video_path", "offload_video_to_cpu", "offload_state_to_cpu", "async_loading_frames"),
                    "propagate_in_video": ("start_frame_idx", "max_frame_num_to_track", "reverse"),
                    "add_new_mask": ("frame_idx", "obj_id", "mask")}
        for name, parameters in expected.items():
            available = inspect.signature(getattr(SAM2VideoPredictor, name)).parameters
            if not set(parameters) <= set(available):
                raise RuntimeError("Incompatible SAM2 API: " + name)
    elif kind == "foundationpose":
        from estimater import FoundationPose
        from learning.training.predict_pose_refine import PoseRefinePredictor
        from learning.training.predict_score import ScorePredictor
        parameters = inspect.signature(FoundationPose.register).parameters
        if not {"K", "rgb", "depth", "ob_mask", "iteration"} <= set(parameters):
            raise RuntimeError("Incompatible FoundationPose.register API")
    else:
        raise ValueError(kind)
    print("Persistent perception import/API check passed:", kind)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        check_environment(sys.argv[2], os.path.abspath(sys.argv[3]))
    else:
        sys.exit(main())
