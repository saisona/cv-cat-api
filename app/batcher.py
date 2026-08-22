import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceJob:
    tensor: torch.Tensor  # Single preprocessed item (C, H, W) on CPU
    future: asyncio.Future[float]


def configure_torch_concurrency() -> int:
    """
    Configures PyTorch intra-op threads to prevent CPU thrashing under high concurrency.
    Prioritizes explicit TORCH_NUM_THREADS, falls back to cgroup/CPU core allocation.
    """
    num_threads = None
    if "TORCH_NUM_THREADS" in os.environ:
        num_threads = int(os.environ["TORCH_NUM_THREADS"])
    else:
        # Respect Docker/cgroup limits if available, fallback to os.cpu_count() or 2
        try:
            # 2. Linux-specific sched_getaffinity (Docker / Kubernetes on Linux)
            if num_threads is None and hasattr(os, "sched_getaffinity"):
                try:
                    num_threads = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
                except Exception:
                    pass

            # 3. Fallback for macOS, Windows, or undetermined affinity
            if num_threads is None:
                num_threads = os.cpu_count() or 2
        except (AttributeError, NotImplementedError):
            num_threads = os.cpu_count() or 2

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)

    logger.info(f"PyTorch thread pool configured: intra-op={num_threads}, inter-op=1")
    return num_threads


class DynamicBatcher:
    def __init__(
        self,
        model: torch.nn.Module,
        max_batch_size: int = 16,
        max_wait_time_ms: float = 10.0,
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.max_batch_size = max_batch_size
        self.max_wait_time_sec = max_wait_time_ms / 1000.0
        self.use_fp16 = self.device.type == "cuda"
        logger.info(
            f"Initializing DynamicBatcher: device={self.device,}, fp16={self.use_fp16,}, max_batch_size={self.max_batch_size,}, max_wait_ms=%.1fms",
            max_wait_time_ms,
        )
        # Initialize and compile the execution engine
        self.engine = self._prepare_engine(model)

        # Internal queue and background worker handle
        self.queue: asyncio.Queue[InferenceJob] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._running: bool = False

    def _prepare_engine(self, model: torch.nn.Module) -> torch.jit.ScriptModule:
        """Traces, freezes, and optimizes a PyTorch model for inference.

        Transfers the model to the target compute device, applies half-precision (FP16)
        if running on CUDA, and exports the module to TorchScript via tracing. For CPU
        runtimes, it additionally applies graph-level optimizations (kernel fusions,
        memory planning).

        Args:
            model (torch.nn.Module): The standard PyTorch module to compile.

        Returns:
            torch.jit.ScriptModule: The optimized, immutable TorchScript execution engine.

        Raises:
            AssertionError: If tracing fails to produce a valid `torch.jit.ScriptModule`.
        """

        t0 = time.perf_counter()
        model.eval()
        model.to(self.device)

        if self.use_fp16:
            model.half()

        dummy_input = torch.randn(
            1,
            3,
            224,
            224,
            dtype=torch.float16 if self.use_fp16 else torch.float32,
            device=self.device,
        )

        with torch.inference_mode():
            traced = torch.jit.trace(model, dummy_input)
            assert isinstance(
                traced, torch.jit.ScriptModule
            ), "Traced model must be a ScriptModule"

            frozen = torch.jit.freeze(traced)

            logger.info("Inference Worker is ready")
            # optimize_for_inference is designed specifically for CPU execution
            compile_time_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "TorchScript engine compiled & optimized successfully in %.2f ms",
                compile_time_ms,
            )

            if self.device.type == "cpu":
                return torch.jit.optimize_for_inference(frozen)
            return frozen

    def start(self) -> None:
        """Start the background consumer loop."""
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("DynamicBatcher background consumer worker started.")

    async def stop(self) -> None:
        """Gracefully stop the background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("DynamicBatcher worker stopped.")

    async def predict(self, tensor: torch.Tensor) -> float:
        """FastAPI route entrypoint: Enqueues an individual tensor and awaits result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[float] = loop.create_future()

        job = InferenceJob(tensor=tensor, future=future)
        await self.queue.put(job)
        return await future

    async def _worker_loop(self) -> None:
        """Background worker that batches incoming inference jobs dynamically.

        Runs continuously while the batcher is active:
        1. Waits asynchronously for at least one item to arrive in the queue.
        2. Gathers additional items until `max_batch_size` is reached or the
           `max_wait_time_sec` window expires (micro-batching).
        3. Offloads the heavy forward pass to a separate worker thread pool via
           `asyncio.to_thread` to prevent blocking the event loop.
        4. Resolves each job's `asyncio.Future` with its corresponding prediction,
           or propagates exceptions to all futures if the batch forward pass fails.
        5. Acknowledges processed items via `queue.task_done()`.
        """
        while self._running:
            # 1. Block until at least one request arrives
            first_job = await self.queue.get()
            batch: List[InferenceJob] = [first_job]

            # 2. Collect more items up to MAX_BATCH_SIZE or timeout
            deadline = asyncio.get_running_loop().time() + self.max_wait_time_sec
            while len(batch) < self.max_batch_size:
                remaining_time = deadline - asyncio.get_running_loop().time()
                if remaining_time <= 0:
                    break
                try:
                    job = await asyncio.wait_for(
                        self.queue.get(), timeout=remaining_time
                    )
                    batch.append(job)
                except asyncio.TimeoutError:
                    break

            logger.info(
                "Batch formed: size=%d/%d | pending_in_queue=%d",
                len(batch),
                self.max_batch_size,
                self.queue.qsize(),
            )

            # 3. Offload compute-heavy inference to a worker thread
            try:
                results = await asyncio.to_thread(self._run_inference_sync, batch)
                for job, result in zip(batch, results):
                    if not job.future.done():
                        job.future.set_result(result)
            except Exception as exc:
                logger.exception("Inference batch execution failed.")
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)
            finally:
                for _ in range(len(batch)):
                    self.queue.task_done()

    def _run_inference_sync(self, batch: List[InferenceJob]) -> List[float]:
        """Synchronous forward pass running in the thread pool."""
        t0 = time.perf_counter()
        tensors = [job.tensor for job in batch]
        logger.info(f"inference job launched with {len(batch)} items !!!!!")

        sys.stderr.write(f"\n>>> INFERENCE EXECUTING ON {len(batch)} ITEMS <<<\n")
        sys.stderr.flush()

        # Stack CPU tensors into (N, 3, H, W) and transfer to target device & dtype
        target_dtype = torch.float16 if self.use_fp16 else torch.float32
        batch_tensor = torch.stack(tensors).to(self.device, dtype=target_dtype)

        with torch.inference_mode():
            logits = self.engine(batch_tensor)
            probs = torch.softmax(logits, dim=-1)

            # Extract class 1 (cat probability) efficiently
            # 281: tabby, 282: tiger cat, 283: Persian cat, 284: Siamese cat, 285: Egyptian cat
            CAT_INDICES = [281, 282, 283, 284, 285]

            # Sum probabilities across all domestic cat breeds
            cat_probs = probs[:, CAT_INDICES].sum(dim=-1).tolist()
            infer_time_ms = (time.perf_counter() - t0) * 1000
            ms_per_item = infer_time_ms / len(batch)

            logger.info(
                "Inference pass finished: batch_size=%d | total_compute=%.2f ms (%.2f ms/item)",
                len(batch),
                infer_time_ms,
                ms_per_item,
            )
            return [(p) for p in cat_probs]
