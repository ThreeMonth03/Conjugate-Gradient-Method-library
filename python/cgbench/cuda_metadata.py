"""Audit the actual CUDA device/runtime/extension used, outside timed blocks."""

import csv
import hashlib
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path


def cuda_metadata(device):
    metadata = {
        "device": device,
        "native": None,
        "torch": None,
        "nvidia_smi": None,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "CUDA_LAUNCH_BLOCKING",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
    }
    extension = sys.modules.get("_cgcuda")
    package = sys.modules.get("cgpy.cuda")
    if extension is not None and getattr(extension, "__file__", None):
        binary = Path(extension.__file__).resolve()
        metadata["native"] = {
            "module_path": str(binary),
            "module_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "build_info": package.build_info() if package is not None else None,
            "runtime_info": package.runtime_info(device=device) if package is not None else None,
        }
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and device < torch.cuda.device_count():
        properties = torch.cuda.get_device_properties(device)
        metadata["torch"] = {
            "runtime_version": torch.version.cuda,
            "device": device,
            "name": properties.name,
            "uuid": str(properties.uuid) if getattr(properties, "uuid", None) is not None else None,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    elif torch is not None:
        metadata["torch"] = {"runtime_version": torch.version.cuda, "available": False}
    executable = shutil.which("nvidia-smi")
    wsl_executable = Path("/usr/lib/wsl/lib/nvidia-smi")
    if executable is None and wsl_executable.is_file():
        executable = str(wsl_executable)
    if executable is not None:
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=index,name,uuid,driver_version,pci.bus_id",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            names = ("physical_index", "name", "uuid", "driver_version", "pci_bus_id")
            metadata["nvidia_smi"] = [
                dict(zip(names, (value.strip() for value in row), strict=True))
                for row in csv.reader(io.StringIO(result.stdout))
            ]
            metadata["nvidia_smi_mapping"] = (
                "physical device inventory; CUDA_VISIBLE_DEVICES may change logical ordinals"
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return metadata
