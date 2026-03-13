import subprocess
import sys
import textwrap


PYTHON_EXCEPTION_EXIT = 10


PROBES = [
    (
        "runtime_info",
        """
import sys
import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"hip={getattr(torch.version, 'hip', None)}")
print(f"cuda={getattr(torch.version, 'cuda', None)}")
if torch.cuda.is_available():
    print(f"device_count={torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"device0={torch.cuda.get_device_name(0)}")
""",
    ),
    (
        "tensor_to_cuda",
        """
import torch

x = torch.zeros(4)
y = x.to("cuda")
print(f"device={y.device}")
print(f"shape={tuple(y.shape)}")
""",
    ),
    (
        "randn_cuda",
        """
import torch

x = torch.randn(2, 3, device="cuda")
print(f"device={x.device}")
print(f"sum={float(x.sum().item())}")
""",
    ),
    (
        "torch_cat",
        """
import torch

a = torch.ones(2, device="cuda")
b = torch.zeros(2, device="cuda")
c = torch.cat([a, b], dim=0)
print(f"device={c.device}")
print(f"shape={tuple(c.shape)}")
""",
    ),
    (
        "cosine_similarity",
        """
import torch
import torch.nn.functional as F

x = torch.ones((1, 8), device="cuda")
y = torch.arange(1, 9, device="cuda", dtype=torch.float32).unsqueeze(0)
score = F.cosine_similarity(x, y, dim=-1)
print(f"device={score.device}")
print(f"value={float(score[0].item())}")
""",
    ),
]


def _wrap_probe(code: str) -> str:
    body = textwrap.indent(textwrap.dedent(code).strip(), "    ")
    return "\n".join(
        [
            "import sys",
            "import traceback",
            "try:",
            body,
            "except Exception:",
            "    traceback.print_exc()",
            f"    sys.exit({PYTHON_EXCEPTION_EXIT})",
        ]
    )


def _run_probe(name: str, code: str):
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _wrap_probe(code)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode == 0:
            status = "pass"
        elif completed.returncode == PYTHON_EXCEPTION_EXIT:
            status = "python_exception"
        else:
            status = "process_crash"
        return {
            "name": name,
            "status": status,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "status": "process_crash",
            "returncode": "timeout",
            "stdout": (exc.stdout or "").strip(),
            "stderr": (exc.stderr or "Probe timed out.").strip(),
        }


def main():
    print(f"python_executable={sys.executable}")
    for name, code in PROBES:
        result = _run_probe(name, code)
        print(f"[{result['status']}] {result['name']} (returncode={result['returncode']})")
        if result["stdout"]:
            print(textwrap.indent(result["stdout"], "  stdout: "))
        if result["stderr"]:
            print(textwrap.indent(result["stderr"], "  stderr: "))


if __name__ == "__main__":
    main()
