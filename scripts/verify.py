"""Run repository quality checks on Windows, Linux, or macOS."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def check_cpp(compiler: str) -> None:
    with tempfile.TemporaryDirectory(prefix="ostosense-cpp-") as temporary:
        for name in ("data_contract", "ordinal_inference", "capacitive_features"):
            target = Path(temporary) / (name + (".exe" if os.name == "nt" else ""))
            run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-I",
                    str(ROOT / "firmware/include"),
                    str(ROOT / f"firmware/tests/{name}_test.cpp"),
                    "-o",
                    str(target),
                ]
            )
            run([str(target)])


def check_package() -> None:
    with tempfile.TemporaryDirectory(prefix="ostosense-wheel-") as temporary:
        temp = Path(temporary)
        run([sys.executable, "-m", "build", "--outdir", str(temp / "dist")], cwd=ROOT / "ai")
        run([sys.executable, "-m", "venv", str(temp / "venv")])
        python = temp / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        wheels = list((temp / "dist").glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected one built wheel")
        run([str(python), "-I", "-m", "pip", "install", "--no-deps", str(wheels[0])], cwd=temp)
        run(
            [
                str(python),
                "-I",
                "-c",
                "import ostosense_ai, ostosense_contract; from importlib.metadata import version; "
                "assert ostosense_ai.__version__ == version('ostosense-contract')",
            ],
            cwd=temp,
        )
        run([str(python), "-I", "-m", "ostosense_ai.bundled_model"], cwd=temp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpp-only", action="store_true", help="Only compile and run C++ host tests."
    )
    args = parser.parse_args()
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if not compiler:
        parser.error("A C++17 compiler is required. Add g++ to PATH or set CXX to its path.")
    os.environ["CXX"] = compiler
    if os.name == "nt":
        # MinGW executables may need DLLs from the compiler's directory.
        os.environ["PATH"] = str(Path(compiler).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = str(ROOT / "ai/src")
    try:
        if not args.cpp_only:
            required = (
                "build",
                "jsonschema",
                "pyright",
                "ruff",
                "fastapi",
                "httpx",
                "uvicorn",
                "numpy",
                "scipy",
                "sklearn",
                "mord",
                "matplotlib",
            )
            missing = [name for name in required if importlib.util.find_spec(name) is None]
            if missing:
                parser.error(
                    "Missing dependencies: "
                    + ", ".join(missing)
                    + '. Install "./ai[pipeline,quality,serve,serve-test]" first.'
                )
            run([sys.executable, "-W", "error", "-m", "unittest", "discover", "-s", "ai/tests"])
            run([sys.executable, "-m", "ruff", "check", "ai/src", "ai/tests", "scripts"])
            run([sys.executable, "-m", "pyright", "--pythonpath", sys.executable], cwd=ROOT / "ai")
            check_package()
        check_cpp(compiler)
        if not args.cpp_only:
            run([sys.executable, str(ROOT / "scripts/check_repository.py")])
            run(["git", "diff", "--check"])
            run(["git", "diff", "--cached", "--check"])
    except (subprocess.CalledProcessError, OSError, RuntimeError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print("All requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
