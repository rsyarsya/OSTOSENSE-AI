"""Check tracked and candidate files without scanning ignored local data."""

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROHIBITED = re.compile(
    r"(^|/)(\.env($|\.)|.*\.(pem|key|p12|pfx)$|P00[1-7](_[0-9]+)?\.csv$|"
    r"(ordinal_model_params|golden_vectors|reconstructed_windows)\.hpp$)"
)


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    errors = []
    for name in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
        if PROHIBITED.search(name) and Path(name).name != ".env.example":
            errors.append(f"Prohibited file: {name}")
        path = ROOT / name
        if not path.is_file():
            continue  # A tracked deletion can remain in the index until staging.
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if line.endswith((" ", "\t")):
                errors.append(f"Trailing whitespace: {name}:{number}")
    for error in errors:
        print(error)
    if not errors:
        print("Repository hygiene passed.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
