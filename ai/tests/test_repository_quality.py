import re
import subprocess
import tomllib
import unittest
from pathlib import Path
from urllib.parse import unquote


AI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_ROOT.parent
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


class RepositoryQualityTests(unittest.TestCase):
    def _repository_files(self):
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return [REPO_ROOT / name for name in result.stdout.splitlines()]

    def test_python_minimum_and_quality_tools_are_declared(self):
        config = tomllib.loads((AI_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["project"]["requires-python"], ">=3.11")
        quality = config["project"]["optional-dependencies"]["quality"]
        self.assertEqual(
            {requirement.split("==", 1)[0] for requirement in quality},
            {"build", "jsonschema", "pyright", "ruff"},
        )

    def test_local_markdown_links_resolve(self):
        documents = [path for path in self._repository_files()
                     if path.suffix == ".md" and path.is_file()]
        broken = []
        for document in documents:
            text = document.read_text(encoding="utf-8")
            for raw_target in MARKDOWN_LINK.findall(text):
                target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
                if target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                relative_path = unquote(target.split("#", 1)[0])
                if relative_path and not (document.parent / relative_path).exists():
                    broken.append(f"{document.relative_to(REPO_ROOT)} -> {target}")
        self.assertEqual(broken, [])

    def test_required_integration_artifacts_exist(self):
        required = (
            AI_ROOT / "contracts" / "ai-runtime-output-v0.2.schema.json",
            AI_ROOT / "contracts" / "ai-feature-input-v0.1.schema.json",
            AI_ROOT / "contracts" / "typescript" / "ai-runtime-output-v0.2.ts",
            AI_ROOT / "contracts" / "typescript" / "ai-feature-input-v0.1.ts",
            AI_ROOT
            / "contracts"
            / "examples"
            / "feature-input-v0.1"
            / "real-kap7.json",
            REPO_ROOT / "docs" / "ai-software-integration-contract-v0.2.md",
            REPO_ROOT / ".github" / "workflows" / "quality.yml",
            REPO_ROOT / "scripts" / "verify.sh",
            REPO_ROOT / "scripts" / "build_engineering_demo.sh",
        )
        self.assertTrue(all(path.is_file() for path in required))

    def test_only_shell_scripts_are_executable(self):
        # Git's index carries portable executable modes; Windows st_mode does not.
        result = subprocess.run(
            ["git", "ls-files", "--stage", "-z"], cwd=REPO_ROOT,
            check=True, capture_output=True, text=True,
        )
        executable = set()
        for entry in result.stdout.split("\0"):
            if not entry:
                continue
            metadata, name = entry.split("\t", 1)
            if metadata.split()[0] == "100755":
                executable.add(name)
        self.assertEqual({name for name in executable if not name.endswith(".sh")}, set())
        self.assertTrue({"scripts/verify.sh", "scripts/build_engineering_demo.sh"} <= executable)


if __name__ == "__main__":
    unittest.main()
