import unittest
import tomllib
from pathlib import Path


class PackageTests(unittest.TestCase):
    def test_package_imports(self) -> None:
        import ostosense_ai

        self.assertTrue(hasattr(ostosense_ai, "__version__"))

    def test_package_version_matches_project_release(self) -> None:
        import ostosense_ai

        project = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        )
        self.assertEqual(ostosense_ai.__version__, project["project"]["version"])

    def test_package_avoids_eager_public_imports(self) -> None:
        import ostosense_ai

        self.assertEqual(ostosense_ai.__all__, [])


if __name__ == "__main__":
    unittest.main()
