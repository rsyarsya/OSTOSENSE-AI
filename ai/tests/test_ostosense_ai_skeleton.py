import unittest


class OstosenseAiSkeletonTests(unittest.TestCase):
    def test_package_imports(self) -> None:
        import ostosense_ai

        self.assertTrue(hasattr(ostosense_ai, "__version__"))

    def test_package_version_matches_project_release(self) -> None:
        import ostosense_ai

        self.assertEqual(ostosense_ai.__version__, "0.1.0")

    def test_package_avoids_eager_public_imports(self) -> None:
        import ostosense_ai

        self.assertEqual(ostosense_ai.__all__, [])


if __name__ == "__main__":
    unittest.main()
