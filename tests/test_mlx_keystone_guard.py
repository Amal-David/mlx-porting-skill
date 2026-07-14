"""Contracts for required Apple-Silicon MLX keystone execution."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from tests.mlx_keystone import MLX_KEYSTONE_REQUIRED_ENV, require_mlx_keystone


class MLXKeystoneGuardTests(unittest.TestCase):
    def test_required_flag_converts_missing_mlx_keystone_skip_into_failure(self) -> None:
        with mock.patch.dict(os.environ, {MLX_KEYSTONE_REQUIRED_ENV: "1"}):

            @require_mlx_keystone(False, "fixture MLX runtime is missing")
            class MissingMLXKeystone(unittest.TestCase):
                def test_generated_model_math(self) -> None:
                    self.fail("the original math test must not run without MLX")

            result = unittest.TestResult()
            unittest.defaultTestLoader.loadTestsFromTestCase(MissingMLXKeystone).run(result)

        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.skipped, [])
        self.assertEqual(len(result.failures), 1)
        self.assertIn("MLX_KEYSTONE_REQUIRED=1 requires this MLX keystone", result.failures[0][1])

    def test_required_flag_still_skips_missing_real_model_keystone(self) -> None:
        with mock.patch.dict(os.environ, {MLX_KEYSTONE_REQUIRED_ENV: "1"}):

            @require_mlx_keystone(
                False,
                "cached real checkpoint is unavailable",
                needs_real_model=True,
            )
            class MissingRealModelKeystone(unittest.TestCase):
                def test_real_checkpoint_parity(self) -> None:
                    self.fail("a missing downloaded model must not fail a stock CI lane")

            result = unittest.TestResult()
            unittest.defaultTestLoader.loadTestsFromTestCase(MissingRealModelKeystone).run(result)

        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.failures, [])

    def test_missing_mlx_keystone_still_skips_when_not_required(self) -> None:
        with mock.patch.dict(os.environ, {MLX_KEYSTONE_REQUIRED_ENV: "0"}):

            @require_mlx_keystone(False, "fixture MLX runtime is missing")
            class OptionalMLXKeystone(unittest.TestCase):
                def test_generated_model_math(self) -> None:
                    self.fail("optional missing MLX must skip")

            result = unittest.TestResult()
            unittest.defaultTestLoader.loadTestsFromTestCase(OptionalMLXKeystone).run(result)

        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.failures, [])


if __name__ == "__main__":
    unittest.main()
