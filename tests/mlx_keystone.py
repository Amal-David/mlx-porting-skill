"""Fail-loud guard for MLX math tests on required Apple-Silicon lanes."""
from __future__ import annotations

import functools
import os
import unittest
from typing import Callable, TypeVar


MLX_KEYSTONE_REQUIRED_ENV = "MLX_KEYSTONE_REQUIRED"
TestCaseType = TypeVar("TestCaseType", bound=type[unittest.TestCase])
_INITIAL_REQUIRED_VALUE = os.environ.pop(MLX_KEYSTONE_REQUIRED_ENV, None)


def mlx_keystone_required() -> bool:
    override = os.environ.get(MLX_KEYSTONE_REQUIRED_ENV)
    if override is not None:
        return override == "1"
    return _INITIAL_REQUIRED_VALUE == "1"


def require_mlx_keystone(
    mlx_available: bool,
    reason: str,
) -> Callable[[TestCaseType], TestCaseType]:
    """Skip normally, but replace MLX tests with failures on required lanes."""

    def decorate(test_case: TestCaseType) -> TestCaseType:
        if mlx_available:
            return test_case
        if not mlx_keystone_required():
            return unittest.skip(reason)(test_case)

        message = (
            f"{MLX_KEYSTONE_REQUIRED_ENV}=1 requires this MLX keystone, "
            f"but the runtime is unavailable: {reason}"
        )
        for name, member in tuple(vars(test_case).items()):
            if not name.startswith("test") or not callable(member):
                continue

            @functools.wraps(member)
            def fail_required_keystone(self: unittest.TestCase, failure: str = message) -> None:
                self.fail(failure)

            setattr(test_case, name, fail_required_keystone)
        test_case.__unittest_skip__ = False
        test_case.__unittest_skip_why__ = ""
        return test_case

    return decorate
