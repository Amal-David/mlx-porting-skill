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
    *,
    needs_real_model: bool = False,
) -> Callable[[TestCaseType], TestCaseType]:
    """Skip normally, but replace MLX-math tests with failures on required lanes.

    A required lane (``MLX_KEYSTONE_REQUIRED=1``) exists to catch a silently
    skipped MLX-math keystone: pure MLX ops the runner can execute. When such a
    keystone is unavailable there, its skip becomes a hard failure.

    Real-checkpoint keystones (``needs_real_model=True``) additionally depend on
    a multi-gigabyte downloaded model that a stock CI runner cannot provision.
    Their unavailability is an environment gap, not an MLX-math regression, so
    they always skip when unavailable -- even on a required lane. Enforcing them
    belongs to a separate, model-provisioned lane.
    """

    def decorate(test_case: TestCaseType) -> TestCaseType:
        if mlx_available:
            return test_case
        if needs_real_model or not mlx_keystone_required():
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
