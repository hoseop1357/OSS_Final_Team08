"""저장소 골격 스모크 테스트: 모든 하위 패키지가 import되는지 확인한다."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    ["src", "src.data", "src.train", "src.logic", "src.utils"],
)
def test_package_importable(module: str) -> None:
    assert importlib.import_module(module) is not None
