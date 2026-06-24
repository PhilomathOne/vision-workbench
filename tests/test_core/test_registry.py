"""Tests for the Registry system — component discovery backbone."""

import pytest

from vision_workbench.core.exceptions import RegistryError
from vision_workbench.core.registry import Registry


class FakeBackbone:
    """Dummy class for registry testing."""

    task_type = "classification"
    framework = "test"


class FakeNeck:
    task_type = "detection"
    framework = "test"


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry("test")
        reg.register_direct("fake_backbone", FakeBackbone, task="classification")
        cls = reg.get("fake_backbone")
        assert cls is FakeBackbone

    def test_register_duplicate_raises(self):
        reg = Registry("test")
        reg.register_direct("fake", FakeBackbone)
        with pytest.raises(KeyError):
            reg.register_direct("fake", FakeBackbone)

    def test_get_unknown_raises(self):
        reg = Registry("test")
        with pytest.raises(RegistryError, match="Unknown test 'nonexistent'"):
            reg.get("nonexistent")

    def test_list(self):
        reg = Registry("test")
        reg.register_direct("a", FakeBackbone)
        reg.register_direct("b", FakeNeck)
        entries = reg.list()
        assert len(entries) == 2
        assert "a" in entries
        assert "b" in entries

    def test_list_by_metadata(self):
        reg = Registry("test")
        reg.register_direct("backbone", FakeBackbone, task="classification")
        reg.register_direct("neck", FakeNeck, task="detection")
        results = reg.list_by(task="classification")
        assert len(results) == 1
        assert "backbone" in results

    def test_decorator_register(self):
        reg = Registry("test")

        @reg.register("my_component", task="detection")
        class MyComponent:
            pass

        assert reg.get("my_component") is MyComponent
        meta = getattr(MyComponent, "_registry_metadata", {})
        assert meta.get("task") == "detection"

    def test_contains(self):
        reg = Registry("test")
        reg.register_direct("x", FakeBackbone)
        assert "x" in reg
        assert "y" not in reg

    def test_len(self):
        reg = Registry("test")
        assert len(reg) == 0
        reg.register_direct("x", FakeBackbone)
        assert len(reg) == 1
