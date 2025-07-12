"""Unit tests for the TAgent module."""

import pytest
from tagent.agent import Store, run_agent


def test_store_creation():
    """Test that Store can be created with initial state."""
    initial_state = {"test": "value"}
    store = Store(initial_state)
    assert store.state.data == initial_state


def test_store_import():
    """Test that we can import key components."""
    from tagent import Store, run_agent
    assert Store is not None
    assert run_agent is not None


def test_basic_functionality():
    """Test basic TAgent functionality."""
    # This is a minimal test to ensure the package works
    store = Store({"goal": "test"})
    assert "goal" in store.state.data
    assert store.state.data["goal"] == "test"
