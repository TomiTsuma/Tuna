"""Scaffolding smoke test for the Python integration test harness.

Proves pytest discovery and execution works from this directory. Replaced
with a real mock QUIC server (accepting connections, responding to
heartbeat frames) once Phase 2's tunnel protocol exists.
"""


def test_pytest_harness_is_wired_up():
    assert 1 + 1 == 2
