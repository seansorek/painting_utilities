"""Tests for _run_sync / _run_cpu and their use in command handlers.

Verifies that:
1. _run_sync correctly offloads work to a dedicated worker thread.
2. Each image command handler offloads CPU-bound work via _run_sync / _run_cpu.
"""
import asyncio
import inspect
import threading

import pytest

import bot


class TestRunSync:
    """Unit tests for the _run_sync / _run_cpu helper."""

    @pytest.mark.asyncio
    async def test_delegates_to_executor(self):
        """_run_sync must run the function on the dedicated CPU executor thread."""
        worker_name = await bot._run_sync(lambda: threading.current_thread().name)
        assert worker_name.startswith("img-worker"), (
            f"Expected work to run on img-worker thread, got {worker_name!r}"
        )

    @pytest.mark.asyncio
    async def test_passes_kwargs(self):
        """_run_sync must forward keyword arguments."""
        result = await bot._run_sync(int, "0b101", base=2)
        assert result == 5

    @pytest.mark.asyncio
    async def test_returns_function_result(self):
        """_run_sync should actually run the function in a thread and return its result."""
        result = await bot._run_sync(sum, [10, 20, 30])
        assert result == 60

    @pytest.mark.asyncio
    async def test_propagates_exceptions(self):
        """Exceptions from the sync function must propagate."""
        def _fail():
            raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            await bot._run_sync(_fail)

    @pytest.mark.asyncio
    async def test_runs_in_separate_thread(self):
        """The function should execute in a different thread than the event loop."""
        loop_thread = threading.current_thread().ident

        def _get_thread():
            return threading.current_thread().ident

        worker_thread = await bot._run_sync(_get_thread)
        assert worker_thread != loop_thread

    def test_run_sync_is_run_cpu(self):
        """_run_sync must be the same function as _run_cpu (alias)."""
        assert bot._run_sync is bot._run_cpu


class TestHandlersUseRunSync:
    """Verify that each image command handler contains a call to _run_sync or _run_cpu."""

    # All command handlers that process images and must offload CPU work.
    HANDLERS = [
        "analyze",
        "palette",
        "gradient_map_cmd",
        "palette_gradient_cmd",
        "export_palette_cmd",
        "export_gradient_cmd",
        "color_info_cmd",
        "compare_cmd",
        "colorblind_cmd",
        "recolor_cmd",
        "suggest_harmony_cmd",
    ]

    @pytest.mark.parametrize("handler_name", HANDLERS)
    def test_handler_awaits_run_sync(self, handler_name):
        """Each handler's source must contain an offload call to prove CPU work is threaded."""
        func = getattr(bot, handler_name)
        # Unwrap the discord.py command wrapper to get the actual coroutine
        callback = getattr(func, "callback", func)
        source = inspect.getsource(callback)
        assert (
            "await _run_sync(" in source
            or "await _run_cpu(" in source
            or "await asyncio.to_thread(" in source
        ), (
            f"Handler {handler_name!r} does not offload CPU work via "
            "_run_sync, _run_cpu, or asyncio.to_thread"
        )

    @pytest.mark.parametrize("handler_name", HANDLERS)
    def test_handler_is_coroutine(self, handler_name):
        """Each handler must be an async function."""
        func = getattr(bot, handler_name)
        callback = getattr(func, "callback", func)
        assert asyncio.iscoroutinefunction(callback), (
            f"Handler {handler_name!r} is not an async function"
        )
