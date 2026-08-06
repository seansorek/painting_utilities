"""Regression test for #80: heavy image commands must not be able to fill an
unbounded executor queue and exhaust memory.

``_CPU_EXECUTOR`` has ``max_workers=1``, but a ``ThreadPoolExecutor``'s
submission queue is unbounded — nothing previously stopped a burst of
concurrent requests from all reading their attachments into memory and
submitting closures that queue up behind the single worker. ``bot`` now
guards every heavy command with a small process-wide
``asyncio.Semaphore`` (``_HEAVY_SEMAPHORE``), acquired via
``_acquire_heavy_slot`` BEFORE any attachment is downloaded.

This test starts more concurrent ``/analyze`` invocations than the
semaphore's capacity allows and asserts that the excess requests are
rejected — without ever calling ``attachment.read()`` or submitting any
work to the CPU executor (``bot._run_cpu``) — while the admitted requests
do proceed to read their attachment and submit executor work.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import bot


class _FakeAttachment:
    """Minimal stand-in for discord.Attachment whose .read() blocks on a
    shared gate, so we can hold "in-flight" requests open long enough to
    observe how many were admitted before letting them complete.
    """

    def __init__(self, gate: asyncio.Event, filename: str = "art.png"):
        self.content_type = "image/png"
        self.size = 1000
        self.width = 32
        self.height = 32
        self.filename = filename
        self._gate = gate
        self.read = AsyncMock(side_effect=self._read)

    async def _read(self) -> bytes:
        await self._gate.wait()
        return b"fake-image-bytes"


def _make_ctx(user_id: int) -> MagicMock:
    ctx = MagicMock()
    ctx.author.id = user_id
    ctx.defer = AsyncMock()
    ctx.followup = MagicMock()
    ctx.followup.send = AsyncMock()
    return ctx


@pytest.fixture(autouse=True)
def _reset_state():
    bot._USER_COOLDOWNS.clear()
    yield
    bot._USER_COOLDOWNS.clear()


class TestHeavyCommandAdmissionControl:
    @pytest.mark.asyncio
    async def test_excess_requests_rejected_without_reading_or_submitting(self):
        capacity = 2
        n_requests = capacity + 3  # 5 total: 2 should be admitted, 3 rejected

        test_semaphore = asyncio.Semaphore(capacity)
        gate = asyncio.Event()

        attachments = [_FakeAttachment(gate) for _ in range(n_requests)]
        ctxs = [_make_ctx(user_id=i) for i in range(n_requests)]

        with patch.object(bot, "_HEAVY_SEMAPHORE", test_semaphore), \
             patch.object(bot, "_run_cpu", new=AsyncMock(
                 return_value=(None, None, None, None, None)
             )) as mock_run_cpu:

            tasks = [
                asyncio.create_task(
                    bot.analyze.callback(
                        ctxs[i], attachments[i],
                        num_colors=10, show_rgb=False, show_cmyk=False,
                        saturation_boost=0.0, brightness_boost=0.0,
                    )
                )
                for i in range(n_requests)
            ]

            # Let every task run up to its first real suspension point:
            # admitted requests block inside attachment.read() (on `gate`),
            # rejected requests block inside the semaphore's short timeout.
            await asyncio.sleep(0.02)

            admitted_reads = [a.read.called for a in attachments]
            assert sum(admitted_reads) == capacity, (
                f"Expected exactly {capacity} attachments to be read "
                f"(admitted requests), got {sum(admitted_reads)}"
            )

            # No executor work must have been submitted yet — admitted
            # requests are still stuck at `await image.read()`.
            mock_run_cpu.assert_not_called()

            # Let the rejected requests' short acquire-timeout elapse so
            # they send their "busy" response and return.
            await asyncio.sleep(0.2)

            for i, ctx in enumerate(ctxs):
                if admitted_reads[i]:
                    continue
                # A rejected request must not have read its attachment...
                assert not attachments[i].read.called, (
                    f"Rejected request {i} must not read its attachment"
                )
                # ...and must have sent an ephemeral "busy" response instead.
                assert ctx.followup.send.called, (
                    f"Rejected request {i} should have sent a busy response"
                )
                sent_text = ctx.followup.send.call_args.args[0] if ctx.followup.send.call_args.args else ""
                kwargs = ctx.followup.send.call_args.kwargs
                assert "busy" in sent_text.lower()
                assert kwargs.get("ephemeral") is True

            # Now let the admitted requests finish.
            gate.set()
            await asyncio.gather(*tasks)

            # Only the admitted requests should ever have submitted
            # executor work — never more than the semaphore's capacity.
            assert mock_run_cpu.call_count == capacity, (
                f"Expected exactly {capacity} calls to _run_cpu (one per "
                f"admitted request), got {mock_run_cpu.call_count}"
            )

            # The semaphore must be fully released again afterwards (every
            # acquire path has a matching release in `finally`).
            assert test_semaphore._value == capacity

    @pytest.mark.asyncio
    async def test_acquire_heavy_slot_rejects_when_full_without_side_effects(self):
        """Focused unit test on _acquire_heavy_slot itself: when the
        semaphore is fully held, it must return False immediately (within
        its short timeout) and send an ephemeral busy message, without the
        caller ever reaching attachment download or executor submission.
        """
        test_semaphore = asyncio.Semaphore(1)
        await test_semaphore.acquire()  # simulate one in-flight request

        ctx = _make_ctx(user_id=99)

        with patch.object(bot, "_HEAVY_SEMAPHORE", test_semaphore):
            acquired = await bot._acquire_heavy_slot(ctx)

        assert acquired is False
        ctx.followup.send.assert_called_once()
        args, kwargs = ctx.followup.send.call_args
        assert "busy" in args[0].lower()
        assert kwargs.get("ephemeral") is True

    @pytest.mark.asyncio
    async def test_acquire_and_release_round_trip(self):
        """A slot acquired via _acquire_heavy_slot must be returned to the
        semaphore by _release_heavy_slot, freeing it for the next request.
        """
        test_semaphore = asyncio.Semaphore(1)
        ctx = _make_ctx(user_id=1)

        with patch.object(bot, "_HEAVY_SEMAPHORE", test_semaphore):
            acquired = await bot._acquire_heavy_slot(ctx)
            assert acquired is True
            assert test_semaphore.locked()

            bot._release_heavy_slot()
            assert not test_semaphore.locked()
