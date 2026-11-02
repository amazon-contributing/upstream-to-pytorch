# Owner(s): ["oncall: distributed"]

"""Checks how MultiProcContinuousTest combines rank results and shuts down."""

import multiprocessing
import time
import unittest
from unittest.mock import patch

import torch.testing._internal.common_distributed as cd
from torch.testing._internal.common_utils import run_tests, TestCase


# Built in a function so unittest doesn't collect and spawn them.
def _classes():
    class Fake(cd.MultiProcContinuousTest):
        def test_x(self):
            pass

    class Hung(cd.MultiProcContinuousTest):
        pass

    return Fake, Hung


def _sleep():
    time.sleep(1000)


def _rank_with_unread_failure(task_queue, completion_queue, returned):
    class Rank(cd.MultiProcContinuousTest):
        @classmethod
        def _init_pg(cls, rank, world_size, rdvz_file):
            pass

        @classmethod
        def _run_test_given_id(cls, test_id, **kwargs):
            # Keep an unread traceback larger than a normal pipe buffer.
            raise RuntimeError("x" * 2**22)

    Rank._worker_loop(0, 1, None, task_queue, completion_queue)
    returned.set()


class TestMultiProcContinuousHarness(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.Fake, cls.Hung = _classes()

    def _wait(self, results):
        self.Fake.poison_pill = False
        t = self.Fake("test_x")
        t.rank = t.MAIN_PROCESS_RANK
        t.processes = [None] * len(results)
        t.completion_queues = [None] * len(results)
        it = iter(t.id() if r == "ok" else r for r in results)
        with patch.object(
            cd, "retrieve_result_from_completion_queue", lambda *a, **k: next(it)
        ):
            t.test_x()
        # Every queue must be drained so the next test sees its own results.
        self.assertIsNone(next(it, None))

    def test_failure_beats_skip(self):
        for results in (
            [unittest.SkipTest("s"), ValueError("v")],
            [ValueError("v"), unittest.SkipTest("s")],
        ):
            with self.assertRaises(ValueError):
                self._wait(results)
            self.assertTrue(self.Fake.poison_pill)

    def test_wrong_result_is_failure(self):
        with self.assertRaisesRegex(AssertionError, "Expected rv"):
            self._wait(["other", unittest.SkipTest("s")])
        self.assertTrue(self.Fake.poison_pill)

    def test_skip(self):
        with self.assertRaises(unittest.SkipTest):
            self._wait([unittest.SkipTest("s"), "ok"])
        self.assertFalse(self.Fake.poison_pill)

    def test_pass(self):
        self._wait(["ok", "ok"])
        self.assertFalse(self.Fake.poison_pill)

    def test_failure_stops_waiting(self):
        self.Fake.poison_pill = False
        t = self.Fake("test_x")
        t.rank = t.MAIN_PROCESS_RANK
        t.processes = [None, None]
        t.completion_queues = [None, None]
        results = iter([ValueError("v"), t.id()])
        with patch.object(
            cd, "retrieve_result_from_completion_queue", lambda *a, **k: next(results)
        ):
            with self.assertRaises(ValueError):
                t.test_x()
        self.assertEqual(next(results, None), t.id())
        self.assertTrue(self.Fake.poison_pill)

    def test_unread_result_does_not_block_exit(self):
        ctx = multiprocessing.get_context("spawn")
        task_queue, completion_queue = ctx.Queue(), ctx.Queue()
        returned = ctx.Event()
        p = ctx.Process(
            target=_rank_with_unread_failure,
            args=(task_queue, completion_queue, returned),
            daemon=True,
        )
        p.start()
        try:
            task_queue.put("test")
            task_queue.put(None)
            self.assertTrue(returned.wait(60), "rank did not finish its worker loop")
            p.join(60)
            self.assertEqual(p.exitcode, 0)
        finally:
            if p.is_alive():
                p.kill()
                p.join()
            task_queue.close()
            task_queue.join_thread()
            completion_queue.close()
            completion_queue.join_thread()

    def test_teardown_kills_hung_ranks(self):
        ctx = multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=_sleep, daemon=True) for _ in range(3)]
        for p in procs:
            p.start()
        self.Hung._processes_spawned = True
        self.Hung.processes = procs
        self.Hung.task_queues = []
        self.Hung.rdvz_file = "/nonexistent"
        start = time.monotonic()
        with patch.object(cd, "TIMEOUT_DEFAULT", 2):
            with self.assertRaisesRegex(
                RuntimeError, r"ranks \[0, 1, 2\] did not exit"
            ):
                self.Hung.tearDownClass()
        # One shared deadline, not one per rank.
        self.assertLess(time.monotonic() - start, 5)
        for p in procs:
            self.assertFalse(p.is_alive())


if __name__ == "__main__":
    run_tests()
