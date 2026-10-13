import asyncio
import contextlib
import io
import os
import re
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from optchat import api
from optchat.jobs import STATUS_AT
from optchat.service import Service


def answer(lines, stop="end_turn"):
    import json
    return {"content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": json.dumps({"lines": lines})}],
            "stop_reason": stop, "usage": {"input_tokens": 1000, "output_tokens": 100}}


class FakeApi:
    """Answers each job with one short line per task, and keeps a copy of every request."""

    def __init__(self, empty_first=False):
        self.requests, self.empty_first, self.n = [], empty_first, 0

    def __call__(self, key, messages):
        self.requests.append([dict(m) for m in messages])
        last = messages[-1]["content"]
        found = re.search(r"This job has (\d+) task", last) or re.search(r"Answer with (\d+) new line", last) or re.search(r"Answer again with (\d+) lines", last)
        lines = []
        for _ in range(int(found.group(1))):
            self.n += 1
            lines.append(f"note: summary {self.n}")
        if self.empty_first:
            self.empty_first = False
            lines[0] = ""
        return answer(lines)


class ApiWorker(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"OPTCHAT_CONFIG": str(Path(self.temp.name) / "config/config.json")})
        self.env.start()
        self.service = Service(Path(self.temp.name) / "chat")
        self.n = 0

    def tearDown(self):
        self.service.close()
        self.env.stop()
        self.temp.cleanup()

    def note(self, count):
        for _ in range(count):
            self.n += 1
            self.service.call("append", {"event_id": f"n{self.n}", "kind": "note", "text": f"note {self.n} " + "x" * 700})

    def compact(self, fake):
        self.service.api.ask = fake
        asyncio.run(self.service.api.compact(self.service.call, "sk-ant-test"))

    def test_the_api_worker_summarizes_the_backlog_in_one_append_only_conversation(self):
        self.note(25)
        fake = FakeApi()
        self.compact(fake)
        self.assertEqual(self.service.memory.first(), self.service.memory.total)
        self.assertEqual(self.service.api.last["result"], "done")
        self.assertEqual(self.service.api.last["lines"], 25)
        self.assertGreater(len(fake.requests), 1)
        for before, after in zip(fake.requests, fake.requests[1:]):
            self.assertEqual(after[:len(before)], before, "a worker only appends to its conversation")
        self.assertEqual(fake.requests[1][1]["content"][0]["type"], "thinking", "thinking blocks go back unchanged")
        self.assertEqual(self.service.api.total["requests"], len(fake.requests))
        self.assertAlmostEqual(self.service.api.total["usd"], len(fake.requests) * api.cost({"input_tokens": 1000, "output_tokens": 100}))

    def test_without_credit_the_api_pauses_and_the_subagent_gets_the_job(self):
        self.note(STATUS_AT)
        api.save_key("sk-ant-test")

        def no_credit(key, messages):
            raise api.ApiError("no credit", "HTTP 402: Your credit balance is too low to access the Anthropic API.")
        self.compact(no_credit)
        self.assertEqual(self.service.api.last["result"], "paused: no credit")
        self.assertEqual(self.service.board.leases, {})
        self.assertEqual(self.service.board.failures, {}, "a missing credit says nothing about the messages")
        self.assertFalse(self.service.api.ready("sk-ant-test"))
        self.assertTrue(self.service.api.ready("sk-ant-test", forced=True))
        self.assertTrue(self.service.api.ready("sk-ant-other"), "a new key is tried at once")
        self.assertEqual(self.service.call("api_status", {})["status"], "paused")
        self.service.api_enabled = True
        self.assertEqual(self.service.call("status_line", {})["text"], f"OptChat: {STATUS_AT} to summarize, API: no credit")
        self.assertEqual(self.service.call("compact_next", {})["status"], "claimed")

    def test_an_empty_line_releases_only_its_task(self):
        self.note(3)
        self.compact(FakeApi(empty_first=True))
        self.assertEqual(len(self.service.board.failures), 1)
        self.assertEqual(self.service.api.last["result"], "waiting", "the released task waits for a retry")
        self.assertEqual(self.service.api.last["lines"], 2)

    def test_a_refusal_releases_the_job(self):
        self.note(2)
        self.compact(lambda key, messages: {**answer([]), "stop_reason": "refusal", "stop_details": {"category": "cyber"}})
        self.assertEqual(self.service.board.failures.get((0, 0)), "The API declined the job (cyber).")

    def test_the_service_starts_only_when_enough_waits_or_after_a_quiet_time(self):
        now = [1000.0]
        worker = api.ApiCompactor(Path(self.temp.name), clock=lambda: now[0])
        self.assertTrue(worker.due(50, api.API_AT))
        self.assertFalse(worker.due(51, 1))
        now[0] += api.IDLE
        self.assertTrue(worker.due(51, 1))
        self.assertFalse(worker.due(52, 0))

    def test_compact_runs_the_api_now_when_the_command_asks(self):
        self.note(12)
        api.save_key("sk-ant-test")
        self.service.api_enabled = True
        self.service.api.ask = FakeApi()

        async def scenario():
            loop = self.service.summarize()
            task = asyncio.create_task(loop)
            started = self.service.call("api_compact", {})
            self.assertEqual(started["target"], 1)
            for _ in range(500):
                await asyncio.sleep(0.01)
                if self.service.api.run == 1 and not self.service.api.running:
                    break
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        asyncio.run(scenario())
        self.assertEqual(self.service.api.last["result"], "done")
        self.assertEqual(self.service.memory.first(), self.service.memory.total)


class Errors(unittest.TestCase):
    def test_http_errors_map_to_reasons(self):
        self.assertEqual(api.classify(402, b'{"error":{"type":"billing_error","message":"x"}}').kind, "no credit")
        self.assertEqual(api.classify(400, b'{"error":{"type":"invalid_request_error","message":"Your credit balance is too low"}}').kind, "no credit")
        self.assertEqual(api.classify(401, b"{}").kind, "key rejected")
        self.assertEqual(api.classify(404, b"{}").kind, "access denied")
        limited = api.classify(429, b"{}", "30")
        self.assertEqual((limited.kind, limited.retry_after), ("rate limited", 30.0))
        self.assertEqual(api.classify(529, b"not json").kind, "unreachable")
        self.assertEqual(api.classify(400, b'{"error":{"message":"bad field"}}').kind, "error")

    def test_cost_uses_the_rates_of_haiku(self):
        self.assertAlmostEqual(api.cost({"input_tokens": 100_000, "output_tokens": 10_000}), 0.01 + 0.005)
        self.assertAlmostEqual(api.cost({"input_tokens": 0, "cache_read_input_tokens": 1_000_000, "output_tokens": 0}), 0.50 * 0.1, msg="a prompt over 100K tokens costs more")


class KeyCommand(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"OPTCHAT_CONFIG": str(Path(self.temp.name) / "config.json")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def run_command(self, key, checker=lambda key: None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = api.key_command(browser=False, read=lambda prompt: key, checker=checker)
        return status, out.getvalue()

    def test_a_working_key_is_saved_for_the_user_only(self):
        status, text = self.run_command("sk-ant-api03-good")
        self.assertEqual(status, 0)
        self.assertIn(api.KEY_URL, text)
        self.assertNotIn("sk-ant-api03-good", text)
        self.assertEqual(api.load_key(), "sk-ant-api03-good")
        self.assertEqual(stat.S_IMODE(api.key_path().stat().st_mode), 0o600)

    def test_a_rejected_or_foreign_key_changes_nothing(self):
        def rejected(key):
            raise api.ApiError("key rejected", "HTTP 401: invalid x-api-key")
        self.assertEqual(self.run_command("sk-ant-bad", rejected)[0], 1)
        self.assertEqual(self.run_command("not-a-key")[0], 1)
        self.assertIsNone(api.load_key())

    def test_a_key_without_credit_is_kept_for_later(self):
        def no_credit(key):
            raise api.ApiError("no credit", "HTTP 402: no credit")
        status, text = self.run_command("sk-ant-empty", no_credit)
        self.assertEqual(status, 0)
        self.assertIn(api.BILLING_URL, text)
        self.assertEqual(api.load_key(), "sk-ant-empty")


class CompactCommand(unittest.TestCase):
    class Client:
        def __init__(self, started, status):
            self.started, self.status = started, status

        def call(self, method, args=None):
            return self.started if method == "api_compact" else self.status

    def run_command(self, client):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = api.compact_command(client, "/repo/run", sleep=lambda s: None)
        return status, out.getvalue()

    def test_the_command_tells_the_agent_when_to_start_the_subagent(self):
        status, text = self.run_command(self.Client({"status": "no key"}, None))
        self.assertEqual(status, 3)
        self.assertIn("/repo/run api-key", text)
        paused = {"run": 1, "running": False, "kind": "no credit", "problem": "no credit: HTTP 402", "last_run": {"result": "paused: no credit", "lines": 0, "usd": 0.0}}
        status, text = self.run_command(self.Client({"status": "ready", "target": 1}, paused))
        self.assertEqual(status, 3)
        self.assertIn(api.BILLING_URL, text)
        done = {"run": 1, "running": False, "last_run": {"result": "done", "lines": 12, "usd": 0.0021}}
        self.assertEqual(self.run_command(self.Client({"status": "ready", "target": 1}, done)), (0, "OptChat API: done, 12 lines, about $0.0021.\n"))


if __name__ == "__main__":
    unittest.main()
