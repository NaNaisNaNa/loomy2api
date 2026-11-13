"""Offline regressions for transparent leases and durable usage history."""

import errno
import io
import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from loomy2api.pool import AccountPool
from loomy2api.server import Gateway, Handler
from loomy2api.usage import UsageLedger
from tests.support import make_config, write_accounts


class TestPassthroughLeases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        cfg = make_config(self.root, "http://127.0.0.1:1/api/v1",
                          max_inflight_per_account=1,
                          proxy="http://127.0.0.1:22222")
        write_accounts(cfg.path("accounts_file"), [
            {"name": "a", "session": "offline-session",
             "proxy": "http://127.0.0.1:33333"}])
        pool = AccountPool(cfg)
        gateway = SimpleNamespace(pool=pool, models_client=mock.Mock(),
                                  usage=UsageLedger())
        handler = SimpleNamespace(
            gateway=gateway, _read_body=lambda: b'{"model":"offline-model"}',
            send_response=mock.Mock(), send_header=mock.Mock(),
            end_headers=mock.Mock(), wfile=io.BytesIO())
        return gateway, handler, pool.get("a")

    def test_both_passthroughs_release_once_and_forward_account_proxy(self):
        for suffix in ("embeddings", "images/generations"):
            with self.subTest(suffix=suffix):
                gateway, handler, account = self.fixture()
                body = ({"data": [], "points_consumed": 7} if "images" in suffix
                        else {"data": [], "usage": {"points_consumed": 7}})
                payload = json.dumps(body).encode("utf-8")
                gateway.models_client.call.return_value = (
                    200, {"Content-Type": "application/json"}, payload)
                with mock.patch.object(gateway.pool, "release",
                                       wraps=gateway.pool.release) as release:
                    Handler._passthrough(handler, suffix)
                    release.assert_called_once_with(account)
                    self.assertEqual(gateway.pool.inflight.snapshot(), {})
                    handler.send_response.assert_called_once_with(200)
                    self.assertEqual(handler.wfile.getvalue(), payload)
                    gateway.models_client.call.assert_called_once_with(
                        account.session, suffix, {"model": "offline-model"},
                        proxy=account.proxy)
                    # A real second request must still get the only account.
                    Handler._passthrough(handler, suffix)
                    self.assertEqual(release.call_args_list,
                                     [mock.call(account), mock.call(account)])
                    self.assertEqual(gateway.pool.inflight.snapshot(), {})
                self.assertEqual(gateway.usage.summary()["windows"]["24h"]
                                 ["totals"]["points"], 14)

    def test_both_passthroughs_release_once_when_upstream_raises(self):
        for suffix in ("embeddings", "images/generations"):
            with self.subTest(suffix=suffix):
                gateway, handler, account = self.fixture()
                gateway.models_client.call.side_effect = OSError("offline failure")
                with mock.patch.object(gateway.pool, "release",
                                       wraps=gateway.pool.release) as release:
                    with self.assertRaisesRegex(OSError, "offline failure"):
                        Handler._passthrough(handler, suffix)
                    release.assert_called_once_with(account)
                gateway.models_client.call.assert_called_once_with(
                    account.session, suffix, {"model": "offline-model"},
                    proxy=account.proxy)
                self.assertEqual(gateway.pool.inflight.snapshot(), {})
                next_account = gateway.pool.acquire()
                try:
                    self.assertIs(next_account, account)
                    self.assertEqual(gateway.pool.inflight.snapshot(), {"a": 1})
                finally:
                    gateway.pool.release(next_account)
                handler.send_response.assert_not_called()


class TestUsagePersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def legacy_fixture(self):
        legacy = self.root / "old-image" / "state" / "usage.jsonl"
        legacy.parent.mkdir(parents=True)
        entry = UsageLedger().record(account="legacy", model="model",
                                     usage={"points_consumed": 7})
        payload = (json.dumps(entry) + "\r\n\n").encode("utf-8")
        legacy.write_bytes(payload)
        return legacy, entry, payload

    def test_gateway_restores_history_across_image_roots(self):
        data = self.root / "external-data"
        data.mkdir()
        cfg = make_config(self.root, "http://127.0.0.1:1/api/v1",
                          accounts_file=str(data / "accounts.json"))
        gateways = []
        for image in ("image-before", "image-after"):
            source = self.root / image / "site-packages" / "loomy2api" / "server.py"
            with mock.patch("loomy2api.server.__file__", str(source)):
                gateways.append(Gateway(cfg, log=mock.Mock()))
            if len(gateways) == 1:
                entry = gateways[0].usage.record(
                    account="a", model="offline-model",
                    usage={"prompt_tokens": 9, "points_consumed": 4})
        target = data / "usage.jsonl"
        self.assertTrue(target.is_file())
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), entry)
        self.assertEqual(gateways[1].usage.recent(), [entry])
        self.assertEqual(gateways[1].usage.summary()["windows"]["24h"]
                         ["totals"]["points"], 4)
        self.assertFalse((self.root / "usage.jsonl").exists())

    def test_legacy_migration_copies_bytes_and_preserves_source(self):
        legacy, entry, payload = self.legacy_fixture()
        target = self.root / "external-data" / "usage.jsonl"
        logger = mock.Mock()
        ledger = UsageLedger(persist_path=target, legacy_path=legacy, logger=logger)
        self.assertEqual(target.read_bytes(), payload)
        self.assertEqual(legacy.read_bytes(), payload)
        self.assertEqual(ledger.recent(), [entry])
        logger.assert_not_called()

    def test_legacy_migration_never_overwrites_existing_target(self):
        legacy, _entry, payload = self.legacy_fixture()
        target = self.root / "usage.jsonl"
        current = UsageLedger().record(account="current", model="model")
        for existing in (b"", (json.dumps(current) + "\n").encode("utf-8")):
            with self.subTest(empty=not existing):
                target.write_bytes(existing)
                ledger = UsageLedger(persist_path=target, legacy_path=legacy)
                self.assertEqual(target.read_bytes(), existing)
                self.assertEqual(legacy.read_bytes(), payload)
                self.assertEqual(ledger.recent(), [current] if existing else [])

    def test_migration_mid_write_enospc_cleans_up_and_retries_after_restart(self):
        legacy, entry, payload = self.legacy_fixture()
        target = self.root / "external-data" / "usage.jsonl"
        factory = tempfile.NamedTemporaryFile
        def fail_mid_write(**kwargs):
            temporary = factory(**kwargs)
            write = temporary.write
            def write_prefix(data):
                write(data[:8])
                temporary.flush()
                raise OSError(errno.ENOSPC, "fixture disk full")
            temporary.write = write_prefix
            return temporary
        logger = mock.Mock()
        with mock.patch("loomy2api.usage.tempfile.NamedTemporaryFile",
                        side_effect=fail_mid_write):
            ledger = UsageLedger(persist_path=target, legacy_path=legacy, logger=logger)
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.iterdir()), [])
        self.assertEqual(legacy.read_bytes(), payload)
        self.assertEqual(ledger.recent(), [entry])
        logger.assert_called_once()
        restarted = UsageLedger(persist_path=target, legacy_path=legacy, logger=logger)
        self.assertEqual(target.read_bytes(), payload)
        self.assertEqual(legacy.read_bytes(), payload)
        self.assertEqual(restarted.recent(), [entry])
        self.assertEqual(list(target.parent.iterdir()), [target])
        logger.assert_called_once()

    def test_failed_migration_retries_before_new_record_then_restores_both(self):
        legacy, old, payload = self.legacy_fixture()
        target = self.root / "external-data" / "usage.jsonl"
        logger = mock.Mock()
        with mock.patch("loomy2api.usage.os.link", side_effect=OSError(errno.ENOSPC, "fixture disk full")):
            ledger = UsageLedger(persist_path=target, legacy_path=legacy, logger=logger)
        self.assertEqual(ledger.recent(), [old])
        new = ledger.record(account="new", usage={"points_consumed": 3})
        self.assertTrue(target.read_bytes().startswith(payload))
        self.assertEqual(json.loads(target.read_bytes()[len(payload):]), new)
        self.assertEqual(legacy.read_bytes(), payload)
        self.assertEqual(ledger.recent(), [new, old])
        restarted = UsageLedger(persist_path=target, legacy_path=legacy, logger=logger)
        self.assertEqual(restarted.recent(), [new, old])
        logger.assert_called_once()

    def test_persistent_migration_failure_cannot_create_target_during_record(self):
        legacy, old, payload = self.legacy_fixture()
        target = self.root / "external-data" / "usage.jsonl"
        logger = mock.Mock()
        with mock.patch("loomy2api.usage.os.link", side_effect=OSError(errno.ENOSPC, "fixture disk full")):
            ledger = UsageLedger(persist_path=target, legacy_path=legacy, logger=logger)
            new = ledger.record(account="new", usage={"points_consumed": 3})
        self.assertFalse(target.exists())
        self.assertEqual(legacy.read_bytes(), payload)
        self.assertEqual(list(target.parent.iterdir()), [])
        self.assertEqual(ledger.recent(), [new, old])
        restarted = UsageLedger(persist_path=target, legacy_path=legacy, logger=logger)
        self.assertEqual(restarted.recent(), [old])
        logger.assert_called_once()

    def test_migration_publish_race_preserves_target_and_cleans_temp(self):
        legacy, _entry, payload = self.legacy_fixture()
        link = os.link
        for existing in (b"", b'{"account":"winner"}\n'):
            with self.subTest(empty=not existing):
                target = self.root / ("race-" + str(len(existing))) / "usage.jsonl"
                def publish(source, destination):
                    self.assertEqual(Path(source).parent, target.parent)
                    self.assertEqual(Path(source).read_bytes(), payload)
                    Path(destination).write_bytes(existing)
                    link(source, destination)
                logger = mock.Mock()
                with mock.patch("loomy2api.usage.os.link", side_effect=publish) as attempt:
                    UsageLedger(persist_path=target, legacy_path=legacy, logger=logger)
                attempt.assert_called_once()
                self.assertEqual(target.read_bytes(), existing)
                self.assertEqual(legacy.read_bytes(), payload)
                self.assertEqual(list(target.parent.iterdir()), [target])
                logger.assert_not_called()

    def test_write_failure_warns_once_and_keeps_records_in_memory(self):
        logger = mock.Mock()
        ledger = UsageLedger(persist_path=self.root / "usage.jsonl", logger=logger)
        with mock.patch("loomy2api.usage.open", create=True,
                        side_effect=PermissionError("fixture target is read-only")):
            first = ledger.record(account="first", usage={"points_consumed": 2})
            second = ledger.record(account="second", usage={"points_consumed": 3})
        self.assertEqual(ledger.recent(), [second, first])
        self.assertEqual(ledger.summary()["windows"]["24h"]["totals"]["points"], 5)
        logger.assert_called_once()
        self.assertEqual(logger.call_args.kwargs.get("level"), "WARN")
        self.assertIn("fixture target is read-only", logger.call_args.args[0])
        self.assertFalse((self.root / "usage.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
