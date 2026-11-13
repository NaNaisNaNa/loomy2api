"""One multiplier contract across catalogue normalization and panel rendering."""

import copy
import json
import shutil
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

from loomy2api import constants as C
from loomy2api.model_metadata import normalize_model_metadata
from loomy2api.panel import Panel


def panel_view(models):
    gateway = SimpleNamespace(cfg={"default_model": "free"},
                              catalogue=lambda: models, log=lambda *_a: None)
    return Panel(gateway).models_view()


class TestModelMetadata(unittest.TestCase):
    def test_explicit_valid_rates_including_zero_win_over_name(self):
        for raw, expected in [(0, 0), (0.0, 0), (3, 3), (1.25, 1.25),
                              ("0", 0), (" 0.8 ", 0.8), (".5", 0.5),
                              ("2e1", 20)]:
            with self.subTest(raw=raw):
                model = {"name": "Spark-X2.6 (x9.0)", "multiplier": raw}
                result = normalize_model_metadata(model)
                self.assertEqual(result["multiplier"], expected)
                self.assertIsInstance(result["multiplier"], (int, float))

    def test_explicit_annotations_support_x_times_and_fullwidth_parentheses(self):
        for name, expected in [("Spark-X2.6 正式版(限时免费)(x0.0)", 0),
                               ("Model （x3.0）", 3), ("Model (×0.8)", 0.8),
                               ("Model （ × 1.25 ）", 1.25),
                               ("Model ( X4 )", 4)]:
            for raw in ({}, {"multiplier": None}, {"multiplier": ""}):
                with self.subTest(name=name, raw=raw):
                    result = normalize_model_metadata(dict(raw, name=name))
                    self.assertEqual(result["multiplier"], expected)

    def test_versions_and_malformed_annotations_remain_unknown(self):
        for name in ["Spark-X2.6 正式版", "Model x2.6", "Model (free x0.0)",
                     "Model (x2.6 beta)", "Model (x-1)", "Model (xNaN)",
                     "Model (x2）", "Model （x2)", "Model (x1.2.3)"]:
            with self.subTest(name=name):
                result = normalize_model_metadata({"name": name, "multiplier": None})
                self.assertIsNone(result["multiplier"])

    def test_invalid_values_never_become_rates_without_an_annotation(self):
        values = [None, "", " ", True, False, -1, "-1", float("nan"),
                  float("inf"), float("-inf"), "NaN", "Infinity", "0x10",
                  "1_0", "1.2.3", [], {}, 10 ** 400]
        for raw in values:
            with self.subTest(raw=raw):
                model = {"name": "Spark-X2.6", "multiplier": raw}
                result = normalize_model_metadata(model)
                self.assertIsNone(result["multiplier"])
                json.dumps(result, allow_nan=False)
                model["name"] += " (x0.0)"
                self.assertEqual(normalize_model_metadata(model)["multiplier"], 0)

    def test_overflowing_annotation_stays_unknown(self):
        model = {"name": "Model (x" + "9" * 400 + ")", "multiplier": None}
        self.assertIsNone(normalize_model_metadata(model)["multiplier"])

    def test_copy_preserves_source_record_and_other_metadata(self):
        model = {"id": "spark", "name": "Spark-X2.6 (x0.0)",
                 "multiplier": None, "context_length": 32768,
                 "capabilities": {"reasoning": True, "vision": True},
                 "upstream_extra": {"label": "kept"}}
        original = copy.deepcopy(model)
        result = normalize_model_metadata(model)
        self.assertIsNot(result, model)
        self.assertEqual(model, original)
        self.assertEqual(result["multiplier"], 0)
        self.assertEqual({k: v for k, v in result.items() if k != "multiplier"},
                         {k: v for k, v in model.items() if k != "multiplier"})
        self.assertEqual(normalize_model_metadata(result), result)

    def test_unknown_absent_field_preserves_upstream_shape(self):
        model = {"id": "unknown", "name": "Spark-X2.6"}
        result = normalize_model_metadata(model)
        self.assertEqual(result, model)
        self.assertIsNot(result, model)
        self.assertIsNone(result.get("multiplier"))


class TestPanelModelMetadata(unittest.TestCase):
    def test_panel_uses_same_rules_without_mutating_fallback(self):
        models = [{"id": "unknown", "name": "Spark-X2.6", "multiplier": None},
                  {"id": "paid", "name": "Paid", "multiplier": "0.8"},
                  {"id": "free", "name": "Spark-X2.6 (x9)", "multiplier": 0,
                   "capabilities": {"context_length": 8192, "reasoning": True,
                                    "vision": True, "input_modalities": ["text"]}},
                  {"id": "inferred", "name": "Model （×0.1）"}, None]
        original = copy.deepcopy(models)
        payload = panel_view(models)
        rows = payload["models"]
        self.assertEqual([r["id"] for r in rows], ["free", "inferred", "paid", "unknown"])
        self.assertEqual([r["multiplier"] for r in rows], [0, 0.1, 0.8, None])
        self.assertEqual(rows[0]["context_length"], 8192)
        self.assertTrue(rows[0]["is_default"])
        self.assertTrue(rows[0]["reasoning"])
        self.assertTrue(rows[0]["vision"])
        self.assertEqual(rows[0]["modalities"], ["text"])
        self.assertEqual(models, original)

    def test_panel_filters_invalid_values_before_serialization_and_sorting(self):
        rows = panel_view([{"id": "invalid", "name": "Model", "multiplier": True},
                           {"id": "inferred", "name": "Model (×0)",
                            "multiplier": float("nan")}])["models"]
        self.assertEqual(rows[0]["multiplier"], 0)
        self.assertIsNone(rows[1]["multiplier"])
        json.dumps(rows, allow_nan=False)

    def test_builtin_catalogue_keeps_existing_rates_and_labels(self):
        rows = panel_view(C.FALLBACK_MODELS)["models"]
        expected = {m["id"]: m for m in C.FALLBACK_MODELS}
        self.assertEqual(len(rows), len(expected))
        for row in rows:
            self.assertEqual(row["name"], expected[row["id"]]["name"])
            self.assertEqual(row["multiplier"], expected[row["id"]]["multiplier"])

    @unittest.skipUnless(shutil.which("node"), "Node required for panel rendering probe")
    def test_real_frontend_shows_zero_in_table_and_dropdown_and_never_guesses(self):
        payload = panel_view([
            {"id": "free", "name": "Free (x9)", "multiplier": 0},
            {"id": "paid", "name": "Paid", "multiplier": "0.8"},
            {"id": "unknown", "name": "Spark-X2.6", "multiplier": None}])
        app = Path(__file__).resolve().parents[1] / "loomy2api" / "web" / "app.js"
        script = r"""
const fs = require('fs'), vm = require('vm');
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, { value: '', innerHTML: '', textContent: '' });
  return elements.get(id);
};
const context = {
  localStorage: { getItem: () => '' },
  document: { getElementById: element },
  fetch: async path => ({ ok: true, json: async () =>
    path.includes('/points') ? { accounts: [] } : payload }),
};
vm.createContext(context);
const source = fs.readFileSync(process.argv[1], 'utf8');
vm.runInContext(source.replace(/\binit\(\);\s*$/, ''), context);
vm.runInContext('loadModels()', context).then(() => process.stdout.write(JSON.stringify({
  rows: element('mdRows').innerHTML, options: element('pgModel').innerHTML,
}))).catch(error => { console.error(error); process.exitCode = 1; });
"""
        proc = subprocess.run([shutil.which("node"), "-e", script, str(app)],
                              input=json.dumps(payload), encoding="utf-8",
                              capture_output=True, check=True, timeout=15)
        html = json.loads(proc.stdout)
        self.assertIn("free（x0）", html["options"])
        self.assertIn("paid（x0.8）", html["options"])
        self.assertIn('>unknown</option>', html["options"])
        self.assertIn('>x0</span>', html["rows"])
        unknown_row = next(r for r in html["rows"].split("</tr>") if '<b>unknown</b>' in r)
        self.assertIn('>—</span>', unknown_row)
        self.assertNotIn('>x2.6</span>', unknown_row)


if __name__ == "__main__":
    unittest.main()
