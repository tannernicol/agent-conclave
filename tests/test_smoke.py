import copy
import os
import tempfile
import unittest
from pathlib import Path

from conclave.config import load_config, Config
from conclave.models.ollama import OllamaClient
from conclave.pipeline import ConclavePipeline


class ConclaveSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_config()
        client = OllamaClient()
        models = client.list_models()
        cls.available = {m.get("name") for m in models if m.get("name")}

    def _build_test_config(self) -> Config:
        raw = copy.deepcopy(self.base)
        data_dir = Path(tempfile.mkdtemp(prefix="conclave-test-"))
        raw["data_dir"] = str(data_dir)
        raw.setdefault("models", {})
        raw["models"]["registry_path"] = str(data_dir / "models" / "registry.json")
        raw["models"]["benchmarks_path"] = str(data_dir / "models" / "benchmarks.jsonl")
        raw["models"]["health_path"] = str(data_dir / "models" / "health.json")
        raw.setdefault("index", {})
        raw["index"]["enabled"] = False
        raw.setdefault("calibration", {})
        raw["calibration"]["enabled"] = False
        cards = raw.get("models", {}).get("cards", [])
        filtered = [c for c in cards if c.get("id", "").startswith("ollama:") and c.get("id", "").split(":", 1)[1] in self.available]
        if not filtered:
            strict = str((os.environ.get("CONCLAVE_SMOKE_STRICT") or os.environ.get("CI") or "")).lower() in {"1", "true", "yes"}
            if strict:
                self.fail("No required Ollama models are available. Run: ollama pull qwen2.5-coder:7b")
            raise unittest.SkipTest("Ollama not available; start ollama or set CONCLAVE_SMOKE_STRICT=1")
        raw["models"]["cards"] = filtered
        return Config(raw)

    def test_pipeline_examples(self):
        config = self._build_test_config()
        pipeline = ConclavePipeline(config)
        examples = [
            "Summarize any tax action items for me.",
            "Summarize any health follow-ups I should consider.",
        ]
        for query in examples:
            result = pipeline.run(query)
            self.assertTrue(result.consensus.get("answer"))
            self.assertIn(result.consensus.get("confidence"), {"low", "medium", "high"})


if __name__ == "__main__":
    unittest.main()
