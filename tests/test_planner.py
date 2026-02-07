import unittest

from conclave.config import load_config
from conclave.models.registry import ModelRegistry
from conclave.models.planner import Planner


class PlannerTest(unittest.TestCase):
    def test_planner_assigns_roles(self):
        raw = load_config()
        registry = ModelRegistry.from_config(raw.get("models", {}))
        planner = Planner.from_config(raw.get("planner", {}))
        plan = planner.choose_models_for_roles(["router", "reasoner", "critic"], registry.list_models())
        self.assertTrue(plan)
        self.assertIn("reasoner", plan)


if __name__ == "__main__":
    unittest.main()
