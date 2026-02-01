import tempfile
import unittest
from pathlib import Path

from conclave.scheduler import apply_schedule, slugify, render_service, render_timer


class SchedulerTests(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Tax Checkup"), "tax-checkup")
        self.assertEqual(slugify("  "), "topic")

    def test_render_units(self):
        service = render_service("tax-checkup", "/usr/bin/python3", Path("/home/tanner/conclave"))
        self.assertIn("ExecStart=/usr/bin/python3 -m conclave.cli reconcile --topic tax-checkup", service)
        timer = render_timer("tax-checkup", "weekly")
        self.assertIn("OnCalendar=weekly", timer)

    def test_apply_schedule_dry_run(self):
        topics = [{"id": "tax-checkup", "schedule": "weekly"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = apply_schedule(
                topics,
                unit_dir=Path(tmpdir),
                dry_run=True,
                reload_systemd=False,
                validate=False,
            )
            self.assertEqual(len(result["created"]), 1)
            self.assertEqual(result["errors"], [])


if __name__ == "__main__":
    unittest.main()
