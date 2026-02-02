import unittest

from conclave.sources import extract_text


class SourceTests(unittest.TestCase):
    def test_extract_text(self):
        html = "<html><head><title>Title</title></head><body><h1>Hello</h1><p>World</p></body></html>"
        text = extract_text(html)
        self.assertIn("Hello", text)
        self.assertIn("World", text)


if __name__ == "__main__":
    unittest.main()
