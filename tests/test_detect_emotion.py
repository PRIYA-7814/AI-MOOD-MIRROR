import unittest
import numpy as np

from src import detect_emotion


class TestDetectEmotion(unittest.TestCase):
    def test_get_emotion_fallback(self):
        """Force the module to use the final fallback and ensure output shape/types."""
        # Force fallback path so test doesn't depend on heavy external models
        detect_emotion.USE_BACKEND = "none"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        label, conf = detect_emotion.get_emotion(frame)
        self.assertIsInstance(label, str)
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)


if __name__ == "__main__":
    unittest.main()
