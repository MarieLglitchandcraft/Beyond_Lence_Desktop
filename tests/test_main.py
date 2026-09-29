import unittest

import numpy as np

from main import _read_first_valid_frame, parse_arguments


class FakeCamera:
    def __init__(self, frames):
        self.frames = iter(frames)

    def read(self):
        item = next(self.frames)
        if isinstance(item, Exception):
            raise item
        return item


class MainTests(unittest.TestCase):
    def test_fullscreen_is_the_default(self):
        arguments = parse_arguments([])

        self.assertFalse(arguments.windowed)
        self.assertFalse(arguments.fullscreen)

    def test_window_modes_can_be_selected_explicitly(self):
        self.assertTrue(parse_arguments(["--windowed"]).windowed)
        self.assertTrue(parse_arguments(["--fullscreen"]).fullscreen)

    def test_window_modes_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            parse_arguments(["--windowed", "--fullscreen"])

    def test_first_frame_reader_recovers_from_temporary_failures(self):
        expected = np.full((20, 30, 3), 99, dtype=np.uint8)
        camera = FakeCamera([None, RuntimeError("kurzer Aussetzer"), expected])

        frame, reason = _read_first_valid_frame(camera, attempts=3, delay=0)

        np.testing.assert_array_equal(frame, expected)
        self.assertEqual(reason, "ok")

    def test_first_frame_reader_stops_after_bounded_attempts(self):
        camera = FakeCamera([None, None])

        frame, reason = _read_first_valid_frame(camera, attempts=2, delay=0)

        self.assertIsNone(frame)
        self.assertIn("None", reason)


if __name__ == "__main__":
    unittest.main()
