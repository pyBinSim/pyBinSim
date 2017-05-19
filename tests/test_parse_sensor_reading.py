from unittest import TestCase

from pybinsim.spark_fun import parse_sensor_reading, get_intact_reading, get_float_values


class TestParse_sensor_reading(TestCase):
    def test_parse_sensor_reading(self):
        test_readings = [
            ['!ANG:-0.37,0.50,-42.74\r', '!ANG:-0.37,0.49,-43.67\r', '!ANG:-0.42,0.47,-44.72\r', '!ANG:-0.37,0.4'],
            ['6,-45.73\r', '!ANG:-0.30,0.48,-46.81\r', '!ANG:-0.31,0.47,-47.85\r', '!ANG:-0.33,0.48,-48.92\r', ''],
            ['!ANG:-0.33,0.47,-50.03\r', '!ANG:-0.36,0.44,-51.10\r', '!ANG:-0.38,0.45,-52.25\r', ''],
            ['374\r', '377\r', '374\r', '-172\r', '3\r', '-467\r', ''],
            ['']
        ]

        expected_results = [
            [-0.42, 0.47, -44.72],
            [-0.33, 0.48, -48.92],
            [-0.38, 0.45, -52.25],
            [],
            []
        ]

        results = [parse_sensor_reading(x) for x in test_readings]

        for i, result in enumerate(results):
            self.assertCountEqual(expected_results[i], result)

    def test_get_last_intact_reading(self):
        test_readings = [
            ['!ANG:-0.37,0.50,-42.74\r', '!ANG:-0.37,0.49,-43.67\r', '!ANG:-0.42,0.47,-44.72\r', '!ANG:-0.37,0.4'],
            ['!ANG:-0.37,0.50,-42.74\r', '!ANG:-0.37,0.49,-43.67\r', '!ANG:0,1,2\r', '!ANG:-0.37,0.4'],
            ['6,-45.73\r', '!ANG:-0.30,0.48,-46.81\r', '!ANG:-0.31,0.47,-47.85\r', '!ANG:-0.33,0.48,-48.92\r', ''],
            ['!ANG:-0.33,0.47,-50.03\r', '!ANG:-0.36,0.44,-51.10\r', '!ANG:-0.38,0.45,-52.25\r', ''],
            ['374\r', '377\r', '374\r', '-172\r', '3\r', '-467\r', ''],
            ['']
        ]

        expected_results = [
            "-0.42,0.47,-44.72",
            "0,1,2",
            "-0.33,0.48,-48.92",
            "-0.38,0.45,-52.25",
            None,
            None

        ]

        results = [get_intact_reading(x) for x in test_readings]

        self.assertCountEqual(expected_results, results)

    def test_get_float_values(self):
        test_readings = [
            '-0.37,0.50,-42.74',
            '-0.37,,-42.74',
            '.37,0.50,-42.74',
            '0,1,2'
        ]

        expected_results = [
            [-0.37, 0.5, -42.74],
            [-0.37, 0, -42.74],
            [0.37, 0.5, -42.74],
            [0, 1, 2]
        ]

        results = [get_float_values(x) for x in test_readings]

        for i, result in enumerate(results):
            self.assertCountEqual(expected_results[i], result)
