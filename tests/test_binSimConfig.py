from unittest import TestCase

from pybinsim.application import parse_boolean


class TestBinSimConfig(TestCase):
    def test_parse_boolean_regular(self):
        inputs = [True, False, "True", "False", None, "Something Strange", 12]
        expected_outputs = [True, False, True, False, None, None, None]

        for i, test_value in enumerate(inputs):
            output = parse_boolean(test_value)
            self.assertEqual(output, expected_outputs[i], "i={}".format(i))

