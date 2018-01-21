from unittest import TestCase

from pybinsim.pose import Orientation, Position, Pose


class TestPose(TestCase):
    def test_create_key(self):
        orientation = Orientation('10','20','30')
        position = Position('1','2','3')

        pose = Pose(orientation, position)

        self.assertEqual(orientation.pitch, '20')
        self.assertEqual(position.z, '3')

        self.assertEqual(pose.create_key(), "10,20,30,1,2,3,0,0,0")

    def test_from_filter_value_list_6(self):

        pose = Pose.from_filterValueList([10, 20, 30, 1, 2, 3])
        self.assertTrue(pose.orientation.yaw, 10)
        self.assertTrue(pose.position.x, 1)

    def test_from_filter_value_list_9(self):

        pose = Pose.from_filterValueList([10, 20, 30, 1, 2, 3, 11, 22, 33])
        self.assertTrue(pose.orientation.yaw, 10)
        self.assertTrue(pose.position.x, 1)
        self.assertTrue(pose.custom.b, 22)

    def test_from_filter_value_invalid(self):
        with self.assertRaises(RuntimeError):
            Pose.from_filterValueList([1, 2, 3])