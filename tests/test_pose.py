from unittest import TestCase

from pybinsim.pose import Orientation, Position, Pose


class TestPose(TestCase):
    def test_create_key(self):
        orientation = Orientation('10','20','30')
        position = Position('1','2','3')

        pose = Pose(orientation, position)

        self.assertEqual(orientation.pitch, '20')
        self.assertEqual(position.z, '3')

        self.assertEqual(pose.create_key(), "10,20,30,1,2,3")

    def test_from_filter_value_list(self):

        pose = Pose.from_filterValueList([10, 20, 30, 1, 2, 3])
        self.assertTrue(pose.orientation.yaw, 10)
        self.assertTrue(pose.position.x, 1)
