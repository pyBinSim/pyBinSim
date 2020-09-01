import logging
from collections import namedtuple

logger = logging.getLogger("pybinsim.Pose")


class Orientation(namedtuple('Orientation', ['yaw', 'pitch', 'roll'])):
    pass


class Position(namedtuple('Position', ['x', 'y', 'z'])):
    pass


class Custom(namedtuple('CustomValues', ['a', 'b', 'c'])):
    pass


class Pose:
    def __init__(self, orientation, position, custom=Custom(0, 0, 0)):
        self.orientation = orientation
        self.position = position
        self.custom = custom

    def create_key(self):
        value_list = list(self.orientation) + \
            list(self.position) + list(self.custom)

        return ','.join([str(x) for x in value_list])

    @staticmethod
    def from_filterValueList(filter_value_list):

        # 'old' format: orientation - position
        if len(filter_value_list) == 6:
            orientation = Orientation(
                filter_value_list[0], filter_value_list[1], filter_value_list[2])
            position = Position(
                filter_value_list[3], filter_value_list[4], filter_value_list[5])

            return Pose(orientation, position)

        # 'new' format: orientation - position - custom
        if len(filter_value_list) == 9:
            orientation = Orientation(
                filter_value_list[0], filter_value_list[1], filter_value_list[2])
            position = Position(
                filter_value_list[3], filter_value_list[4], filter_value_list[5])
            custom = Custom(
                filter_value_list[6], filter_value_list[7], filter_value_list[8])

            return Pose(orientation, position, custom)

        raise RuntimeError(
            "Unable to parse filter list: {}".format(filter_value_list))
