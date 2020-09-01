# This file is part of the pyBinSim project.
#
# Copyright (c) 2017 A. Neidhardt, F. Klein, N. Knoop, T. Köllmer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Module for interacting with sparkFun 9DOF Razor IMU
Product URL: https://www.sparkfun.com/products/10736
"""

from __future__ import print_function

import re

import serial


def get_intact_reading(sensor_reading, prefix='!ANG:'):
    """
    Return last intact sensor data (i.e., the newest) in a list of readings.
    :param sensor_reading: List of sender readings, e.g. from read_all() split into lines.
    :param prefix: Prefix of sensor reading, e.g. "!ANG"
    :return: CSV string without prefix. None if parsing failed.
    """
    digit_regex = r"(([-]?\d*\.\d+)|[-]?\d+)?"
    reading_regex = r"{}{},{},{}".format(
        prefix, digit_regex, digit_regex, digit_regex)

    for item in reversed(sensor_reading):
        item = item.strip()
        if re.match(reading_regex, item):
            return item[len(prefix):]
    return None


def get_float_values(line):
    """
    Parse csv string with float values to list of floats.
    :param line:  csv string, e.g., ".4,2,-3.4"
    :return: List of floats. Empty list if parsing failed.
    """
    result_list = []
    yrp = line.split(",")

    for x in yrp:
        if x.strip() == "":
            result_list += [0]
        else:
            try:
                result_list += [float(x)]
            except:
                print("Could not parse line: {}", line)
                return []

    return result_list


def parse_sensor_reading(sensor_reading):
    """
    Parses sensor reading and returns list of floats.
    :param sensor_reading: List of sender readings, e.g. from read_all() split into lines.
    :return: List of floats with sensor values. Empty list if parsing failed.
    """
    if len(sensor_reading) == 0:
        return []

    line = get_intact_reading(sensor_reading)

    if not line:
        return []

    result_list = get_float_values(line)
    return result_list


class Spark9dof(object):
    """
    Class allows access to Yaw, Pitch, Roll Data of spark fun's 9DoF Board.
    """

    def __init__(self, com_port='COM4', baudrate=57600):
        """
        Initiializes Serial connection to 9DOF board
        :param com_port: COM Port, defaults to 'COM4'.
        :param baudrate: baudrate, defaults to 57600
        """

        self.com_port = com_port
        self.baudrate = baudrate

        try:
            self.ser = serial.Serial(self.com_port, self.baudrate, timeout=1)
        except serial.SerialException as e:
            raise RuntimeError(e)

    def get_sensor_data(self):
        """
        Return parsed sensor_reading.

        Returns the most recent value in all sensor readings that have been cached
        between two calls to get_sensor_data. Beware that too short polling times might
        result in no values returned, as the buffer does not (yet) contain valid sensor
        readings, but only fragments.
        :return: Sensor reading as list of float.
        """
        sensor_reading = self.ser.read_all()
        reading_list = sensor_reading.decode().split("\r\n")

        return parse_sensor_reading(reading_list)
