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

import logging
import multiprocessing
from pathlib import Path

import numpy as np
import soundfile as sf

from pybinsim.pose import Pose
from pybinsim.utility import total_size

nThreads = multiprocessing.cpu_count()


class Filter(object):

    def __init__(self, inputfilter, irBlocks, block_size, filename=None):

        self.IR_left_blocked = np.reshape(
            inputfilter[:, 0], (irBlocks, block_size))
        self.IR_right_blocked = np.reshape(
            inputfilter[:, 1], (irBlocks, block_size))
        self.filename = filename

    def getFilter(self):
        return self.IR_left_blocked, self.IR_right_blocked


class FilterStorage(object):
    """ Class for storing all filters mentioned in the filter list """

    def __init__(self, irSize, block_size, filter_list_name):

        self.log = logging.getLogger("pybinsim.FilterStorage")
        self.log.info("FilterStorage: init")

        self.ir_size = irSize
        self.ir_blocks = irSize // block_size
        self.block_size = block_size
        self.default_filter = Filter(
            np.zeros((self.ir_size, 2), dtype='float32'), self.ir_blocks, self.block_size)

        self.filter_list_path = filter_list_name
        self.filter_list = open(self.filter_list_path, 'r')

        self.headphone_filter = None

        # format: [key,{filter}]
        self.filter_dict = {}

        # Start to load filters
        self.load_filters()

    def parse_filter_list(self):
        """
        Generator for filter list lines

        Lines are assumed to have a format like
        0 0 40 1 1 0 brirWav_APA/Ref_A01_1_040.wav

        The headphone filter starts with HPFILTER instead of the positions.

        Lines can be commented with a '#' as first character.

        :return: Iterator of (Pose, filter-path) tuples
        """

        for line in self.filter_list:

            # comment out lines in the list with a '#'
            if line.startswith('#') or line == "\n":
                continue

            line_content = line.split()
            filter_path = line_content[-1]

            if line.startswith('HPFILTER'):
                self.log.info(
                    "Loading headphone filter: {}".format(filter_path))
                self.headphone_filter = Filter(self.load_filter(
                    filter_path), self.ir_blocks, self.block_size)
                continue

            filter_value_list = tuple(line_content[0:-1])

            pose = Pose.from_filterValueList(filter_value_list)

            yield pose, filter_path

    def load_filters(self):
        """
        Load filters from files

        :return: None
        """

        self.log.info("Start loading filters...")
        parsed_filter_list = list(self.parse_filter_list())

        # check if all files are available
        are_files_missing = False
        for pose, filter_path in parsed_filter_list:
            fn_filter = Path(filter_path)
            if not fn_filter.exists():
                self.log.warn(f'Wavefile not found: {fn_filter}')
                are_files_missing = True
        if are_files_missing:
            raise FileNotFoundError("Some files are missing")

        for pose, filter_path in parsed_filter_list:
            self.log.debug('Loading {}'.format(filter_path))

            loaded_filter = self.load_filter(filter_path)
            current_filter = Filter(
                loaded_filter, self.ir_blocks, self.block_size, filename=filter_path)

            # create key and store in dict.
            key = pose.create_key()
            self.filter_dict.update({key: current_filter})

        self.log.info("Finished loading filters.")
        #self.log.info("filter_dict size: {}MiB".format(total_size(self.filter_dict) // 1024 // 1024))

    def get_filter(self, pose):
        """
        Searches in the dict if key is available and return corresponding filter
        When no filter is found, defaultFilter is returned which results in silence

        :param pose
        :return: corresponding filter for pose
        """

        key = pose.create_key()

        if key in self.filter_dict:
            self.log.info("Filter found: key: {}".format(key))
            result_filter = self.filter_dict.get(key)
            if result_filter.filename is not None:
                self.log.info("   use file:: {}".format(result_filter.filename))
            return result_filter
        else:
            self.log.warning('Filter not found: key: {}'.format(key))
            return self.default_filter

    def close(self):
        self.log.info('FilterStorage: close()')
        # TODO: do something in here?

    def get_headphone_filter(self):
        if self.headphone_filter is None:
            raise RuntimeError("Headphone filter not loaded")

        return self.headphone_filter

    def load_filter(self, filter_path):

        current_filter, fs = sf.read(filter_path, dtype='float32')

        filter_size = np.shape(current_filter)

        # Fill filter with zeros if to short
        if filter_size[0] < self.ir_size:
            self.log.warning('Filter too short: Fill up with zeros')
            current_filter = np.concatenate((current_filter, np.zeros(
                (self.ir_size - filter_size[0], 2), np.float32)), 0)
        if filter_size[0] > self.ir_size:
            self.log.warning('Filter too long: shorten')
            current_filter = current_filter[:self.ir_size]

        return current_filter
