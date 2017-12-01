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

import multiprocessing
import logging
import sys

import numpy as np
import pyfftw
from scipy.io.wavfile import read

from pybinsim.utility import pcm2float
from pybinsim.utility import total_size

nThreads = multiprocessing.cpu_count()


class FilterStorage(object):
    """ Class for storing all filters mentioned in the filter list """

    def __init__(self, irSize, block_size, filter_list_name):
        print('FilterStorage: init')

        self.log = logging.getLogger("pybinsim.FilterStorage")
        self.log.info("FilterStorage: init")

        self.ir_size = irSize
        self.ir_blocks = irSize // block_size
        self.block_size = block_size
        # self.filterListPath=os.path.join(os.path.dirname(__file__),filterListName)
        self.filter_list_path = filter_list_name
        self.filter_list = open(self.filter_list_path, 'r')

        # Filter format: [nBlocks,blockSize*4]
        # 0 to blockSize*2: left filter
        # blockSize*2 to blockSize*4: right filter
        self.default_filter = np.zeros([self.ir_blocks, 2 * (block_size + 1)], np.dtype(np.float32))

        self.fftw_plan = pyfftw.builders.rfft(np.zeros(block_size * 2), overwrite_input=True,
                                              planner_effort='FFTW_MEASURE',
                                              threads=nThreads)

        # format: [key,{filterLeft,filterRight}]
        self.filter_dict = {}

        # Start to load filters
        self.load_filters()

    def parse_filter_list(self):
        """
        Generator for filter list lines

        Lines are assumed to have a format like
        0 0 40 1 1 0 brirWav_APA/Ref_A01_1_040.wav

        :return: Iterator of filter_value_list - filter-path tuples
        """

        for line in self.filter_list:

            line_content = line.split()

            filter_value_list = tuple(line_content[0:-1])
            filter_path = line_content[-1]

            yield (filter_value_list, filter_path,)

    def load_filters(self):
        """
        Load filters from files

        :return: None
        """

        self.log.info("Start loading filters")

        for filter_value_list, filter_path in self.parse_filter_list():

            # load filter
            self.log.info('Loading {}'.format(filter_path))

            _, current_filter_pcm = read(filter_path)

            # doubles size (int16 -> float32)
            current_filter = pcm2float(current_filter_pcm, 'float32')

            filter_size = np.shape(current_filter)

            # Fill filter with zeros if to short
            if filter_size[0] < self.ir_size:
                print('Filter to short: Fill up with zeros')
                current_filter = np.concatenate((current_filter, np.zeros((self.ir_size - filter_size[0], 2))), 0)

            # Transform filter to freq domain before storing
            # doubles size in RAM (float32 -> complex64)
            transformed_filter = self.transform_filter(current_filter)

            # create key and store in dict.
            key = self.create_key_from_values(filter_value_list)
            self.filter_dict.update({key: transformed_filter})

        self.log.info("Finished loading filters.")
        self.log.info("filter_dict size: {}MiB".format(total_size(self.filter_dict) // 1024 // 1024))

    def transform_filter(self, filter):
        """
        Transform filter to freq domain

        :param filter:
        :return: transformed filter
        """
        IR_left = filter[:, 0]
        IR_right = filter[:, 1]

        # Split IRs in blocks
        IR_left_blocked = np.reshape(IR_left, (self.ir_blocks, self.block_size))
        IR_right_blocked = np.reshape(IR_right, (self.ir_blocks, self.block_size))

        # Add zeroes to each block
        IR_left_blocked = np.concatenate((IR_left_blocked, np.zeros([self.ir_blocks, self.block_size])), axis=1)
        IR_right_blocked = np.concatenate((IR_right_blocked, np.zeros([self.ir_blocks, self.block_size])), axis=1)

        TF_left_blocked = np.zeros([self.ir_blocks, self.block_size + 1], np.dtype(np.complex64))
        TF_right_blocked = np.zeros([self.ir_blocks, self.block_size + 1], np.dtype(np.complex64))

        for ir_block_count in range(0, self.ir_blocks):
            TF_left_blocked[ir_block_count] = self.fftw_plan(IR_left_blocked[ir_block_count])
            TF_right_blocked[ir_block_count] = self.fftw_plan(IR_right_blocked[ir_block_count])

        # Concatenate left and right filter for storage
        transformed_filter = np.concatenate((TF_left_blocked, TF_right_blocked), axis=1)
        return transformed_filter

    def get_filter(self, filter_value_list):
        """
        Searches in the dict if key is available and return corresponding filter
        When no filter is found, defaultFilter is returned which results in silence

        :param filter_value_list:
        :return: corresponding filter for key
        """

        key = self.create_key_from_values(filter_value_list)

        if key in self.filter_dict:
            print('Filter found: key: ' + key)
            return (self.filter_dict.get(key)[:, 0:self.block_size + 1],
                    self.filter_dict.get(key)[:, (self.block_size + 1):2 * (self.block_size + 1)])
        else:
            print('Filter not found; key: ' + key)
            return (self.default_filter[:, 0:self.block_size + 1],
                    self.default_filter[:, (self.block_size + 1):2 * (self.block_size + 1)])

    def create_key_from_values(self, filter_value_list):
        """
        Just create the key

        :param filter_value_list:
        :return: key created from filter_value_list
        """
        key = ','.join(map(str, filter_value_list))
        return key

    def close(self):
        print('FilterStorage: close()')
        # TODO: do something in here?
