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

import numpy as np
import pyfftw
from past.builtins import xrange

nThreads = multiprocessing.cpu_count()


class ConvolverFFTW(object):
    """
    Class for convolving mono (usually for virtual sources) or stereo input (usually for HP compensation)
    with a BRIRsor HRTF
    """

    def __init__(self, ir_size, block_size, process_stereo):

        self.log = logging.getLogger("pybinsim.ConvolverFFTW")
        self.log.info("Convolver: init")

        # Get Basic infos
        self.IR_size = ir_size
        self.block_size = block_size

        # floor (integer) division in python 2 & 3
        self.IR_blocks = self.IR_size // block_size

        # Calculate crossfade windows
        self.crossFadeIn = np.zeros(self.block_size, dtype='float32')
        self.crossFadeOut = np.zeros(self.block_size, dtype='float32')
        self.crossFadeIn[:] = xrange(0, self.block_size)
        # float division in python 2 & 3
        self.crossFadeIn *= 1 / float((self.block_size - 1))
        self.crossFadeOut[:] = np.flipud(self.crossFadeIn)

        # Create default filter and fftw plan
        # Filter format: [nBlocks,blockSize*4]
        # 0 to blockSize*2: left filter
        # blockSize*2 to blockSize*4: right filter
        self.default_filter = pyfftw.zeros_aligned([self.IR_blocks, 2 * (self.block_size + 1)], np.dtype(np.float32))

        self.filter_fftw_plan = pyfftw.builders.rfft(np.zeros(self.block_size * 2), overwrite_input=True,
                                                     planner_effort='FFTW_MEASURE',
                                                     threads=nThreads)

        # Create Input Buffers and create fftw plans
        self.buffer = pyfftw.zeros_aligned(self.block_size * 2, dtype='float32')
        self.bufferFftPlan = pyfftw.builders.rfft(self.buffer, overwrite_input=True,
                                                  planner_effort='FFTW_MEASURE', threads=nThreads)

        self.buffer2 = pyfftw.zeros_aligned(self.block_size * 2, dtype='float32')
        self.buffer2FftPlan = pyfftw.builders.rfft(self.buffer2, overwrite_input=True,
                                                   planner_effort='FFTW_MEASURE', threads=nThreads)

        # Create arrays for the filters and the FDLs.
        # Note: Probably these do not need to be memory aligned
        self.TF_left_blocked = pyfftw.zeros_aligned((self.IR_blocks, self.block_size + 1), dtype='complex64')
        self.TF_right_blocked = pyfftw.zeros_aligned((self.IR_blocks, self.block_size + 1), dtype='complex64')
        self.TF_left_blocked_previous = pyfftw.zeros_aligned((self.IR_blocks, self.block_size + 1), dtype='complex64')
        self.TF_right_blocked_previous = pyfftw.zeros_aligned((self.IR_blocks, self.block_size + 1), dtype='complex64')

        self.FDL_left = pyfftw.zeros_aligned(self.IR_blocks * (self.block_size + 1), dtype='complex64')
        self.FDL_right = pyfftw.zeros_aligned(self.IR_blocks * (self.block_size + 1), dtype='complex64')

        # Arrays for the result of the complex multiply and add
        # These should be memory aligned because ifft is performed with these data
        self.resultLeftFreq = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')
        self.resultRightFreq = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')
        self.resultLeftFreqPrevious = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')
        self.resultRightFreqPrevious = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')
        self.resultLeftIFFTPlan = pyfftw.builders.irfft(np.zeros(self.block_size + 1, dtype='complex64'),
                                                        overwrite_input=False, planner_effort='FFTW_MEASURE',
                                                        threads=nThreads)
        self.resultRightIFFTPlan = pyfftw.builders.irfft(np.zeros(self.block_size + 1, dtype='complex64'),
                                                         overwrite_input=False, planner_effort='FFTW_MEASURE',
                                                         threads=nThreads)
        self.resultLeftPreviousIFFTPlan = pyfftw.builders.irfft(np.zeros(self.block_size + 1, dtype='complex64'),
                                                                overwrite_input=False, planner_effort='FFTW_MEASURE',
                                                                threads=nThreads)
        self.resultRightPreviousIFFTPlan = pyfftw.builders.irfft(np.zeros(self.block_size + 1, dtype='complex64'),
                                                                 overwrite_input=False, planner_effort='FFTW_MEASURE',
                                                                 threads=nThreads)

        # Result of the iffft is stored here
        self.outputLeft = pyfftw.zeros_aligned(self.block_size, dtype='float32')
        self.outputRight = pyfftw.zeros_aligned(self.block_size, dtype='float32')

        # Counts how often process() is called
        self.processCounter = 0

        # Flag for interpolation of output blocks (result of process())
        self.interpolate = False

        # Select mono or stereo processing
        self.processStereo = process_stereo

    def get_counter(self):
        """
        Returns processing counter
        :return: processing counter
        """
        return self.processCounter

    def transform_filter(self, filter):
        """
        Transform filter to freq domain

        :param filter:
        :return: transformed filter
        """
        IR_left = filter[:, 0]
        IR_right = filter[:, 1]

        # Split IRs in blocks
        IR_left_blocked = np.reshape(IR_left, (self.IR_blocks, self.block_size))
        IR_right_blocked = np.reshape(IR_right, (self.IR_blocks, self.block_size))

        # Add zeroes to each block
        IR_left_blocked = np.concatenate((IR_left_blocked, np.zeros([self.IR_blocks, self.block_size])), axis=1)
        IR_right_blocked = np.concatenate((IR_right_blocked, np.zeros([self.IR_blocks, self.block_size])), axis=1)

        TF_left_blocked = np.zeros([self.IR_blocks, self.block_size + 1], np.dtype(np.complex64))
        TF_right_blocked = np.zeros([self.IR_blocks, self.block_size + 1], np.dtype(np.complex64))

        for ir_block_count in range(0, self.IR_blocks):
            TF_left_blocked[ir_block_count] = self.filter_fftw_plan(IR_left_blocked[ir_block_count])
            TF_right_blocked[ir_block_count] = self.filter_fftw_plan(IR_right_blocked[ir_block_count])

        return TF_left_blocked, TF_right_blocked

    def setIR(self, filter, do_interpolation):
        """
        Hand over a new set of filters to the convolver
        and define if you want to perform an interpolation/crossfade

        :param filter:
        :param do_interpolation:
        :return: None
        """
        # Save old filters in case interpolation is needed
        self.TF_left_blocked_previous = self.TF_left_blocked
        self.TF_right_blocked_previous = self.TF_right_blocked

        # apply new filters
        self.TF_left_blocked, self.TF_right_blocked = self.transform_filter(filter)

        # Interpolation means cross fading the output blocks (linear interpolation)
        self.interpolate = do_interpolation

    def process_nothing(self):
        """
        Just for testing
        :return: None
        """
        self.processCounter += 1

    def fill_buffer_mono(self, block):
        """
        Copy mono soundblock to input Buffer;
        Transform to Freq. Domain and store result in FDLs
        :param block: Mono sound block
        :return: None
        """

        if block.size < self.block_size:
            # print('Fill up last block')
            block = np.concatenate((block, np.zeros((1, (self.block_size - block.size)))), 1)

        if self.processCounter == 0:
            # insert first block to buffer
            self.buffer[self.block_size:self.block_size * 2] = block

        else:
            # shift buffer
            self.buffer = np.roll(self.buffer, -self.block_size)
            # insert new block to buffer
            self.buffer[self.block_size:self.block_size * 2] = block
            # shift FDLs
            self.FDL_left = np.roll(self.FDL_left, self.block_size + 1)
            self.FDL_right = np.roll(self.FDL_right, self.block_size + 1)

        # transform buffer into freq domain and copy to FDLs
        self.FDL_left[0:self.block_size + 1] = self.FDL_right[0:self.block_size + 1] = self.bufferFftPlan(
            self.buffer)

    def fill_buffer_stereo(self, block):
        """
        Copy stereo soundblock to input Buffer1 and Buffer2;
        Transform to Freq. Domain and store result in FDLs

        :param block:
        :return: None
        """

        if block.size < self.block_size:
            # print('Fill up last block')
            # print(np.shape(block))
            block = np.concatenate((block, np.zeros(((self.block_size - block.size), 2))), 0)

        if self.processCounter == 0:
            # insert first block to buffer
            self.buffer[self.block_size:self.block_size * 2] = block[:, 0]
            self.buffer2[self.block_size:self.block_size * 2] = block[:, 1]

        else:
            # shift buffer
            self.buffer = np.roll(self.buffer, -self.block_size)
            self.buffer2 = np.roll(self.buffer2, -self.block_size)
            # insert new block to buffer
            self.buffer[self.block_size:self.block_size * 2] = block[:, 0]
            self.buffer2[self.block_size:self.block_size * 2] = block[:, 1]
            # shift FDLs
            self.FDL_left = np.roll(self.FDL_left, self.block_size + 1)
            self.FDL_right = np.roll(self.FDL_right, self.block_size + 1)

        # transform buffer into freq domain and copy to FDLs
        self.FDL_left[0:self.block_size + 1] = self.bufferFftPlan(self.buffer)
        self.FDL_right[0:self.block_size + 1] = self.buffer2FftPlan(self.buffer2)

    def multiply_and_add(self, IR_block_count):
        """
        Multiply Current Filters with data stored in the FDL

        :param IR_block_count:
        :return: None
        """

        if IR_block_count == 0:
            self.resultLeftFreq = np.multiply(self.TF_left_blocked[IR_block_count], self.FDL_left[(
                IR_block_count * (self.block_size + 1)):((IR_block_count + 1) * (self.block_size + 1))])

            self.resultRightFreq = np.multiply(self.TF_right_blocked[IR_block_count], self.FDL_right[(
                IR_block_count * (self.block_size + 1)):((IR_block_count + 1) * (self.block_size + 1))])
        else:
            self.resultLeftFreq += np.multiply(self.TF_left_blocked[IR_block_count],
                                               self.FDL_left[
                                               (IR_block_count * (self.block_size + 1)):(
                                                   (IR_block_count + 1) * (self.block_size + 1))])

            self.resultRightFreq += np.multiply(self.TF_right_blocked[IR_block_count],
                                                self.FDL_right[
                                                (IR_block_count * (self.block_size + 1)):(
                                                    (IR_block_count + 1) * (self.block_size + 1))])

    def multiply_and_add_previous(self, irBlockCount):
        """
        Multiply Previous Filters with data stored in the FDL
        Needed when doing block crossfade/interpolation

        :param irBlockCount:
        :return: None
        """

        if irBlockCount == 0:
            self.resultLeftFreqPrevious = np.multiply(self.TF_left_blocked_previous[irBlockCount], self.FDL_left[(
                irBlockCount * (self.block_size + 1)):((irBlockCount + 1) * (self.block_size + 1))])

            self.resultRightFreqPrevious = np.multiply(self.TF_right_blocked_previous[irBlockCount], self.FDL_right[(
                irBlockCount * (self.block_size + 1)):((irBlockCount + 1) * (self.block_size + 1))])
        else:
            self.resultLeftFreqPrevious += np.multiply(self.TF_left_blocked_previous[irBlockCount],
                                                       self.FDL_left[
                                                       (irBlockCount * (self.block_size + 1)):(
                                                           (irBlockCount + 1) * (self.block_size + 1))])

            self.resultRightFreqPrevious += np.multiply(self.TF_right_blocked_previous[irBlockCount],
                                                        self.FDL_right[
                                                        (irBlockCount * (self.block_size + 1)):(
                                                            (irBlockCount + 1) * (self.block_size + 1))])

    def process(self, block):
        """
        Main function

        :param block:
        :return: (outputLeft, outputRight)
        """
        # print("Convolver: process")

        # First: Fill buffer and FDLs with current block
        if not self.processStereo:
            # print('Convolver Mono Processing')
            self.fill_buffer_mono(block)
        else:
            # print('Convolver Stereo Processing')
            self.fill_buffer_stereo(block)

        # Second: Multiplikation with IR block und accumulation with previous data
        for irBlockCount in xrange(0, self.IR_blocks):
            # Always convolute current filter
            self.multiply_and_add(irBlockCount)

            # Also convolute old filter if interpolation needed
            if self.interpolate:
                self.multiply_and_add_previous(irBlockCount)

        # Third: Transformation back to time domain
        if self.interpolate:
            # fade over full block size
            # print('do block interpolation')
            self.outputLeft = np.multiply(self.resultLeftPreviousIFFTPlan(self.resultLeftFreqPrevious).real[
                                          self.block_size:self.block_size * 2], self.crossFadeOut) + \
                              np.multiply(self.resultLeftIFFTPlan(self.resultLeftFreq).real[
                                          self.block_size:self.block_size * 2], self.crossFadeIn)

            self.outputRight = np.multiply(self.resultRightPreviousIFFTPlan(self.resultRightFreqPrevious).real[
                                           self.block_size:self.block_size * 2], self.crossFadeOut) + \
                               np.multiply(self.resultRightIFFTPlan(self.resultRightFreq).real[
                                           self.block_size:self.block_size * 2], self.crossFadeIn)

        else:
            self.outputLeft = self.resultLeftIFFTPlan(self.resultLeftFreq).real[self.block_size:self.block_size * 2]
            self.outputRight = self.resultRightIFFTPlan(self.resultRightFreq).real[self.block_size:self.block_size * 2]

        self.processCounter += 1
        self.interpolate = False

        return self.outputLeft, self.outputRight

    def close(self):
        print("Convolver: close")
        # TODO: do something here?
