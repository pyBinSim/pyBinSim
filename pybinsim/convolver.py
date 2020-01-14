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


nThreads = multiprocessing.cpu_count()


class ConvolverFFTW(object):
    """
    Class for convolving mono (usually for virtual sources) or stereo input (usually for HP compensation)
    with a BRIRsor HRTF
    """

    def __init__(self, ir_size, block_size, process_stereo, useSplittedFilters = False, lateReverbSize = 0):

        self.log = logging.getLogger("pybinsim.ConvolverFFTW")
        self.log.info("Convolver: Start Init")

        # pyFFTW Options
        pyfftw.interfaces.cache.enable()
        #self.fftw_planning_effort='FFTW_MEASURE'
        self.fftw_planning_effort ='FFTW_PATIENT'

        # Get Basic infos
        self.IR_size = ir_size
        self.block_size = block_size
        self.reverbSize = lateReverbSize
        self.late_IR_blocks = 0

        self.useSplittedFilters = useSplittedFilters
        if self.useSplittedFilters:
            # Needed for concatenating filters
            self.late_IR_blocks = self.reverbSize // block_size
            # Size used for convolution changes
            self.IR_size += self.reverbSize

        self.late_IR_left_blocked = None
        self.late_IR_right_blocked = None

        # floor (integer) division in python 2 & 3
        self.IR_blocks = self.IR_size // block_size

        # Calculate LINEAR crossfade windows
        #self.crossFadeIn = np.array(range(0, self.block_size), dtype='float32')
        #self.crossFadeIn *= 1 / float((self.block_size - 1))
        #self.crossFadeOut = np.flipud(self.crossFadeIn)

        # Calculate COSINE-Square crossfade windows
        self.crossFadeOut = np.array(range(0, self.block_size), dtype='float32')
        self.crossFadeOut = np.square(np.cos(self.crossFadeOut/(self.block_size-1)*(np.pi/2)))
        self.crossFadeIn = np.flipud(self.crossFadeOut)

        # Filter format: [nBlocks,blockSize*2]

        # Create Input Buffers and create fftw plans. These need to be memory aligned, because they are transformed to
        # freq domain regularly
        self.buffer = pyfftw.zeros_aligned(self.block_size * 2, dtype='float32')
        self.bufferFftPlan = pyfftw.builders.rfft(self.buffer, overwrite_input=True, threads=nThreads,
                                                     planner_effort=self.fftw_planning_effort,avoid_copy=True)

        self.buffer2 = pyfftw.zeros_aligned(self.block_size * 2, dtype='float32')
        self.buffer2FftPlan = pyfftw.builders.rfft(self.buffer2, overwrite_input=True, threads=nThreads,
                                                     planner_effort=self.fftw_planning_effort,avoid_copy=True)

        # Create arrays for the filters and the FDLs.
        if self.useSplittedFilters:
            self.TF_late_left_blocked = np.zeros((self.late_IR_blocks, self.block_size + 1), dtype='complex64')
            self.TF_late_right_blocked = np.zeros((self.late_IR_blocks, self.block_size + 1), dtype='complex64')

        self.TF_left_blocked = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')
        self.TF_right_blocked = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')
        self.TF_left_blocked_previous = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')
        self.TF_right_blocked_previous = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')

        self.filter_fftw_plan = pyfftw.builders.rfft(np.zeros(self.block_size),n=self.block_size * 2, overwrite_input=True,
                                                     threads=nThreads, planner_effort=self.fftw_planning_effort,
                                                     avoid_copy=False)

        self.FDL_left = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')
        self.FDL_right = np.zeros((self.IR_blocks, self.block_size + 1), dtype='complex64')

        # Arrays for the result of the complex multiply and add
        # These should be memory aligned because ifft is performed with these data
        self.resultLeftFreq = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')
        self.resultRightFreq = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')
        self.resultLeftFreqPrevious = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')
        self.resultRightFreqPrevious = pyfftw.zeros_aligned(self.block_size + 1, dtype='complex64')
        self.resultLeftIFFTPlan = pyfftw.builders.irfft(self.resultLeftFreq,
                                                        overwrite_input=True, threads=nThreads,
                                                        planner_effort=self.fftw_planning_effort,avoid_copy=True)
        self.resultRightIFFTPlan = pyfftw.builders.irfft(self.resultRightFreq,
                                                         overwrite_input=True, threads=nThreads,
                                                         planner_effort=self.fftw_planning_effort,avoid_copy=True)
        self.resultLeftPreviousIFFTPlan = pyfftw.builders.irfft(self.resultLeftFreqPrevious ,
                                                                overwrite_input=True, threads=nThreads,
                                                                planner_effort=self.fftw_planning_effort,avoid_copy=True)
        self.resultRightPreviousIFFTPlan = pyfftw.builders.irfft(self.resultRightFreqPrevious,
                                                                 overwrite_input=True, threads=nThreads,
                                                                 planner_effort=self.fftw_planning_effort,avoid_copy=True)

        # Result of the ifft is stored here
        self.outputLeft = np.zeros(self.block_size, dtype='float32')
        self.outputRight = np.zeros(self.block_size, dtype='float32')

        # Counts how often process() is called
        self.processCounter = 0

        # Flag for interpolation of output blocks (result of process())
        self.interpolate = False

        # Select mono or stereo processing
        self.processStereo = process_stereo

        self.log.info("Convolver: Finished Init")

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

        # Get blocked IRs
        IR_left_blocked, IR_right_blocked = filter.getFilter()

        # TODO: Remove for loop
        # Update early part of filter
        for ir_block_count in range(0, self.IR_blocks-self.late_IR_blocks):
            self.TF_left_blocked[ir_block_count,:] = self.filter_fftw_plan(IR_left_blocked[ir_block_count])
            self.TF_right_blocked[ir_block_count,:] = self.filter_fftw_plan(IR_right_blocked[ir_block_count])

        # Attach late part
        if self.useSplittedFilters:
            self.TF_left_blocked[(self.late_IR_blocks-1):,:] = self.TF_late_left_blocked[:(self.IR_blocks-1),:]
            self.TF_right_blocked[(self.late_IR_blocks-1):,:] = self.TF_late_right_blocked[:(self.IR_blocks-1),:]

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
        self.transform_filter(filter)

        # Interpolation means cross fading the output blocks (linear interpolation)
        self.interpolate = do_interpolation

    def setLateReverb(self, filter):
        """
        Hand over latereverb filter to the convolver

        :param filter:
        :return: None
        """
        self.late_IR_left_blocked, self.late_IR_right_blocked = filter.getFilter()

        for ir_block_count in range(0,self.late_IR_blocks):
            self.TF_late_left_blocked[ir_block_count,:] = self.filter_fftw_plan(self.late_IR_left_blocked[ir_block_count])
            self.TF_late_right_blocked[ir_block_count,:] = self.filter_fftw_plan(self.late_IR_right_blocked[ir_block_count])


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
            self.buffer[self.block_size:] = block

        else:
            # shift buffer
            self.buffer[:self.block_size] = self.buffer[self.block_size:]
            # insert new block to buffer
            self.buffer[self.block_size:] = block
            # shift FDLs
            self.FDL_left = np.roll(self.FDL_left, 1, axis=0)
            self.FDL_right = np.roll(self.FDL_right, 1, axis=0)

            # transform buffer into freq domain and copy to FDLs
        self.FDL_left[0,] = self.FDL_right[0,] = self.bufferFftPlan(self.buffer)

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
            self.buffer[self.block_size:] = block[:, 0]
            self.buffer2[self.block_size:] = block[:, 1]

        else:
            # shift buffer
            self.buffer[:self.block_size] = self.buffer[self.block_size:]
            self.buffer2[:self.block_size] = self.buffer2[self.block_size:]
            # insert new block to buffer
            self.buffer[self.block_size:] = block[:, 0]
            self.buffer2[self.block_size:] = block[:, 1]
            # shift FDLs
            self.FDL_left = np.roll(self.FDL_left, 1, axis=0)
            self.FDL_right = np.roll(self.FDL_right, 1, axis=0)

        # transform buffer into freq domain and copy to FDLs
        self.FDL_left[0,] = self.bufferFftPlan(self.buffer)
        self.FDL_right[0,] = self.buffer2FftPlan(self.buffer2)

    def process(self, block):
        """
        Main function

        :param block:
        :return: (outputLeft, outputRight)
        """

        # First: Fill buffer and FDLs with current block
        if not self.processStereo:
            # print('Convolver Mono Processing')
            self.fill_buffer_mono(block)
        else:
            # print('Convolver Stereo Processing')
            self.fill_buffer_stereo(block)

        # Second: Multiplikation with IR block und accumulation with previous data
        self.resultLeftFreq[:] = np.sum(np.multiply(self.TF_left_blocked,self.FDL_left),axis=0)
        self.resultRightFreq[:] = np.sum(np.multiply(self.TF_right_blocked,self.FDL_right),axis=0)

        # Also convolute old filter if interpolation needed
        if self.interpolate:
            self.resultLeftFreqPrevious[:] = np.sum(np.multiply(self.TF_left_blocked_previous, self.FDL_left), axis=0)
            self.resultRightFreqPrevious[:] = np.sum(np.multiply(self.TF_right_blocked_previous, self.FDL_right), axis=0)

        # Third: Transformation back to time domain
        self.outputLeft = self.resultLeftIFFTPlan()[self.block_size:self.block_size * 2]
        self.outputRight = self.resultRightIFFTPlan()[self.block_size:self.block_size * 2]

        if self.interpolate:
            # fade over full block size
            # print('do block interpolation')
            self.outputLeft = np.add(np.multiply(self.outputLeft,self.crossFadeIn),
                                     np.multiply(self.resultLeftPreviousIFFTPlan()[self.block_size:self.block_size * 2],
                                                 self.crossFadeOut))
            self.outputRight = np.add(np.multiply(self.outputRight,self.crossFadeIn),
                                      np.multiply(
                                          self.resultRightPreviousIFFTPlan()[self.block_size:self.block_size * 2],
                                          self.crossFadeOut))

        self.processCounter += 1
        self.interpolate = False

        return self.outputLeft, self.outputRight

    def close(self):
        print("Convolver: close")
        # TODO: do something here?
