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

""" Module contains main loop and configuration of pyBinSim """
import logging
import time

import numpy as np
import sounddevice as sd

from pybinsim.convolver import ConvolverFFTW
from pybinsim.filterstorage import FilterStorage
from pybinsim.osc_receiver import OscReceiver
from pybinsim.pose import Pose
from pybinsim.soundhandler import SoundHandler


def parse_boolean(any_value):

    if type(any_value) == bool:
        return any_value

    # str -> bool
    if any_value == 'True':
        return True
    if any_value == 'False':
        return False

    return None


class BinSimConfig(object):
    def __init__(self):

        self.log = logging.getLogger("pybinsim.BinSimConfig")

        # Default Configuration
        self.configurationDict = {'soundfile': '',
                                  'blockSize': 256,
                                  'filterSize': 16384,
                                  'filterList': 'brirs/filter_list_kemar5.txt',
                                  'enableCrossfading': False,
                                  'useHeadphoneFilter': False,
                                  'loudnessFactor': float(1),
                                  'maxChannels': 8,
                                  'samplingRate': 44100,
                                  'loopSound': True}

    def read_from_file(self, filepath):
        config = open(filepath, 'r')

        for line in config:
            line_content = str.split(line)
            key = line_content[0]
            value = line_content[1]

            if key in self.configurationDict:
                config_value_type = type(self.configurationDict[key])

                if config_value_type is bool:
                    # evaluate 'False' to False
                    boolean_config = parse_boolean(value)

                    if boolean_config is None:
                        self.log.warning(
                            "Cannot convert {} to bool. (key: {}".format(value, key))

                    self.configurationDict[key] = boolean_config
                else:
                    # use type(str) - ctors of int, float, ...
                    self.configurationDict[key] = config_value_type(value)

            else:
                self.log.warning('Entry ' + key + ' is unknown')

    def get(self, setting):
        return self.configurationDict[setting]


class BinSim(object):
    """
    Main pyBinSim program logic
    """

    def __init__(self, config_file):

        self.log = logging.getLogger("pybinsim.BinSim")
        self.log.info("BinSim: init")

        # Read Configuration File
        self.config = BinSimConfig()
        self.config.read_from_file(config_file)

        self.nChannels = self.config.get('maxChannels')
        self.sampleRate = self.config.get('samplingRate')
        self.blockSize = self.config.get('blockSize')

        self.result = None
        self.block = None
        self.stream = None

        self.convolverWorkers = []
        self.convolverHP, self.convolvers, self.filterStorage, self.oscReceiver, self.soundHandler = self.initialize_pybinsim()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__cleanup()

    def stream_start(self):
        self.log.info("BinSim: stream_start")
        try:
            self.stream = sd.OutputStream(samplerate=self.sampleRate,
                                          dtype=np.float32,
                                          channels=2,
                                          blocksize=self.blockSize,
                                          callback=audio_callback(self))
            with self.stream:
                while True:
                    sd.sleep(1000)

        except KeyboardInterrupt:
            print("KEYBOARD")
        except Exception as e:
            print(e)

    def initialize_pybinsim(self):
        self.result = np.empty([self.blockSize, 2], dtype=np.float32)
        self.block = np.empty(
            [self.nChannels, self.blockSize], dtype=np.float32)

        # Create FilterStorage
        filterStorage = FilterStorage(self.config.get('filterSize'),
                                      self.blockSize,
                                      self.config.get('filterList'))

        # Start an oscReceiver
        oscReceiver = OscReceiver()
        oscReceiver.start_listening()
        time.sleep(1)

        # Create SoundHandler
        soundHandler = SoundHandler(self.blockSize, self.nChannels,
                                    self.sampleRate, self.config.get('loopSound'))

        soundfile_list = self.config.get('soundfile')
        soundHandler.request_new_sound_file(soundfile_list)

        # Create N convolvers depending on the number of wav channels
        self.log.info('Number of Channels: ' + str(self.nChannels))
        convolvers = [None] * self.nChannels
        for n in range(self.nChannels):
            convolvers[n] = ConvolverFFTW(self.config.get(
                'filterSize'), self.blockSize, False)

        # HP Equalization convolver
        convolverHP = None
        if self.config.get('useHeadphoneFilter'):
            convolverHP = ConvolverFFTW(self.config.get(
                'filterSize'), self.blockSize, True)
            hpfilter = filterStorage.get_headphone_filter()
            convolverHP.setIR(hpfilter, False)

        return convolverHP, convolvers, filterStorage, oscReceiver, soundHandler

    def __cleanup(self):
        # Close everything when BinSim is finished
        self.oscReceiver.close()
        self.stream.close()
        self.filterStorage.close()

        for n in range(self.nChannels):
            self.convolvers[n].close()

        if self.config.get('useHeadphoneFilter'):
            if self.convolverHP:
                self.convolverHP.close()


def audio_callback(binsim):
    """ Wrapper for callback to hand over custom data """
    assert isinstance(binsim, BinSim)

    # The python-sounddevice Callback
    def callback(outdata, frame_count, time_info, status):
        # print("python-sounddevice callback")

        current_soundfile_list = binsim.oscReceiver.get_sound_file_list()
        if current_soundfile_list:
            binsim.soundHandler.request_new_sound_file(current_soundfile_list)

        # Get sound block. At least one convolver should exist
        binsim.block[:binsim.soundHandler.get_sound_channels(
        ), :] = binsim.soundHandler.buffer_read()

        # Update Filters and run each convolver with the current block
        for n in range(binsim.soundHandler.get_sound_channels()):

            # Get new Filter
            if binsim.oscReceiver.is_filter_update_necessary(n):
                filterValueList = binsim.oscReceiver.get_current_values(n)
                filter = binsim.filterStorage.get_filter(
                    Pose.from_filterValueList(filterValueList))
                binsim.convolvers[n].setIR(
                    filter, callback.config.get('enableCrossfading'))

            left, right = binsim.convolvers[n].process(binsim.block[n, :])

            # Sum results from all convolvers
            if n == 0:
                binsim.result[:, 0] = left
                binsim.result[:, 1] = right
            else:
                binsim.result[:, 0] = np.add(binsim.result[:, 0], left)
                binsim.result[:, 1] = np.add(binsim.result[:, 1], right)

        # Finally apply Headphone Filter
        if callback.config.get('useHeadphoneFilter'):
            binsim.result[:, 0], binsim.result[:,
                                               1] = binsim.convolverHP.process(binsim.result)

        # Scale data
        binsim.result = np.divide(binsim.result, float(
            (binsim.soundHandler.get_sound_channels()) * 2))
        binsim.result = np.multiply(
            binsim.result, callback.config.get('loudnessFactor'))

        outdata[:, 0] = binsim.result[:, 0]
        outdata[:, 1] = binsim.result[:, 1]

        if np.max(np.abs(binsim.result)) > 1:
            binsim.log.warn('Clipping occurred: Adjust loudnessFactor!')

    callback.config = binsim.config

    return callback
