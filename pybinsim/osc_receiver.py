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
import threading
import numpy as np

from pythonosc import dispatcher
from pythonosc import osc_server


class OscReceiver(object):
    """
    Class for receiving OSC Messages to control pyBinSim
    """

    def __init__(self,current_config):

        self.log = logging.getLogger("pybinsim.OscReceiver")
        self.log.info("oscReceiver: init")

        # Basic settings
        self.ip = '127.0.0.1'
        self.port = 10000
        self.maxChannels = 100

        self.currentConfig = current_config

        # Default values; Stores filter keys for all channles/convolvers
        self.filters_updated = [True] * self.maxChannels
        self.late_reverb_filters_updated = [True] * self.maxChannels

        self.default_filter_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.valueList_filter = np.tile(self.default_filter_value, [self.maxChannels, 1])

        self.default_late_reverb_value = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.valueList_late_reverb = np.tile(self.default_late_reverb_value, [self.maxChannels, 1])

        # self.valueList = [()] * self.maxChannels
        self.soundFileList = ''
        self.soundFileNew = False

        osc_dispatcher = dispatcher.Dispatcher()
        osc_dispatcher.map("/pyBinSimFilter", self.handle_filter_input)
        osc_dispatcher.map("/pyBinSimFilterShort", self.handle_filter_input)
        osc_dispatcher.map("/pyBinSimFilterOrientation", self.handle_filter_input)
        osc_dispatcher.map("/pyBinSimFilterPosition", self.handle_filter_input)
        osc_dispatcher.map("/pyBinSimFilterCustom", self.handle_filter_input)
        osc_dispatcher.map("/pyBinSimLateReverbFilter", self.handle_late_reverb_input)
        osc_dispatcher.map("/pyBinSimLateReverbFilterShort", self.handle_late_reverb_input)
        osc_dispatcher.map("/pyBinSimLateReverbFilterOrientation", self.handle_late_reverb_input)
        osc_dispatcher.map("/pyBinSimLateReverbFilterPosition", self.handle_late_reverb_input)
        osc_dispatcher.map("/pyBinSimLateReverbFilterCustom", self.handle_late_reverb_input)
        osc_dispatcher.map("/pyBinSimFile", self.handle_file_input)
        osc_dispatcher.map("/pyBinSimPauseAudioPlayback", self.handle_audio_pause)
        osc_dispatcher.map("/pyBinSimPauseConvolution", self.handle_convolution_pause)


        self.server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port), osc_dispatcher)

    def select_slice(self, i):
        switcher = {
            "/pyBinSimFilter": slice(0, 9),
            "/pyBinSimFilterShort": slice(0, 6),
            "/pyBinSimFilterOrientation": slice(0, 3),
            "/pyBinSimFilterPosition": slice(3, 6),
            "/pyBinSimFilterCustom": slice(6, 9),
            "/pyBinSimLateReverbFilter": slice(0, 9),
            "/pyBinSimLateReverbFilterShort": slice(0, 6),
            "/pyBinSimLateReverbFilterOrientation": slice(0, 3),
            "/pyBinSimLateReverbFilterPosition": slice(3, 6),
            "/pyBinSimLateReverbFilterCustom": slice(6, 9)
        }
        return switcher.get(i, [])

    def handle_filter_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """
        current_channel = channel
        key_slice = self.select_slice(identifier)

        if len(args) == len(self.valueList_filter[current_channel, key_slice]):
            if all(args == self.valueList_filter[current_channel, key_slice]):
                self.log.info("Same filter as before")
            else:
                self.filters_updated[current_channel] = True
                self.valueList_filter[current_channel, key_slice] = args
        else:
            self.log.info('OSC identifier and key mismatch')


        self.log.info("Channel: {}".format(str(channel)))
        self.log.info("Current Filter List: {}".format(str(self.valueList_filter[current_channel, :])))

    def handle_late_reverb_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """
        current_channel = channel
        key_slice = self.select_slice(identifier)

        if len(args) == len(self.valueList_late_reverb[current_channel, key_slice]):
            if all(args == self.valueList_late_reverb[current_channel, key_slice]):
                self.log.info("Same late reverb filter as before")
            else:
                self.late_reverb_filters_updated[current_channel] = True
                self.valueList_late_reverb[current_channel, key_slice] = args
        else:
            self.log.info('OSC identifier and key mismatch')

        self.log.info("Channel: {}".format(str(channel)))
        self.log.info("Current Late Reverb Filter List: {}".format(str(self.valueList_late_reverb[current_channel, :])))

    def handle_file_input(self, identifier, soundpath):
        """ Handler for playlist control"""

        assert identifier == "/pyBinSimFile"
        # assert type(soundpath) == 'str'

        self.log.info("soundPath: {}".format(soundpath))
        self.soundFileList = soundpath

    def handle_audio_pause(self, identifier, value):
        """ Handler for playback control"""
        assert identifier == "/pyBinSimPauseAudioPlayback"

        self.currentConfig.set('pauseAudioPlayback', value)

    def handle_convolution_pause(self, identifier, value):
        """ Handler for playback control"""
        assert identifier == "/pyBinSimPauseConvolution"

        self.currentConfig.set('pauseConvolution', value)

    def start_listening(self):
        """Start osc receiver in background Thread"""

        self.log.info("Serving on {}".format(self.server.server_address))

        osc_thread = threading.Thread(target=self.server.serve_forever)
        osc_thread.daemon = True
        osc_thread.start()

    def is_filter_update_necessary(self, channel):
        """ Check if there is a new filter for channel """
        return self.filters_updated[channel]

    def is_late_reverb_update_necessary(self, channel):
        """ Check if there is a new late reverb filter for channel """
        if self.currentConfig.get('useSplittedFilters'):
            return self.late_reverb_filters_updated[channel]
        else:
            return False

    def get_current_filter_values(self, channel):
        """ Return key for filter """
        self.filters_updated[channel] = False
        return self.valueList_filter[channel,:]

    def get_current_late_reverb_values(self, channel):
        """ Return key for late reverb filters """
        self.late_reverb_filters_updated[channel] = False
        return self.valueList_late_reverb[channel,:]

    def get_current_config(self):
        return self.currentConfig

    def get_sound_file_list(self):
        ret_list = self.soundFileList
        self.soundFileList = ''
        return ret_list

    def close(self):
        """
        Close the osc receiver

        :return: None
        """
        self.log.info('oscReiver: close()')
        self.server.shutdown()
