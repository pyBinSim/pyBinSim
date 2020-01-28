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

        self.default_filter_value = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.valueList_filter = [self.default_filter_value] * self.maxChannels

        self.default_late_reverb_value = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.valueList_late_reverb = [self.default_late_reverb_value] * self.maxChannels

        # self.valueList = [()] * self.maxChannels
        self.soundFileList = ''
        self.soundFileNew = False

        osc_dispatcher = dispatcher.Dispatcher()
        osc_dispatcher.map("/pyBinSimFilter", self.handle_filter_input)
        osc_dispatcher.map("/pyBinSimLateReverbFilter", self.handle_late_reverb_input)
        osc_dispatcher.map("/pyBinSimFile", self.handle_file_input)
        osc_dispatcher.map("/pyBinSimPauseAudioPlayback", self.handle_audio_pause)
        osc_dispatcher.map("/pyBinSimPauseConvolution", self.handle_convolution_pause)


        self.server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port), osc_dispatcher)

    def handle_filter_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """

        assert identifier == "/pyBinSimFilter"
        # assert all(isinstance(x, int) for x in args) == True

        # Extend value list to support older scripts
        # if (len(args)<6):
        #    args=(args+(0,)*6)[:6]
        #    print("filter value list incomplete")

        self.log.info("Channel: {}".format(str(channel)))
        self.log.info("Args: {}".format(str(args)))

        current_channel = channel

        if args != self.valueList_filter[current_channel]:
            #self.log.info("new filter")
            self.filters_updated[current_channel] = True
            self.valueList_filter[current_channel] = tuple(args)
        else:
            self.log.info("same filter as before")

    def handle_late_reverb_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """

        assert identifier == "/pyBinSimLateReverbFilter"

        self.log.info("Channel: {}".format(str(channel)))
        self.log.info("Args: {}".format(str(args)))

        current_channel = channel

        if args != self.valueList_late_reverb[current_channel]:
            #self.log.info("new late reverb filter")
            self.late_reverb_filters_updated[current_channel] = True
            self.valueList_late_reverb[current_channel] = tuple(args)
        else:
            self.log.info("same late reverb filter as before")

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
        return self.valueList_filter[channel]

    def get_current_late_reverb_values(self, channel):
        """ Return key for late reverb filters """
        self.late_reverb_filters_updated[channel] = False
        return self.valueList_late_reverb[channel]

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
