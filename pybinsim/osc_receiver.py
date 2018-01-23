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

    def __init__(self):

        self.log = logging.getLogger("pybinsim.OscReceiver")
        self.log.info("oscReceiver: init")

        # Basic settings
        self.ip = '127.0.0.1'
        self.port = 10000
        self.maxChannels = 100

        # Default values; Stores filter keys for all channles/convolvers
        self.filters_updated = [True] * self.maxChannels

        self.defaultValue = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.valueList = [self.defaultValue] * self.maxChannels
        # self.valueList = [()] * self.maxChannels
        self.soundFileList = ''
        self.soundFileNew = False

        osc_dispatcher = dispatcher.Dispatcher()
        osc_dispatcher.map("/pyBinSim", self.handle_filter_input)
        osc_dispatcher.map("/pyBinSimFile", self.handle_file_input)

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

        assert identifier == "/pyBinSim"
        # assert all(isinstance(x, int) for x in args) == True

        # Extend value list to support older scripts
        # if (len(args)<6):
        #    args=(args+(0,)*6)[:6]
        #    print("filter value list incomplete")

        self.log.info("Channel: {}".format(str(channel)))
        self.log.info("Args: {}".format(str(args)))

        current_channel = channel

        if args != self.valueList[current_channel]:
            #self.log.info("new filter")
            self.filters_updated[current_channel] = True
            self.valueList[current_channel] = tuple(args)
        else:
            self.log.info("same filter as before")

    def handle_file_input(self, identifier, soundpath):
        """ Handler for playlist control"""

        assert identifier == "/pyBinSimFile"
        # assert type(soundpath) == 'str'

        self.log.info("soundPath: {}".format(soundpath))
        self.soundFileList = soundpath

    def start_listening(self):
        """Start osc receiver in background Thread"""

        self.log.info("Serving on {}".format(self.server.server_address))

        osc_thread = threading.Thread(target=self.server.serve_forever)
        osc_thread.daemon = True
        osc_thread.start()

    def is_filter_update_necessary(self, channel):
        """ Check if there is a new filter for channel """
        return self.filters_updated[channel]

    def get_current_values(self, channel):
        """ Return key for filter """
        self.filters_updated[channel] = False
        return self.valueList[channel]

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
