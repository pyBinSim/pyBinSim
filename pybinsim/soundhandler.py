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
import time

import numpy as np
import soundfile as sf


class SoundHandler(object):
    """ Class to read audio from files and serve it to pyBinSim """

    def __init__(self, block_size, n_channels, fs, loopSound):

        self.log = logging.getLogger("pybinsim.SoundHandler")

        self.fs = fs
        self.n_channels = n_channels
        self.chunk_size = block_size
        self.bufferSize = block_size * 2
        self.buffer = np.zeros(
            [self.n_channels, self.bufferSize], dtype=np.float32)
        self.sound = np.empty((0, 0))
        self.sound_file = np.empty((0, 0))
        self.frame_count = 0
        self.active_channels = 0
        self.soundPath = ''
        self.new_sound_file_request = False
        self.new_sound_file_loaded = False
        self.loopSound = loopSound
        self.currentSoundFile = 1
        self.soundFileList = []

        self._run_file_reader()

    def buffer_add_silence(self):
        self.buffer[:self.active_channels, :-
                    self.chunk_size] = self.buffer[:self.active_channels, self.chunk_size:]
        self.buffer[:self.active_channels, -self.chunk_size:] = np.zeros([self.active_channels, self.chunk_size],
                                                                         dtype=np.float32)

    def buffer_add_sound(self):
        if (self.frame_count + 1) * self.chunk_size < self.sound.shape[1]:
            self.buffer[:self.active_channels, :-
                        self.chunk_size] = self.buffer[:self.active_channels, self.chunk_size:]
            self.buffer[:self.active_channels, -self.chunk_size:] = self.sound[
                :self.active_channels,
                self.frame_count * self.chunk_size: (
                    self.frame_count + 1) * self.chunk_size
            ]
            self.frame_count += 1
        elif self.currentSoundFile < len(self.soundFileList) and not self.new_sound_file_request:
            self.request_next_sound_file()
        elif self.loopSound and not self.new_sound_file_request:
            self.currentSoundFile = 0
            self.request_next_sound_file()
            self.frame_count = 0
        else:
            self.buffer_add_silence()

    def buffer_flush(self):
        self.buffer = np.zeros(
            [self.n_channels, self.bufferSize], dtype=np.float32)

    def buffer_read(self):
        if self.new_sound_file_loaded:
            self.buffer_flush()
            self.sound = self.sound_file
            self.frame_count = 0
            self.new_sound_file_loaded = False

        buffer_content = self.buffer[:self.active_channels, :-self.chunk_size]
        self.buffer_add_sound()
        return buffer_content

    def _run_file_reader(self):
        file_read_thread = threading.Thread(target=self.read_sound_file)
        file_read_thread.daemon = True
        file_read_thread.start()

    def read_sound_file(self):

        while True:
            if self.new_sound_file_request:
                self.log.info('Loading new sound file')
                audio_file_data, fs = sf.read(self.soundPath, dtype='float32', )
                assert fs == self.fs

                self.log.debug("audio_file_data: {} MB".format(
                    audio_file_data.nbytes // 1024 // 1024))
                self.sound_file = np.asmatrix(audio_file_data)

                # free data
                audio_file_data = None

                if self.sound_file.shape[0] > self.sound_file.shape[1]:
                    self.sound_file = self.sound_file.transpose()

                self.active_channels = self.sound_file.shape[0]

                if self.sound_file.shape[1] % self.chunk_size != 0:
                    length_diff = self.chunk_size - \
                        (self.sound_file.shape[1] % self.chunk_size)
                    zeros = np.zeros(
                        (self.sound_file.shape[0], length_diff), dtype=np.float32)

                    self.log.debug("Zeros size: {} Byte".format(zeros.nbytes))
                    self.log.debug("Zeros shape: {} ({})".format(
                        zeros.shape, zeros.dtype))
                    self.log.debug("Soundfile size: {} MiB".format(
                        self.sound_file.nbytes // 1024 // 1024))
                    self.log.debug("Soundfile shape: {} ({})".format(
                        self.sound_file.shape, self.sound_file.dtype))
                    self.sound_file = np.concatenate(
                        (self.sound_file, zeros),
                        1
                    )
                    self.log.debug("Soundfile size after concat: {} MiB".format(
                        self.sound_file.nbytes // 1024 // 1024))
                    self.log.debug(
                        "Soundfile shape after concat: {} ({})".format(self.sound_file.shape, self.sound_file.dtype))
                    self.log.info('Loaded new sound file\n')
                self.new_sound_file_request = False
                self.new_sound_file_loaded = True
            time.sleep(0.05)

    def request_new_sound_file(self, sound_file_list):

        sound_file_list = str.split(sound_file_list, '#')
        self.soundPath = sound_file_list[0]
        self.soundFileList = sound_file_list
        self.currentSoundFile = 1
        self.new_sound_file_request = True

    def request_next_sound_file(self):

        self.currentSoundFile += 1
        self.soundPath = self.soundFileList[self.currentSoundFile - 1]
        self.new_sound_file_request = True

    def get_sound_channels(self):
        return self.active_channels
