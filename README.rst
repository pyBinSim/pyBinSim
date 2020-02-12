.. image:: https://travis-ci.org/pyBinSim/pyBinSim.svg?branch=master
    :target: https://travis-ci.org/pyBinSim/pyBinSim

PyBinSim
========

Install
-------

::

    $ conda create --name binsim python=3.5 numpy
    $ source activate binsim
    $ pip install pybinsim
    
On linux, make sure that gcc and the development headers for libfftw and portaudio are installed, before invoking `pip install pybinsim`.
For ubuntu::

    $ apt-get install gcc portaudio19-dev libfftw3-dev
    

Run
---

Create ``pyBinSimSettings.txt`` file with content like this

::

    soundfile signals/test441kHz.wav
    blockSize 512
    filterSize 16384
    filterList brirs/filter_list_kemar5.txt
    maxChannels 2
    samplingRate 44100
    enableCrossfading True
    useHeadphoneFilter False
    headphoneFilterSize 1024
    loudnessFactor 0.5
    loopSound False
    useSplittedFilters False
    lateReverbSize 16384
    pauseConvolution False
    pauseAudioPlayback False



Start Binaural Simulation

::

    import pybinsim
    import logging

    pybinsim.logger.setLevel(logging.DEBUG)    # defaults to INFO
    #Use logging.WARNING for printing warnings only

    with pybinsim.BinSim('pyBinSimSettings.txt') as binsim:
        binsim.stream_start()

Description
===========

Basic principle:
----------------

Depending on the number of input channels (wave-file channels) the corresponding number of virtual sound sources is created. The filter for each sound source can selected and activitated via OSC messages. The messages basically contain the number
index of the source for which the filter should be switched and an identifier string to address the correct filter. The correspondence between parameter value and filter is determined by a filter list which can be adjusted individually for the specific use case.
    
Config parameter description:
-----------------------------

soundfile: 
    Defines \*.wav file which is played back at startup. Sound file can contain up to maxChannels audio channels. Also accepts multiple files separated by '#'; Example: 'soundfile signals/sound1.wav#signals/sound2.wav.  Can be changed via OSC.
blockSize: 
    Number of samples which are processed per block. Low values reduce delay but increase cpu load.
filterSize: 
    Defines filter size of the filters loaded with the filter list. Filter size should be a mutltiple of blockSize.
maxChannels: 
    Maximum number of sound sources/audio channels which can be controlled during runtime. The value for maxChannels must match or exceed the number of channels of soundFile(s).
samplingRate: 
    Sample rate for filters and soundfiles. Caution: No automatic sample rate conversion.
enableCrossfading: 
    Enable cross fade between audio blocks. Set 'False' or 'True'.
useHeadphoneFilter: 
    Enables headhpone equalization. The filterset should contain a filter with the identifier HPFILTER. Set 'False' or 'True'.
headphoneFilterSize:
    Size of the headphone filter. Size has to be dividable by blockSize.
loudnessFactor:
    Factor for overall output loudness. Attention: Clipping may occur
loopSound:
    Enables looping of sound file or sound file list. Set 'False' or 'True'.
useSplittedFilters:
    If enabled, a static late reverb filter is attached to all filters in the filterlist. The filterset should contain a filter with the identifier LATEREVERB. Set 'False' or 'True'.
lateReverbSize:
    Size of the late reverb filter. Size has to be dividable by blockSize.
pauseConvolution:
    Bypasses convolution. Set 'False' or 'True'. Can be changed via OSC.
pauseAudioPlayback:
    For pausing the audio playback. Set 'False' or 'True'. Can be changed via OSC.


OSC Messages and filter lists:
------------------------------

Example lines from filter list:

HPFILTER hpirs/DT990_EQ_filter_2ch.wav

FILTER 165 2 0 0 0 0 0 0 0 brirs/kemar_0_165.wav

LATEREVERB 0 2 0 0 0 0 0 0 0 brirs/late_reverb.wav

Lines with the prefix FILTER or LATEREVERB contain a 'filter key' which consist of 6 or 9 positive numbers. These numbers
can be arbitrarily assigned to suit your use case. They are used to tell pyBinSim which filter to apply.
The filter behind the prefix HPFILTER will be loaded and applied automatically when useHeadphoneFilter == True.
Lines which start with FILTER or LATEREVERB have to be called via OSC commands to become active.
To activate a FILTER for the third channel of your wav file you have to send the the identifier
'/pyBinSimFilter', followed by a 2 (corresponding to the third channel) and followed by the nine 9 (or six) key numbers from the filter list
to the pc where pyBinSim runs (UDP, port 10000):

::

    /pyBinSimFilter 2 165 2 0 0 0 0 0 0 0

When you have set useSplittedFilters == True, you can also apply a late reverb filter. This filter gets attached to the
activated FILTER (crossfade with the size of a blocksize). Be careful to make sure you always have combined the FILTER
and LATEREVERB you intended - pyBinSim will combine the filters blindly. Example:

::

    /pyBinSimLateReverbFilter 2 2 0 0 0 0 0 0 0
        
When you want to play another sound file you send:

::

    /pyBinSimFile folder/file_new.wav

Or a sound file list:

::

    /pyBinSimFile folder/file_1.wav#folder/file_2.wav

The audiofile has to be located on the pc where pyBinSim runs. Files are not transmitted over network.

Further OSC Messages:
------------------------------

Pause audio playback. Send 'True' or 'False' (as string, not bool)

::

    /pyBinSimPauseAudioPlayback 'True'

Bypass convolution. Send 'True' or 'False' (as string, not bool)

::

    /pyBinSimPauseConvolution 'True'


Demos
-----

DEPRECATED for this version: Check the https://github.com/pyBinSim/AppExamples repository for ready-to-use demos.




Reference:
----------

Please cite our work:

Neidhardt, A.; Klein, F.; Knoop, N. and Köllmer, T., "Flexible Python tool for dynamic binaural synthesis applications", 142nd AES Convention, Berlin, 2017.



