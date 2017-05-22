.. image:: https://travis-ci.org/pyBinSim/pyBinSim.svg?branch=master
    :target: https://travis-ci.org/pyBinSim/pyBinSim

PyBinSim
========

Install
-------

::

    $ conda create --name binsim35 python=3.5 numpy scipy
    $ source activate binsim35
    $ pip install pybinsim

Run
---

Create ``pyBinSimSettings.txt`` file with content like this

::

    soundfile signals/test.wav
    blockSize 256
    filterSize 16384
    filterList brirs/filter_list_kemar5.txt
    maxChannels 8
    samplingRate 44100
    enableCrossfading False
    useHeadphoneFilter False
    loudnessFactor 1


Start Binaural Simulation

::

    import pybinsim

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
    Defines \*.wav file which is played back at startup. Sound file can contain up to maxChannels audio channels.
blockSize: 
    Number of samples which are processed per block. Low values reduce delay but increase cpu load.
filterSize: 
    Defines filter size of the filters loaded with the filter list. Filter size should be a mutltiple of blockSize.
maxChannels: 
    Maximum number of sound sources/audio channels which can be controlled during runtime.
samplingRate: 
    Sample rate for filters and soundfiles. Caution: No automatic sample rate conversion.
enableCrossfading: 
    Enable cross fade between audio blocks. Set 'False' or 'True'.
useHeadphoneFilter: 
    Enables headhpone equalization. The filterset should contain a filter with the identifier HPFILTER. Set 'False' or 'True'.
loudnessFactor: 
    Factor for overall output loudness.

OSC Messages and filter lists:
------------------------------

Example line from filter list:
165 2 0 0 0 0 brirs/kemar5/kemar_0_165.wav

To activate this filter for the third channel (counting starts at zero) for your wav file you have to send the following message to the pc where pyBinSim runs (port 10000):

::

    /pyBinSim 2 165 2 0 0 0 0
        
When you want to play another sound file you send:

::

    /pyBinSimFile file_new.wav

The audiofile has to be located on the pc where pyBinSim runs. Files are not transmitted over network.


Demos
-----

Check the https://github.com/pyBinSim/AppExamples repository for ready-to-use demos.




Reference:
----------

Please cite our work:

Neidhardt, A.; Klein, F.; Knoop, N. and Köllmer, T., "Flexible Python tool for dyanmic binaural synthesis applications", 142nd AES Convention, Berlin, 2017.



