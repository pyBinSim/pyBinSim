.. image:: https://travis-ci.org/pyBinSim/pyBinSim.svg?branch=master
    :target: https://travis-ci.org/pyBinSim/pyBinSim

PyBinSim
========

Install
-------

Let's create a virtual environment. Use either Python or Conda to do this and then use `pip` to install the dependencies.

Windows
-------

Assuming you are using the default command line 
(navigate to `pyBinSim/` folder in Explorer, click into the address bar, type `cmd` and hit enter).


python

::
    $ <PathToPython >= 3.6> -m venv venv
    $ venv/Scripts/activate.bat
    $ pip install pybinsim

For Powershell, the activation command is `venv/Scripts/Activate.ps1`.


conda

::
    $ conda create --name binsim python>=3.6 numpy
    $ conda activate binsim
    $ pip install pybinsim


Linux
-----

On linux, make sure that gcc and the development headers for libfftw and portaudio are installed, before invoking `pip install pybinsim`.

For ubuntu

::

    $ apt-get install gcc portaudio19-dev libfftw3-dev

For Fedora

::

    $ sudo dnf install gcc portaudio19-devel fftw-devel


python

::
    $ <PathToPython >= 3.6> -m venv venv
    $ source venv/bin/activate
    $ pip install pybinsim


conda

::
    $ conda create --name binsim python>=3.6 numpy
    $ conda activate binsim
    $ pip install pybinsim
    

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
    loudnessFactor 0.5
    loopSound False


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
    Defines \*.wav file which is played back at startup. Sound file can contain up to maxChannels audio channels. Also accepts multiple files separated by '#'; Example: 'soundfile signals/sound1.wav#signals/sound2.wav
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
loudnessFactor: 
    Factor for overall output loudness. Attention: Clipping may occur
loopSound:
    Enables looping of sound file or sound file list. Set 'False' or 'True'.


OSC Messages and filter lists:
------------------------------

Example line from filter list:
165 2 0 0 0 0 brirs/kemar5/kemar_0_165.wav

To activate this filter for the third channel (counting starts at zero) for your wav file you have to send the following message to the pc where pyBinSim runs (port 10000):

::

    /pyBinSim 2 165 2 0 0 0 0
        
When you want to play another sound file you send:

::

    /pyBinSimFile folder/file_new.wav

Or a sound file list

::

    /pyBinSimFile folder/file_1.wav#folder/file_2.wav

The audiofile has to be located on the pc where pyBinSim runs. Files are not transmitted over network.


Demos
-----

Check the https://github.com/pyBinSim/AppExamples repository for ready-to-use demos.




Reference:
----------

Please cite our work:

Neidhardt, A.; Klein, F.; Knoop, N. and Köllmer, T., "Flexible Python tool for dynamic binaural synthesis applications", 142nd AES Convention, Berlin, 2017.



