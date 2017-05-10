.. image:: https://travis-ci.org/pyBinSim/pyBinSim.svg?branch=master
    :target: https://travis-ci.org/pyBinSim/pyBinSim

PyBinSim
========

Install
-------

::

    conda create --name binsim35 python=3.5 numpy scipy
    pip install pybinsim

Run
---

Create ``pyBinSimSettings.txt`` file with content like this

::

    soundfile signals/speech2_44100_mono.wav
    blockSize 256
    filterSize 16384
    filterList brirs/filter_list_kemar5.txt
    maxChannels 8
    samplingRate 44100
    enableCrossfading False
    useHeadphoneFilter False
    loudnessFactor 5


Start Binarual Simulation

::

    from pybinsim import BinSim

    with BinSim('pyBinSimSettings.txt') as binsim:
        binsim.stream_start()



