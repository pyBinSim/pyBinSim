# coding=utf-8
import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand

VERSION = "1.2.0"


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(["--junitxml", "junit_results.xml"])
        sys.exit(errno)


setup(
    name='pybinsim',
    version=VERSION,
    license='MIT',
    author='Annika Neidhardt, Florian Klein, Thomas Koellmer',
    author_email='thomas.koellmer@tu-ilmenau.de',
    url='https://github.com/pyBinSim/pyBinSim',
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    install_requires=[
        "future >= 0.16.0",
        "numpy >= 1.12.1",
        "ovr >= 1.10.101",
        "pyaudio >= 0.2.10",
        "pyfftw >= 0.10.4",
        "pyserial >= 3.2.1",
        "pytest >= 3.0.5",
        "python-osc >= 1.6.3",
        "six >= 1.10.0",
        "Soundfile >= 0.9.0",
    ],

    description='Real-time dynamic binaural synthesis with head tracking.',
    long_description=open('README.rst').read(),
    packages=['pybinsim'],
    include_package_data=True,
    platforms='any',
    data_files=[],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Multimedia :: Sound/Audio',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License'
    ],

)
