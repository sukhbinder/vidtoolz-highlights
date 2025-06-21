# vidtoolz-highlights

[![PyPI](https://img.shields.io/pypi/v/vidtoolz-highlights.svg)](https://pypi.org/project/vidtoolz-highlights/)
[![Changelog](https://img.shields.io/github/v/release/sukhbinder/vidtoolz-highlights?include_prereleases&label=changelog)](https://github.com/sukhbinder/vidtoolz-highlights/releases)
[![Tests](https://github.com/sukhbinder/vidtoolz-highlights/workflows/Test/badge.svg)](https://github.com/sukhbinder/vidtoolz-highlights/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sukhbinder/vidtoolz-highlights/blob/main/LICENSE)

Make highlights from videos. Stitch videos to music

## Installation

First install [vidtoolz](https://github.com/sukhbinder/vidtoolz).

```bash
pip install vidtoolz
```

Then install this plugin in the same environment as your vidtoolz application.

```bash
vidtoolz install vidtoolz-highlights
```
## Usage

type ``vid highlights --help`` to get help

```bash
usage: vid highlights [-h] [-ct CLIP_TIME] [-t THRESHOLD] [-vt {L,NL}]
                      [-p PREFIX] [-fps FPS] [-foout FADEOUT] [-af AFADEOUT]
                      [-aaf AUDFILE] [-st STARTAT]
                      filename

Make highlights from videos

positional arguments:
  filename              File containing the list of files

optional arguments:
  -h, --help            show this help message and exit
  -ct CLIP_TIME, --clip-time CLIP_TIME
                        Clip Time (default: None)
  -t THRESHOLD, --threshold THRESHOLD
                        Clip Time (default: 0.3)
  -vt {L,NL}, --vtype {L,NL}
                        Vtype Linear or Non Linear (default: L)
  -p PREFIX, --prefix PREFIX
                        Filename Prefix (default: IMG)
  -fps FPS, --fps FPS   Video FPS (default: 60)
  -foout FADEOUT, --fadeout FADEOUT
                        Video fadeout (default: 1.0)
  -af AFADEOUT, --afadeout AFADEOUT
                        Audio fadeout (default: 2.0)
  -aaf AUDFILE, --audfile AUDFILE
                        mp3 Audio file (default: None)
  -st STARTAT, --startat STARTAT
                        Audio startat (default: 0.0)

```

type ``vid stitch --help`` to get help

```bash
usage: vid stitch [-h] [-t THRESHOLD] [-p PREFIX] [-fps FPS] [-foout FADEOUT]
                  [-af AFADEOUT] [-d] [-aaf AUDFILE] [-st STARTAT]
                  [-hm HOWMANY]
                  filename

Stitch videos using music

positional arguments:
  filename              File containing the list of files

optional arguments:
  -h, --help            show this help message and exit
  -t THRESHOLD, --threshold THRESHOLD
                        Beats amplitude (default: 0.3)
  -p PREFIX, --prefix PREFIX
                        Filename Prefix (default: IMG)
  -fps FPS, --fps FPS   Video FPS (default: 60)
  -foout FADEOUT, --fadeout FADEOUT
                        Video fadeout (default: 0)
  -af AFADEOUT, --afadeout AFADEOUT
                        Audio FPS (default: 2.0)
  -d, --debug           Debug this (default: False)
  -aaf AUDFILE, --audfile AUDFILE
                        mp3 Audio file (default: None)
  -st STARTAT, --startat STARTAT
                        Audio startat (default: 0.0)
  -hm HOWMANY, --howmany HOWMANY
                        How many clips to take (default: 15)

```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd vidtoolz-highlights
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
