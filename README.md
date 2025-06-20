# vidtoolz-highlights

[![PyPI](https://img.shields.io/pypi/v/vidtoolz-highlights.svg)](https://pypi.org/project/vidtoolz-highlights/)
[![Changelog](https://img.shields.io/github/v/release/sukhbinder/vidtoolz-highlights?include_prereleases&label=changelog)](https://github.com/sukhbinder/vidtoolz-highlights/releases)
[![Tests](https://github.com/sukhbinder/vidtoolz-highlights/workflows/Test/badge.svg)](https://github.com/sukhbinder/vidtoolz-highlights/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sukhbinder/vidtoolz-highlights/blob/main/LICENSE)

Make highlights from videos

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
