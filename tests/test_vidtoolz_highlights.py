import pytest
import vidtoolz_highlights as w

from argparse import ArgumentParser


def test_create_parser():
    subparser = ArgumentParser().add_subparsers()
    parser = w.create_parser(subparser)

    assert parser is not None

    result = parser.parse_args(["--test", "hello"])
    assert result.test == ["hello"]


def test_plugin(capsys):
    w.highlights_plugin.hello(None)
    captured = capsys.readouterr()
    assert "Hello! This is an example ``vidtoolz`` plugin." in captured.out
