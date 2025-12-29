import pytest
import vidtoolz_highlights as w
import pytest
import numpy as np
from itertools import cycle
from unittest.mock import patch
from argparse import ArgumentParser
import builtins


@pytest.fixture
def subparsers():
    parser = ArgumentParser()
    return parser.add_subparsers(dest="command")


def test_create_parser2_required_and_defaults(subparsers):
    parser = w.create_parser2(subparsers)
    args = parser.parse_args(["myfile.txt"])

    assert args.filename == "myfile.txt"
    assert args.threshold == 0.3
    assert args.prefix == "IMG"
    assert args.fps == 60
    assert args.fadeout == 0
    assert args.afadeout == 2.0
    assert args.debug is False
    assert args.audfile is None
    assert args.startat == 0.0
    assert args.howmany == 15


def test_create_parser2_with_all_args(subparsers):
    parser = w.create_parser2(subparsers)
    args = parser.parse_args(
        [
            "myfile.txt",
            "-t",
            "0.5",
            "-p",
            "VID",
            "-fps",
            "30",
            "-foout",
            "1.5",
            "-af",
            "3.0",
            "-d",
            "-aaf",
            "audio.mp3",
            "-st",
            "2.5",
            "-hm",
            "10",
        ]
    )

    assert args.filename == "myfile.txt"
    assert args.threshold == 0.5
    assert args.prefix == "VID"
    assert args.fps == 30
    assert args.fadeout == 1.5
    assert args.afadeout == 3.0
    assert args.debug is True
    assert args.audfile == ["audio.mp3"]
    assert args.startat == 2.5
    assert args.howmany == 10


def test_create_parser_required_and_defaults(subparsers):
    parser = w.create_parser(subparsers)
    args = parser.parse_args(["myfile.txt"])

    assert args.filename == "myfile.txt"
    assert args.clip_time is None
    assert args.threshold == -0.1
    assert args.vtype == "NL"
    assert args.prefix == "IMG"
    assert args.fps == 60
    assert args.fadeout == 1.0
    assert args.afadeout == 2.0
    assert args.audfile is None
    assert args.startat == 0.0
    assert args.skipheader == 0
    assert args.skipfooter == 0


def test_create_parser_with_all_args(subparsers):
    parser = w.create_parser(subparsers)
    args = parser.parse_args(
        [
            "myfile.txt",
            "-ct",
            "5",
            "-t",
            "0.6",
            "-vt",
            "NL",
            "-p",
            "CLIP",
            "-fps",
            "24",
            "-foout",
            "2.0",
            "-af",
            "1.5",
            "-aaf",
            "track.mp3",
            "-st",
            "1.0",
        ]
    )

    assert args.filename == "myfile.txt"
    assert args.clip_time == 5
    assert args.threshold == 0.6
    assert args.vtype == "NL"
    assert args.prefix == "CLIP"
    assert args.fps == 24
    assert args.fadeout == 2.0
    assert args.afadeout == 1.5
    assert args.audfile == ["track.mp3"]
    assert args.startat == 1.0


def test_plugin(capsys):
    w.highlights_plugin.hello(None)
    captured = capsys.readouterr()
    assert "Hello! This is an example ``vidtoolz`` plugin." in captured.out


@pytest.fixture
def mock_random_choice():
    with patch("numpy.random.choice") as mock:
        yield mock


@pytest.fixture
def mock_random_uniform():
    with patch("numpy.random.uniform") as mock:
        yield mock


def test_get_non_linear_subclips_basic(mock_random_choice, mock_random_uniform):
    mov = ["a.mp4", "b.mp4"]
    vdurs = {"a.mp4": 10.0, "b.mp4": 12.0}
    dur = [2.0, 3.0, 4.0]
    time = 10

    # Setup mocks
    mock_random_choice.side_effect = lambda m: m[0]  # Always pick the first
    mock_random_uniform.return_value = 1.0  # Always use 1.0 as start_time

    result = w.get_non_linear_subclips_VDURS(mov[:], vdurs, dur, time)

    assert isinstance(result, list)
    assert all(len(clip) == 4 for clip in result)  # (file, start_time, end_time, speed)
    for file, start, end, speed in result:
        assert file in vdurs
        assert 0 <= start < end <= vdurs[file]
        assert speed in (0, 1)


def test_get_linear_subclips_basic(mock_random_uniform):
    mov = ["x.mp4", "y.mp4"]
    vdurs = {"x.mp4": 10.0, "y.mp4": 8.0}
    dur = [1.0, 2.0]
    ntime = 4

    # Always pick 0.5 as start time
    mock_random_uniform.return_value = 0.5

    result = w.get_linear_subclips(mov, vdurs, dur, ntime)

    assert len(result) == ntime - 1
    for file, start, end, speed in result:
        assert file in vdurs
        assert 0 <= start < end <= vdurs[file]
        assert speed in (0, 1)


def test_get_non_linear_subclips_handles_short_videos(
    mock_random_choice, mock_random_uniform
):
    mov = ["short.mp4"]
    vdurs = {"short.mp4": 1.5}
    dur = [2.0]
    time = 3

    # Force deterministic choice and start
    mock_random_choice.return_value = "short.mp4"
    mock_random_uniform.return_value = 0.0

    result = w.get_non_linear_subclips_VDURS(mov[:], vdurs, dur, time)

    # Should only try once then remove due to short duration
    assert len(result) <= 1


def test_get_linear_subclips_duration_adjustment(mock_random_uniform):
    mov = ["clip.mp4"]
    vdurs = {"clip.mp4": 1.0}
    dur = [2.0]
    ntime = 2

    # When span > duration, it should default to full clip
    mock_random_uniform.return_value = 0.0

    result = w.get_linear_subclips(mov, vdurs, dur, ntime)
    assert result[0][1] == 0.0  # start
    assert result[0][2] == 1.0  # end equals full duration
    assert result[0][3] in (0, 1)


def test_get_non_linear_span_speed_adjustment(mock_random_choice, mock_random_uniform):
    mov = ["test.mp4"]
    vdurs = {"test.mp4": 10.0}
    dur = [2.0, 7.0]
    time = 10

    mock_random_choice.return_value = "test.mp4"
    mock_random_uniform.return_value = 1.0

    result = w.get_non_linear_subclips_VDURS(mov[:], vdurs, dur, time)

    # Check that speed and span adjustments occurred
    for file, start, end, speed in result:
        if end - start > 6.0:
            assert speed == 0
        else:
            assert speed == 1


def test_extract_beat_times_basic():
    beats = np.array([[0.0, 0.1], [1.0, 0.5], [2.0, 0.7], [3.0, 0.2]])
    threshold = 0.3
    result = w.extract_beat_times(beats, threshold)
    assert result == [1.0, 2.0]


def test_extract_beat_times_empty_input():
    beats = np.empty((0, 2))
    threshold = 0.3
    result = w.extract_beat_times(beats, threshold)
    assert result == []


def test_extract_beat_times_all_above_threshold():
    beats = np.array([[0.5, 0.9], [1.5, 0.8]])
    result = w.extract_beat_times(beats, threshold=0.5)
    assert result == [0.5, 1.5]


def test_extract_beat_times_all_below_threshold():
    beats = np.array([[0.5, 0.1], [1.5, 0.2]])
    result = w.extract_beat_times(beats, threshold=0.3)
    assert result == []


def test_compute_segment_durations_basic():
    times = [0.0, 1.0, 2.5, 4.0]
    result = w.compute_segment_durations(times)
    assert result == [1.0, 1.5, 1.5]


def test_compute_segment_durations_single_element():
    times = [1.0]
    result = w.compute_segment_durations(times)
    assert result == []


def test_compute_segment_durations_empty():
    result = w.compute_segment_durations([])
    assert result == []


def test_compute_segment_durations_negative_durations():
    times = [3.0, 2.0, 1.0]
    result = w.compute_segment_durations(times)
    assert result == [-1.0, -1.0]


# Sample video_dict
video_dict = {"video1.mp4": 10.0}

# Mock subclips for trim_and_get_outfiles_for_coninous
subclips = {"video1.mp4": [(0.5, 6.0), (6.5, 9.0)], "video2.mp4": []}


# --- Tests for generate_video_cuts ---
@patch("random.uniform", side_effect=[0.5, 0.6, 0.5])  # control gaps
def test_generate_video_cuts_basic(mock_uniform):
    result = w.generate_video_cuts(video_dict, [2.0, 3.0], max_cuts=3)
    assert "video1.mp4" in result
    cuts = result["video1.mp4"]
    assert len(cuts) > 0
    for start, end in cuts:
        assert 0.0 <= start < end <= video_dict["video1.mp4"]


@patch("random.uniform", return_value=100.0)
def test_generate_video_cuts_no_space(mock_uniform):
    result = w.generate_video_cuts(video_dict, [2.0], max_cuts=5)
    # All generated gaps exceed video duration
    assert result["video1.mp4"] == []


# --- Tests for trim_and_get_outfiles_for_coninous ---
@patch("os.path.exists", return_value=True)
@patch("vidtoolz_highlights.trim_by_ffmpeg")
@patch("vidtoolz_highlights.get_length", return_value=5.0)
@patch("vidtoolz_highlights.choices", return_value=[1])
def test_trim_and_get_outfiles(mock_choices, mock_get_length, mock_trim, mock_exists):
    output = w.trim_and_get_outfiles_for_coninous(subclips)
    assert len(output) == 3  # 2 cuts from video1.mp4, 1 full cut from video2.mp4
    for out in output:
        assert out.endswith(".mp4")
    assert mock_trim.call_count == 3


@patch("os.path.exists", return_value=False)
@patch("vidtoolz_highlights.trim_by_ffmpeg")
@patch("vidtoolz_highlights.get_length", return_value=5.0)
@patch("vidtoolz_highlights.choices", return_value=[0])
def test_trim_skips_missing_files(
    mock_choices, mock_get_length, mock_trim, mock_exists
):
    output = w.trim_and_get_outfiles_for_coninous(subclips)
    assert output == []  # os.path.exists always returns False
