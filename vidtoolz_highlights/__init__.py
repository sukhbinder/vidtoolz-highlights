import vidtoolz
import os
from vidtoolz_beats import detect_beats
import subprocess
import moviepy as mpy
import numpy as np
from itertools import cycle
import tempfile
from moviepy import afx
from pathlib import Path
import json
import copy

import random
from typing import Dict, List, Tuple
from itertools import cycle
from random import choices


def write_subclips_json(fname, subclips):
    with open(fname, "w") as fout:
        json.dump(subclips, fout, indent=4)


def create_parser2(subparser):
    parser = subparser.add_parser("stitch", description="Stitch videos using music")

    parser.add_argument("filename", type=str, help="File containing the list of files")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Beats amplitude (default: %(default)s)",
        default=0.3,
    )

    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        help="Filename Prefix (default: %(default)s)",
        default="IMG",
    )

    parser.add_argument(
        "-fps", "--fps", type=int, help="Video FPS (default: %(default)s)", default=60
    )
    parser.add_argument(
        "-foout",
        "--fadeout",
        type=float,
        help="Video fadeout (default: %(default)s)",
        default=0,
    )
    parser.add_argument(
        "-af",
        "--afadeout",
        type=float,
        help="Audio FPS (default: %(default)s)",
        default=2.0,
    )
    parser.add_argument(
        "-d", "--debug", help="Debug this (default: %(default)s)", action="store_true"
    )

    parser.add_argument(
        "-aaf",
        "--audfile",
        type=str,
        help="mp3 Audio file (default: %(default)s)",
        default=None,
    )
    parser.add_argument(
        "-st",
        "--startat",
        type=float,
        help="Audio startat (default: %(default)s)",
        default=0.0,
    )

    parser.add_argument(
        "-hm",
        "--howmany",
        type=int,
        help="How many clips to take (default: %(default)s)",
        default=15,
    )

    return parser


def create_parser(subparser):
    parser = subparser.add_parser(
        "highlights", description="Make highlights from videos"
    )
    # Add subprser arguments here.
    parser.add_argument("filename", type=str, help="File containing the list of files")
    parser.add_argument(
        "-ct",
        "--clip-time",
        type=int,
        help="Clip Time (default: %(default)s)",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Clip Time (default: %(default)s)",
        default=0.3,
    )
    parser.add_argument(
        "-vt",
        "--vtype",
        type=str,
        help="Vtype Linear or Non Linear (default: %(default)s)",
        choices=["L", "NL"],
        default="L",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        help="Filename Prefix (default: %(default)s)",
        default="IMG",
    )

    parser.add_argument(
        "-fps", "--fps", type=int, help="Video FPS (default: %(default)s)", default=60
    )
    parser.add_argument(
        "-foout",
        "--fadeout",
        type=float,
        help="Video fadeout (default: %(default)s)",
        default=1.0,
    )
    parser.add_argument(
        "-af",
        "--afadeout",
        type=float,
        help="Audio fadeout (default: %(default)s)",
        default=2.0,
    )

    parser.add_argument(
        "-aaf",
        "--audfile",
        type=str,
        help="mp3 Audio file (default: %(default)s)",
        default=None,
    )

    parser.add_argument(
        "-st",
        "--startat",
        type=float,
        help="Audio startat (default: %(default)s)",
        default=0.0,
    )
    return parser


def read_orderfile(fname):
    fname = os.path.abspath(fname)
    fdir = os.path.dirname(fname)
    with open(fname, "r") as fin:
        files = fin.readlines()

    mov = [os.path.join(fdir, f.strip()) for f in files]
    return mov


def get_length(filename):
    """Get video duration using ffprobe, fallback to moviepy if it fails."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
        duration = float(result.stdout.strip())
        if duration <= 0:
            raise ValueError("Duration returned by ffprobe is zero or negative.")
        return duration
    except Exception as e:
        print(
            f"[WARN] ffprobe failed for {filename}, falling back to moviepy. Reason: {e}"
        )
        try:
            with mpy.VideoFileClip(filename) as clip:
                return clip.duration
        except Exception as ex:
            print(f"[ERROR] Failed to get duration via moviepy for {filename}: {ex}")
            return 0.0


def beats_clip(audfile, offset=0.0):
    song_name = os.path.basename(audfile)

    snd = mpy.AudioFileClip(audfile)
    new_audioclip = mpy.CompositeAudioClip([snd.subclipped(start_time=offset)])
    return song_name, new_audioclip


def get_non_linear_subclips_VDURS(mov, vdurs, dur, time):
    subclips = []
    cc = cycle(dur)
    cumdur = 0
    while cumdur <= time:
        if not mov:
            print("No more valid files to choose from.")
            break
        span = next(cc)
        cumdur = cumdur + span
        while True:
            file = np.random.choice(mov)
            try:
                duration = vdurs[file]
            except Exception as ex:
                print(file, "wrong")
                continue
            speed = 0 if span > 6.0 else 1
            if speed == 1:
                span = span / 2
            start_time = np.random.uniform(0, duration - span)
            if duration - span > 0:
                break
        # remove file for less than 2 sec duration
        if int(duration / 2) <= 1:
            mov.remove(file)
        if span > 6:  # Remove if a big cut has been done from a file
            mov.remove(file)
        end_time = start_time + span
        subclips.append((file, start_time, end_time, speed))
        print(span, cumdur, file, start_time, end_time, speed)

    return subclips


def get_linear_subclips(mov, vdurs, dur, ntime):
    subclips = []
    cc = cycle(mov)
    nd = cycle(dur)
    for i in range(ntime - 1):
        span = next(nd)
        while True:
            file = next(cc)
            try:
                duration = vdurs[file]
            except Exception as ex:
                print(file, "wrong")
                continue
            speed = 0 if span < 0.3 else 1
            if duration <= span:
                start_time = 0.0  # take the whole
                span = duration
                break
            if span > 5:  # if more than 10 sec video don't slow
                speed = 0
            if speed == 1:
                span = span / 2
            start_time = np.random.uniform(0, duration - span)
            if duration - span > 0:
                break
        end_time = start_time + span
        subclips.append((file, start_time, end_time, speed))
    return subclips


def get_seconds(ts):
    secs = sum(int(x) * 60**i for i, x in enumerate(reversed(ts.split(":"))))
    return secs


def trim_by_ffmpeg(inputfile, starttime, endtime, outputfile, duration=None):
    if isinstance(starttime, str):
        if ":" in starttime:
            starttime = get_seconds(starttime)
    if isinstance(endtime, str):
        if ":" in endtime:
            endtime = get_seconds(endtime)

    if duration is not None:
        cmdline = "ffmpeg -y -ss {starttime:0.4f} -i {inputfile} -t {duration:0.4f} -c copy {outputfile}".format(
            starttime=float(starttime),
            inputfile=inputfile,
            duration=float(duration),
            outputfile=outputfile,
        )
    else:
        cmdline = "ffmpeg -y -ss {starttime:0.4f} -i {inputfile} -to {endtime:0.4f} -map 0 -vcodec copy -acodec copy  {outputfile}".format(
            starttime=float(starttime),
            inputfile=inputfile,
            endtime=float(endtime),
            outputfile=outputfile,
        )
    cmdlist = cmdline.split()
    iret = subprocess.call(cmdlist)
    return iret


def trim_and_get_outfiles(sc):
    outfiles = []
    for i, clip in enumerate(sc):
        fn, st, et, speed = clip
        # Round to four decimal place.
        st, et = np.round(st, 4), np.round(et, 4)
        fna = os.path.basename(fn)
        fna, ext = os.path.splitext(fna)
        dur = et - st
        if speed == 1:
            outfile = "{0}_{1}_output_s.mp4".format(i, fna)
        else:
            outfile = "{0}_{1}_output.mp4".format(i, fna)
        trim_by_ffmpeg(fn, st, et, outfile, dur)
        if os.path.exists(outfile):
            outfiles.append(outfile)
    return outfiles


def make_video(files, fname):
    base_name = os.path.basename(fname)
    bname, ext = os.path.splitext(base_name)
    out_file = "{}_mylist.txt".format(bname)

    # Slow the videos using ffmpeg
    sfiles = []
    for f in files:
        if f.endswith("_s.mp4"):
            outf = "{0}-s.mp4".format(f)
            cmdline = "ffmpeg -i {0} -an -filter:v 'setpts=2.0*PTS' {1}".format(f, outf)
            print(cmdline)
            iret = os.system(cmdline)
            sfiles.append(outf)
        else:
            outf = "{0}-ns.mp4".format(f)
            cmdline = "ffmpeg -i {0} -an -c:v copy {1}".format(f, outf)
            sfiles.append(outf)

    with open(out_file, "w") as fout:
        for f in sfiles:
            if os.path.exists(f):
                fout.write("file '{}'\n".format(f))
    cmdline = "ffmpeg -f concat -safe 0 -i {0} -c copy {1}".format(out_file, fname)
    print(cmdline)
    iret = os.system(cmdline)
    print(cmdline)
    return iret


def generate_video_hl(
    vc, new_audioclip, outfile, fps=30, fadeout=1, afadeout=2, clip=None
):
    if clip is None:
        clip = mpy.concatenate_videoclips(vc, method="compose")
        clip = clip.fadeout(fadeout)

    try:
        audiofile = os.path.join(os.path.dirname(outfile), "out_audio.mp3")
        clip.audio.write_audiofile(audiofile, fps=44100)
    except Exception as ex:
        print(ex)
        pass

    clipduration = clip.duration
    min, sec = divmod(clipduration, 60)
    print(
        "Duration of generated clip is {0:.2f} seconds or {1:.0f}:{2:.0f} ".format(
            clipduration, min, sec
        )
    )
    if new_audioclip.duration < clipduration:
        naudio = new_audioclip.with_effects([afx.AudioLoop(duration=clipduration)])
    else:
        naudio = new_audioclip.with_duration(clipduration)  # .audio_fadeout(afadeout)

    clip_withsound = clip.with_audio(naudio)
    print("fps is : ", fps)
    clip_withsound.write_videofile(
        outfile, temp_audiofile="out.m4a", audio_codec="aac", fps=fps
    )
    clip.close()
    return clip_withsound


def extract_beat_times(beats: np.ndarray, threshold: float) -> list[float]:
    """Filter beat times based on threshold."""
    return beats[beats[:, 1] >= threshold, 0].tolist()


def compute_segment_durations(times: list[float]) -> list[float]:
    """Compute durations between consecutive beat times."""
    return [t2 - t1 for t1, t2 in zip(times, times[1:])]


def create_subclips(vtype: str, mov, vdurs, durations, clip_time):
    """Choose subclips using linear or non-linear strategy."""
    if vtype == "NL":
        return get_non_linear_subclips_VDURS(mov, vdurs, durations, clip_time)
    return get_linear_subclips(mov, vdurs, durations, clip_time)


class ViztoolzPlugin:
    """Make highlights from videos"""

    __name__ = "highlights"

    @vidtoolz.hookimpl
    def register_commands(self, subparser):
        self.parser = create_parser(subparser)
        self.parser.set_defaults(func=self.run)

    def run(self, args):
        audfile = args.audfile
        startat = args.startat
        threshold = args.threshold
        vtype = args.vtype
        clip_time = args.clip_time
        fps = args.fps
        fadeout = args.fadeout
        afadeout = args.afadeout
        prefix = args.prefix

        # 1. Read inputs
        mov = read_orderfile(args.filename)
        vdir = os.path.dirname(os.path.abspath(args.filename))
        vdursd = {f: get_length(f) for f in mov}
        clip_time = clip_time or len(mov) + 1

        # 2. Analyze audio
        beats = detect_beats(audfile, startat)
        times = extract_beat_times(beats, threshold)
        durations = compute_segment_durations(times)

        _, new_audio = beats_clip(audfile, startat)

        # 3. Select clips
        subclips = create_subclips(vtype, mov, vdursd, durations, clip_time)

        # save json
        argsdict = copy.copy(args.__dict__)
        del argsdict["func"]
        json_file = os.path.join(vdir, "highlights.json")
        write_subclips_json(json_file, {"args": argsdict, "subclips": subclips})

        # total_duration = sum(e - s for _, s, e, _ in subclips)
        # print("Total duration of output video:", total_duration)

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="highlights") as tempdir:
            os.chdir(tempdir)

            trimmed = trim_and_get_outfiles(subclips)
            make_video(trimmed, "combined_withffmpeg.mp4")

            final_clip = mpy.VideoFileClip("combined_withffmpeg.mp4")

            out_name = os.path.join(
                vdir,
                f"{prefix}_{Path(audfile).stem[:10]}_{vtype}_highlights_t_{threshold}_{clip_time}.mp4",
            )

            generate_video_hl(
                [],
                new_audio,
                out_name,
                fps=fps,
                fadeout=fadeout,
                afadeout=afadeout,
                clip=final_clip,
            )
        os.chdir(cwd)

    def hello(self, args):
        # this routine will be called when "vidtoolz "highlights is called."
        print("Hello! This is an example ``vidtoolz`` plugin.")


def generate_video_cuts(
    video_dict: Dict[str, float],
    intervals: List[float],
    max_cuts: int,
    min_gap: float = 0.5,
    max_gap: float = 1.0,
) -> Dict[str, List[Tuple[float, float]]]:

    result = {}

    for video_path, duration in video_dict.items():
        cuts = []
        current_time = 0.0
        interval_iter = cycle(intervals)

        for _ in range(max_cuts):
            gap = random.uniform(min_gap, max_gap)
            start = current_time + gap

            if start >= duration:
                break  # no space even to start the next cut

            interval = next(interval_iter)
            end = start + interval

            if end > duration:
                end = duration  # truncate the last cut

            cuts.append((start, end))
            current_time = end

        result[video_path] = cuts

    return result


def trim_and_get_outfiles_for_coninous(subclips, slow=0.1):
    inum = 1
    outfiles = []
    for item, val in subclips.items():
        fname = os.path.splitext(os.path.basename(item))[0]
        if len(val) == 0:
            dur = get_length(item)
            st = 0.0
            et = st + dur
            outfile = "{0}_output_{1}_s.mp4".format(inum, fname)
            trim_by_ffmpeg(item, st, et, outfile, dur)
            if os.path.exists(outfile):
                outfiles.append(outfile)
                inum = inum + 1
        for st, et in val:
            # round the time to 2 decimal place.
            st, et = np.round(st, 2), np.round(et, 2)
            dur = et - st
            if dur > 5.0:
                outfile = "{0}_output_{1}.mp4".format(inum, fname)
            else:
                if choices([0, 1], weights=[1 - slow, slow]):
                    outfile = "{0}_output_{1}_s.mp4".format(inum, fname)
                else:
                    outfile = "{0}_output_{1}.mp4".format(inum, fname)
            trim_by_ffmpeg(item, st, et, outfile, dur)
            if os.path.exists(outfile):
                outfiles.append(outfile)
                inum = inum + 1
    return outfiles


class ViztoolzPluginStitch:
    """Stitch videos with music"""

    __name__ = "stitch"

    @vidtoolz.hookimpl
    def register_commands(self, subparser):
        self.parser = create_parser2(subparser)
        self.parser.set_defaults(func=self.run)

    def run(self, args):
        audfile = args.audfile
        startat = args.startat
        threshold = args.threshold
        fps = args.fps
        fadeout = args.fadeout
        afadeout = args.afadeout
        prefix = args.prefix
        howmany = args.howmany

        # 1. Read inputs
        mov = read_orderfile(args.filename)
        vdir = os.path.dirname(os.path.abspath(args.filename))
        vdursd = {f: get_length(f) for f in mov}

        # 2. Analyze audio
        beats = detect_beats(audfile, startat)
        times = extract_beat_times(beats, threshold)
        durations = compute_segment_durations(times)

        _, new_audio = beats_clip(audfile, startat)

        # 3. Select clips
        subclips = generate_video_cuts(vdursd, durations, howmany)

        # save json
        argsdict = copy.copy(args.__dict__)
        del argsdict["func"]
        json_file = os.path.join(vdir, "stitch.json")
        write_subclips_json(json_file, {"args": argsdict, "subclips": subclips})
        vtype = "linear"
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="stitch") as tempdir:
            os.chdir(tempdir)

            trimmed = trim_and_get_outfiles_for_coninous(subclips)
            make_video(trimmed, "combined_withffmpeg.mp4")

            final_clip = mpy.VideoFileClip("combined_withffmpeg.mp4")

            out_name = os.path.join(
                vdir,
                f"{prefix}_{Path(audfile).stem[:10]}_{vtype}_stitch_t_{threshold}_hm_{howmany}.mp4",
            )

            generate_video_hl(
                [],
                new_audio,
                out_name,
                fps=fps,
                fadeout=fadeout,
                afadeout=afadeout,
                clip=final_clip,
            )
        os.chdir(cwd)

    def hello(self, args):
        # this routine will be called when "vidtoolz "highlights is called."
        print("Hello! This is an example ``vidtoolz`` plugin.")


highlights_plugin = ViztoolzPlugin()
stitch_plugin = ViztoolzPluginStitch()
