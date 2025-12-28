import copy
import json
import logging
import os
import random
import subprocess
import tempfile
from itertools import cycle
from pathlib import Path
from random import choices
from typing import Dict, List, Optional, Tuple, Union

import moviepy as mpy
import numpy as np
import vidtoolz
from moviepy import afx, vfx
from vidtoolz_beats import detect_beats

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Custom exceptions
class VideoDurationError(Exception):
    """Exception raised when video duration cannot be determined."""

    pass


class VideoProcessingError(Exception):
    """Exception raised when video processing fails."""

    pass


class FileValidationError(Exception):
    """Exception raised when file validation fails."""

    pass


# Constants for magic numbers
LONG_CLIP_THRESHOLD = 6.0  # Seconds
VERY_LONG_CLIP_THRESHOLD = 5.0  # Seconds
MIN_CLIP_DURATION = 2.0  # Seconds
SHORT_CLIP_DURATION = 1.0  # Seconds
SLOW_MOTION_SPEED = 1
NORMAL_SPEED = 0


def _determine_clip_speed_and_span(span: float) -> Tuple[int, float]:
    """
    Determine the playback speed and adjusted span for a clip.

    Args:
        span: Original duration of the clip

    Returns:
        Tuple of (speed, adjusted_span)
    """
    if span > LONG_CLIP_THRESHOLD:
        # Long clips play at normal speed
        return NORMAL_SPEED, span
    else:
        # Short clips play in slow motion (half speed, so double duration)
        return SLOW_MOTION_SPEED, span / 2


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
        default=-0.1,
    )
    parser.add_argument(
        "-vt",
        "--vtype",
        type=str,
        help="Vtype Linear or Non Linear (default: %(default)s)",
        choices=["L", "NL"],
        default="NL",
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
    parser.add_argument(
        "-sh",
        "--skipheader",
        type=int,
        help="Skip headers in filename (default: %(default)s)",
        default=0,
    )
    parser.add_argument(
        "-sf",
        "--skipfooter",
        type=int,
        help="Skip footer in filename (default: %(default)s)",
        default=0,
    )
    return parser


def read_orderfile(fname, skipheader=0, skipfooter=0):
    fname = os.path.abspath(fname)
    fdir = os.path.dirname(fname)
    with open(fname, "r") as fin:
        files = fin.readlines()

    # Return the lines excluding the header and footer
    if skipfooter == 0:
        files = files[skipheader:]
    else:
        files = files[skipheader:-skipfooter]

    mov = [os.path.join(fdir, f.strip()) for f in files]
    return mov


def get_length(filename: str) -> float:
    """Get video duration using ffprobe, fallback to moviepy if it fails.

    Args:
        filename: Path to video file

    Returns:
        Video duration in seconds

    Raises:
        VideoDurationError: If duration cannot be determined
    """
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
        logger.debug(
            f"Successfully got duration {duration} for {filename} using ffprobe"
        )
        return duration
    except Exception as e:
        logger.warning(
            f"ffprobe failed for {filename}, falling back to moviepy. Reason: {e}"
        )
        try:
            with mpy.VideoFileClip(filename) as clip:
                duration = clip.duration
                logger.debug(
                    f"Successfully got duration {duration} for {filename} using moviepy"
                )
                return duration
        except Exception as ex:
            logger.error(f"Failed to get duration via moviepy for {filename}: {ex}")
            raise VideoDurationError(
                f"Could not determine duration for {filename}"
            ) from ex


def beats_clip(audfile, offset=0.0):
    song_name = os.path.basename(audfile)

    snd = mpy.AudioFileClip(audfile)
    new_audioclip = mpy.CompositeAudioClip([snd.subclipped(start_time=offset)])
    return song_name, new_audioclip


def get_non_linear_subclips_VDURS(
    mov: List[str], vdurs: Dict[str, float], dur: List[float], time: float
) -> List[Tuple[str, float, float, int]]:
    """
    Generate non-linear subclips from video files.

    Args:
        mov: List of video file paths
        vdurs: Dictionary mapping file paths to their durations
        dur: List of durations for clip segments
        time: Total target time for the output video

    Returns:
        List of tuples (file, start_time, end_time, speed)
    """
    subclips = []
    cc = cycle(dur)
    cumdur = 0

    logger.debug(
        f"Starting non-linear subclip generation for {len(mov)} videos, target time: {time}s"
    )

    while cumdur <= time:
        if not mov:
            logger.warning("No more valid files to choose from.")
            break

        span = next(cc)
        cumdur = cumdur + span
        max_retries = len(mov) * 2

        for _ in range(max_retries):
            file = np.random.choice(mov)
            try:
                duration = vdurs[file]
            except KeyError:
                logger.warning(f"{file} not found in vdurs")
                continue

            # Determine speed and adjusted span
            speed, span = _determine_clip_speed_and_span(span)

            if duration - span > 0:
                start_time = np.random.uniform(0, duration - span)
                break
        else:
            # If the loop completes without breaking, it means no suitable file was found
            # Fallback to using the last chosen file and starting from the beginning
            start_time = 0
            logger.warning(
                f"Could not find suitable clip after {max_retries} retries, using fallback"
            )

        # Remove files based on duration criteria
        if int(duration / 2) <= 1:
            mov.remove(file)
            logger.debug(f"Removed {file} due to short duration")
        if span > LONG_CLIP_THRESHOLD:  # Remove if a big cut has been done from a file
            mov.remove(file)
            logger.debug(f"Removed {file} after large clip extraction")

        end_time = start_time + span
        subclips.append((file, start_time, end_time, speed))
        logger.debug(
            f"Added subclip: {span:.2f}s segment, cumulative: {cumdur:.2f}s, file: {file}, range: {start_time:.2f}-{end_time:.2f}s, speed: {speed}"
        )

    logger.info(f"Generated {len(subclips)} subclips totaling {cumdur:.2f} seconds")
    return subclips


def get_linear_subclips(
    mov: List[str], vdurs: Dict[str, float], dur: List[float], ntime: int
) -> List[Tuple[str, float, float, int]]:
    """
    Generate linear subclips from video files.

    Args:
        mov: List of video file paths
        vdurs: Dictionary mapping file paths to their durations
        dur: List of durations for clip segments
        ntime: Number of time segments to generate

    Returns:
        List of tuples (file, start_time, end_time, speed)
    """
    subclips = []
    cc = cycle(mov)
    nd = cycle(dur)

    logger.debug(
        f"Starting linear subclip generation for {len(mov)} videos, {ntime - 1} segments"
    )

    for i in range(ntime - 1):
        span = next(nd)
        max_retries = len(mov) * 2

        for _ in range(max_retries):
            file = next(cc)
            try:
                duration = vdurs[file]
            except KeyError:
                logger.warning(f"{file} not found in vdurs")
                continue

            # Determine speed based on clip length
            speed = 0 if span < 0.3 else 1

            if duration <= span:
                start_time = 0.0  # take the whole
                span = duration
                logger.debug(f"Using entire video {file} (duration: {duration}s)")
                break

            if span > VERY_LONG_CLIP_THRESHOLD:  # if more than 10 sec video don't slow
                speed = 0

            if speed == 1:
                span /= 2

            if duration - span > 0:
                start_time = np.random.uniform(0, duration - span)
                break
        else:
            # Fallback if no suitable clip is found after all retries
            start_time = 0.0
            logger.warning(
                f"Could not find suitable clip after {max_retries} retries for segment {i + 1}, using fallback"
            )
            # Ensure span is not greater than the duration of the last checked file
            if "duration" in locals() and duration < span:
                span = duration
                logger.debug(f"Adjusted span to {span}s to fit video duration")

        end_time = start_time + span
        subclips.append((file, start_time, end_time, speed))
        logger.debug(
            f"Added linear subclip {i + 1}: {file}, {start_time:.2f}-{end_time:.2f}s, speed: {speed}"
        )

    logger.info(f"Generated {len(subclips)} linear subclips")
    return subclips


def get_seconds(ts):
    secs = sum(int(x) * 60**i for i, x in enumerate(reversed(ts.split(":"))))
    return secs


def trim_by_ffmpeg(
    inputfile: str,
    starttime: Union[str, float],
    endtime: Union[str, float],
    outputfile: str,
    duration: Optional[float] = None,
) -> int:
    """
    Trim video using ffmpeg.

    Args:
        inputfile: Input video file path
        starttime: Start time (can be string with colons or float)
        endtime: End time (can be string with colons or float)
        outputfile: Output file path
        duration: Optional duration override

    Returns:
        Return code from ffmpeg process

    Raises:
        VideoProcessingError: If ffmpeg command fails
    """
    # Convert time formats if needed
    if isinstance(starttime, str):
        if ":" in starttime:
            starttime = get_seconds(starttime)
    if isinstance(endtime, str):
        if ":" in endtime:
            endtime = get_seconds(endtime)

    # Build ffmpeg command
    try:
        if duration is not None:
            cmdline = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{float(starttime):0.4f}",
                "-i",
                inputfile,
                "-t",
                f"{float(duration):0.4f}",
                "-c",
                "copy",
                outputfile,
            ]
        else:
            cmdline = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{float(starttime):0.4f}",
                "-i",
                inputfile,
                "-to",
                f"{float(endtime):0.4f}",
                "-map",
                "0",
                "-vcodec",
                "copy",
                "-acodec",
                "copy",
                outputfile,
            ]

        logger.debug(f"Running ffmpeg command: {' '.join(cmdline)}")
        result = subprocess.run(cmdline, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"ffmpeg failed with return code {result.returncode}")
            logger.error(f"ffmpeg stderr: {result.stderr}")
            raise VideoProcessingError(f"ffmpeg command failed: {result.stderr}")

        return result.returncode

    except Exception as e:
        logger.error(f"Error running ffmpeg command: {e}")
        raise VideoProcessingError(f"Failed to trim video {inputfile}: {e}") from e


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
            iret = os.system(cmdline)
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
    vc: Optional[List] = None,
    new_audioclip: mpy.AudioClip = None,
    outfile: str = None,
    fps: int = 30,
    fadeout: float = 1,
    afadeout: float = 2,
    clip: Optional[mpy.VideoClip] = None,
) -> mpy.VideoClip:
    """
    Generate final video with highlights and audio.

    Args:
        vc: List of video clips (if clip is None)
        new_audioclip: Audio clip to use
        outfile: Output file path
        fps: Frames per second
        fadeout: Video fadeout duration
        afadeout: Audio fadeout duration
        clip: Optional pre-made video clip

    Returns:
        Final video clip with audio

    Raises:
        VideoProcessingError: If video generation fails
    """
    try:
        if clip is None:
            if not vc:
                raise VideoProcessingError(
                    "No video clips provided and no pre-made clip"
                )
            clip = mpy.concatenate_videoclips(vc, method="compose")
            clip = clip.with_effects([vfx.FadeOut(fadeout)])
            logger.debug("Created video clip from individual clips")
        else:
            logger.debug("Using pre-made video clip")

        # Extract audio from video clip
        try:
            audiofile = os.path.join(os.path.dirname(outfile), "out_audio.mp3")
            clip.audio.write_audiofile(audiofile, fps=44100)
            logger.debug(f"Extracted audio to {audiofile}")
        except Exception as ex:
            logger.warning(f"Failed to extract audio from video clip: {ex}")
            # This is not critical, continue without extracted audio

        clipduration = clip.duration
        min, sec = divmod(clipduration, 60)
        logger.info(
            f"Duration of generated clip is {clipduration:.2f} seconds "
            f"or {int(min):.0f}:{int(sec):.0f}"
        )

        # Handle audio synchronization
        if new_audioclip.duration < clipduration:
            naudio = new_audioclip.with_effects([afx.AudioLoop(duration=clipduration)])
            logger.debug(f"Looped audio to match video duration ({clipduration:.2f}s)")
        else:
            naudio = new_audioclip.with_duration(clipduration)
            logger.debug(f"Trimmed audio to match video duration ({clipduration:.2f}s)")

        naudio = naudio.with_effects([afx.AudioFadeOut(afadeout)])
        logger.debug(f"Applied audio fadeout of {afadeout}s")

        # Combine video and audio
        clip_withsound = clip.with_audio(naudio)
        logger.info(f"Writing final video to {outfile} with fps={fps}")

        clip_withsound.write_videofile(
            outfile,
            temp_audiofile="out.m4a",
            audio_codec="aac",
            fps=fps,
        )
        logger.info(f"Successfully wrote video to {outfile}")

        clip.close()
        return clip_withsound

    except Exception as e:
        logger.error(f"Failed to generate video highlights: {e}")
        raise VideoProcessingError(f"Video generation failed: {e}") from e


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


def _replace_space(text):
    return text.replace(" ", "_")


class ViztoolzPlugin:
    """Make highlights from videos"""

    __name__ = "highlights"

    @vidtoolz.hookimpl
    def register_commands(self, subparser):
        self.parser = create_parser(subparser)
        self.parser.set_defaults(func=self.run)

    def run(self, args):
        """Main execution method for highlights plugin."""
        try:
            # Validate and extract arguments
            self._validate_args(args)

            audfile = args.audfile
            startat = args.startat
            threshold = args.threshold
            vtype = args.vtype
            clip_time = args.clip_time
            fps = args.fps
            fadeout = args.fadeout
            afadeout = args.afadeout
            prefix = args.prefix
            skipheader = args.skipheader
            skipfooter = args.skipfooter

            logger.info(
                f"Starting highlights generation with vtype={vtype}, threshold={threshold}"
            )

            # 1. Read inputs
            logger.debug(f"Reading input files from {args.filename}")
            mov = read_orderfile(args.filename, skipheader, skipfooter)
            if not mov:
                raise FileValidationError("No valid video files found in input file")

            vdir = os.path.dirname(os.path.abspath(args.filename))

            # Get video durations with error handling
            vdursd = {}
            for f in mov:
                try:
                    vdursd[f] = get_length(f)
                    logger.debug(f"Video {f}: duration {vdursd[f]:.2f}s")
                except VideoDurationError as e:
                    logger.warning(f"Skipping video {f} due to duration error: {e}")
                    continue

            if not vdursd:
                raise FileValidationError("No valid videos with determinable durations")

            clip_time = clip_time or len(vdursd) + 1
            logger.debug(f"Target clip time: {clip_time}s")

            # 2. Analyze audio
            logger.debug(f"Analyzing audio file {audfile}")
            beats = detect_beats(audfile, startat)
            times = extract_beat_times(beats, threshold)
            durations = compute_segment_durations(times)
            logger.debug(f"Found {len(durations)} beat segments")

            _, new_audio = beats_clip(audfile, startat)

            # 3. Select clips
            logger.debug("Generating subclips")
            subclips = create_subclips(
                vtype, list(vdursd.keys()), vdursd, durations, clip_time
            )
            logger.info(f"Generated {len(subclips)} subclips")

            # save json
            argsdict = copy.copy(args.__dict__)
            del argsdict["func"]
            json_file = os.path.join(vdir, "highlights.json")
            write_subclips_json(json_file, {"args": argsdict, "subclips": subclips})
            logger.debug(f"Saved subclips metadata to {json_file}")

            # Process videos in temporary directory
            cwd = os.getcwd()
            try:
                with tempfile.TemporaryDirectory(prefix="highlights") as tempdir:
                    logger.debug(f"Working in temporary directory: {tempdir}")
                    os.chdir(tempdir)

                    # Trim and process videos
                    trimmed = trim_and_get_outfiles(subclips)
                    if not trimmed:
                        raise VideoProcessingError(
                            "No valid trimmed video files generated"
                        )

                    make_video(trimmed, "combined_withffmpeg.mp4")

                    final_clip = mpy.VideoFileClip("combined_withffmpeg.mp4")

                    out_name = os.path.join(
                        vdir,
                        f"{prefix}_{_replace_space(str(Path(audfile).stem[:10]))}_{vtype}_highlights_t_{threshold}_{clip_time}.mp4",
                    )

                    logger.info(f"Generating final video: {out_name}")
                    generate_video_hl(
                        [],
                        new_audio,
                        out_name,
                        fps=fps,
                        fadeout=fadeout,
                        afadeout=afadeout,
                        clip=final_clip,
                    )
                    logger.info(f"Successfully created highlights video: {out_name}")
            finally:
                os.chdir(cwd)
                logger.debug(f"Restored working directory to {cwd}")

        except Exception as e:
            logger.error(f"Highlights generation failed: {e}")
            raise

    def _validate_args(self, args):
        """Validate command line arguments."""
        if not os.path.exists(args.filename):
            raise FileValidationError(f"Input file {args.filename} does not exist")

        if args.audfile and not os.path.exists(args.audfile):
            raise FileValidationError(f"Audio file {args.audfile} does not exist")

        if args.clip_time is not None and args.clip_time <= 0:
            raise FileValidationError("clip_time must be positive")

        if args.threshold < -1.0 or args.threshold > 1.0:
            raise FileValidationError("threshold must be between -1.0 and 1.0")

        if args.fps <= 0:
            raise FileValidationError("fps must be positive")

        if args.fadeout < 0:
            raise FileValidationError("fadeout must be non-negative")

        if args.afadeout < 0:
            raise FileValidationError("afadeout must be non-negative")

        logger.debug("Arguments validated successfully")

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
            st, et = np.round(st, 3), np.round(et, 3)
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


def create_video_using_subclips_json(subclips, out_name=None):
    # Load JSON data from file
    with open(subclips, "r") as f:
        data = json.load(f)

    basename = os.path.basename(subclips)
    # basename, _ = os.path.split(basename)

    vdir = os.path.dirname(os.path.abspath(subclips))

    # Extract top-level parameters
    args = data["args"]
    subclips = data["subclips"]

    # Unpack args
    audfile = args.get("audfile")
    startat = args.get("startat", 0.0)
    fps = args.get("fps", 30)
    fadeout = args.get("fadeout", 0)
    afadeout = args.get("afadeout", 0)
    vtype = args.get("vtype", "L")
    prefix = args.get("prefix", "IMG")
    threshold = args.get("threshold", 0.3)
    howmany = args.get("howmany", 5)

    fname = f"{_replace_space(basename[:10])}_{prefix}_{_replace_space(str(Path(audfile).stem[:10]))}_{vtype}_stitch_t_{threshold}_hm_{howmany}"
    out_name = os.path.join(vdir, f"{fname}.mp4")

    cwd = os.getcwd()
    _, new_audio = beats_clip(audfile, startat)
    with tempfile.TemporaryDirectory(prefix="stitch") as tempdir:
        os.chdir(tempdir)

        trimmed = trim_and_get_outfiles_for_coninous(subclips)
        make_video(trimmed, "combined_withffmpeg.mp4")

        final_clip = mpy.VideoFileClip("combined_withffmpeg.mp4")

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
        howmany = args.howmany

        # 1. Read inputs
        mov = read_orderfile(args.filename)
        vdir = os.path.dirname(os.path.abspath(args.filename))
        vdursd = {f: get_length(f) for f in mov}

        # 2. Analyze audio
        beats = detect_beats(audfile, startat)
        times = extract_beat_times(beats, threshold)
        durations = compute_segment_durations(times)
        # durations = [1,2,3,5]

        # 3. Select clips
        subclips = generate_video_cuts(vdursd, durations, howmany)
        basename = os.path.basename(args.filename)
        # save json
        vtype = "linear"
        argsdict = copy.copy(args.__dict__)
        del argsdict["func"]
        fname = f"{basename[:10]}_{Path(audfile).stem[:10]}_{vtype}_stitch_t_{threshold}_hm_{howmany}"
        json_file = os.path.join(vdir, f"{fname}.json")
        write_subclips_json(json_file, {"args": argsdict, "subclips": subclips})
        create_video_using_subclips_json(json_file)

    def hello(self, args):
        # this routine will be called when "vidtoolz "highlights is called."
        print("Hello! This is an example ``vidtoolz`` plugin.")


highlights_plugin = ViztoolzPlugin()
stitch_plugin = ViztoolzPluginStitch()
