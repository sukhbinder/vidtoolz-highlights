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
from typing import Dict, List, Tuple

import moviepy as mpy
import numpy as np
import vidtoolz
from moviepy import afx, vfx
from vidtoolz_beats import detect_beats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for magic numbers
MIN_VIDEO_DURATION = 2.0  # Minimum duration for a video clip
LONG_CLIP_THRESHOLD = 6.0  # Threshold for considering a clip "long"
SHORT_CLIP_THRESHOLD = 5.0  # Threshold for considering a clip "short"
SLOW_MOTION_SPEED = 0.5  # Speed factor for slow motion clips


# Custom exceptions
class VideoDurationError(Exception):
    """Exception raised when video duration cannot be determined."""
    pass


class VideoProcessingError(Exception):
    """Exception raised when video processing fails."""
    pass


class FileNotFoundError(Exception):
    """Exception raised when required files are not found."""
    pass


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
        nargs="+",
        help="mp3 Audio file (default: %(default)s)",
        default=None,
    )
    parser.add_argument(
        "-st",
        "--startat",
        type=float,
        nargs="+",
        help="Audio startat (default: %(default)s)",
        default=None,
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
        nargs="+",
        help="mp3 Audio file (default: %(default)s)",
        default=None,
    )

    parser.add_argument(
        "-st",
        "--startat",
        type=float,
        nargs="+",
        help="Audio startat (default: %(default)s)",
        default=None,
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
        logger.debug(f"Successfully got duration {duration} for {filename} using ffprobe")
        return duration
    except Exception as e:
        logger.warning(f"ffprobe failed for {filename}, falling back to moviepy. Reason: {e}")
        try:
            with mpy.VideoFileClip(filename) as clip:
                duration = clip.duration
                logger.debug(f"Successfully got duration {duration} for {filename} using moviepy")
                return duration
        except Exception as ex:
            logger.error(f"Failed to get duration via moviepy for {filename}: {ex}")
            raise VideoDurationError(f"Could not determine duration for {filename}") from ex


def beats_clip(audfile: str, offset: float = 0.0) -> Tuple[str, mpy.CompositeAudioClip]:
    """Create an audio clip from a file with optional offset.
    
    Args:
        audfile: Path to audio file
        offset: Start offset in seconds
        
    Returns:
        Tuple of (song_name, audio_clip)
    """
    song_name = os.path.basename(audfile)

    snd = mpy.AudioFileClip(audfile)
    new_audioclip = mpy.CompositeAudioClip([snd.subclipped(start_time=offset)])
    return song_name, new_audioclip


def get_non_linear_subclips_VDURS(mov, vdurs, dur, time):
    """Generate non-linear subclips from video files.
    
    Args:
        mov: List of video file paths
        vdurs: Dictionary mapping file paths to their durations
        dur: List of duration values to cycle through
        time: Total target time for subclips
        
    Returns:
        List of tuples containing (file_path, start_time, end_time, speed)
    """
    subclips = []
    cc = cycle(dur)
    cumdur = 0
    
    logger.debug(f"Starting non-linear subclip generation with {len(mov)} files, target time: {time}")
    
    while cumdur <= time:
        if not mov:
            logger.warning("No more valid files to choose from.")
            break
            
        span = next(cc)
        cumdur = cumdur + span
        max_retries = len(mov) * 2
        
        logger.debug(f"Looking for clip with span {span}, cumulative duration: {cumdur}")
        
        for _ in range(max_retries):
            file = np.random.choice(mov)
            try:
                duration = vdurs[file]
            except KeyError:
                logger.error(f"{file} not found in vdurs")
                continue

            # Determine speed based on clip length
            speed = 0 if span > LONG_CLIP_THRESHOLD else 1
            if speed == 1:
                span /= 2

            if duration - span > 0:
                start_time = np.random.uniform(0, duration - span)
                break
        else:
            # If the loop completes without breaking, it means no suitable file was found
            # Fallback to using the last chosen file and starting from the beginning
            start_time = 0
            logger.warning(f"No suitable file found after {max_retries} retries, using fallback")

        # remove file for less than 2 sec duration
        if int(duration / 2) <= 1:
            mov.remove(file)
            logger.debug(f"Removed {file} due to short duration")
        if span > LONG_CLIP_THRESHOLD:  # Remove if a big cut has been done from a file
            mov.remove(file)
            logger.debug(f"Removed {file} after large clip extraction")
            
        end_time = start_time + span
        subclips.append((file, start_time, end_time, speed))
        logger.debug(f"Added subclip: {file} [{start_time:.2f}-{end_time:.2f}] speed={speed}")

    logger.info(f"Generated {len(subclips)} subclips with total duration {cumdur:.2f}s")
    return subclips


def get_linear_subclips(mov, vdurs, dur, ntime):
    """Generate linear subclips from video files.
    
    Args:
        mov: List of video file paths
        vdurs: Dictionary mapping file paths to their durations
        dur: List of duration values to cycle through
        ntime: Number of clips to generate
        
    Returns:
        List of tuples containing (file_path, start_time, end_time, speed)
    """
    subclips = []
    cc = cycle(mov)
    nd = cycle(dur)
    
    logger.debug(f"Starting linear subclip generation for {ntime-1} clips from {len(mov)} files")
    
    for i in range(ntime - 1):
        span = next(nd)
        max_retries = len(mov) * 2
        
        logger.debug(f"Processing clip {i+1}/{ntime-1}, target span: {span}")
        
        for _ in range(max_retries):
            file = next(cc)
            try:
                duration = vdurs[file]
            except KeyError:
                logger.error(f"{file} not found in vdurs")
                continue

            # Determine speed based on clip length
            speed = 0 if span < 0.3 else 1
            if duration <= span:
                start_time = 0.0  # take the whole
                span = duration
                logger.debug(f"Using entire file {file} (duration: {duration})")
                break

            if span > SHORT_CLIP_THRESHOLD:  # if more than 10 sec video don't slow
                speed = 0
            if speed == 1:
                span /= 2

            if duration - span > 0:
                start_time = np.random.uniform(0, duration - span)
                break
        else:
            # Fallback if no suitable clip is found after all retries
            start_time = 0.0
            logger.warning(f"No suitable clip found after {max_retries} retries, using fallback")
            # Ensure span is not greater than the duration of the last checked file
            if "duration" in locals() and duration < span:
                span = duration
                logger.debug(f"Adjusted span to {span} to match file duration")
                
        end_time = start_time + span
        subclips.append((file, start_time, end_time, speed))
        logger.debug(f"Added subclip: {file} [{start_time:.2f}-{end_time:.2f}] speed={speed}")
    
    logger.info(f"Generated {len(subclips)} linear subclips")
    return subclips


def get_seconds(ts: str) -> float:
    """Convert time string (HH:MM:SS) to seconds.
    
    Args:
        ts: Time string in HH:MM:SS format
        
    Returns:
        Time in seconds as float
    """
    secs = sum(int(x) * 60**i for i, x in enumerate(reversed(ts.split(":"))))
    return float(secs)


def trim_by_ffmpeg(inputfile, starttime, endtime, outputfile, duration=None):
    """Trim video using ffmpeg.
    
    Args:
        inputfile: Input video file path
        starttime: Start time (can be string with HH:MM:SS format or float)
        endtime: End time (can be string with HH:MM:SS format or float)
        outputfile: Output file path
        duration: Optional duration parameter
        
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

    try:
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
        
        logger.debug(f"Executing ffmpeg command: {cmdline}")
        cmdlist = cmdline.split()
        iret = subprocess.call(cmdlist)
        
        if iret != 0:
            logger.error(f"ffmpeg command failed with return code {iret}: {cmdline}")
            raise VideoProcessingError(f"ffmpeg failed to process {inputfile}")
        
        logger.debug(f"Successfully trimmed {inputfile} to {outputfile}")
        return iret
        
    except Exception as e:
        logger.error(f"Error in trim_by_ffmpeg: {e}")
        raise VideoProcessingError(f"Failed to trim video: {e}") from e


def trim_and_get_outfiles(sc: List[Tuple[str, float, float, int]]) -> List[str]:
    """Trim video clips and return list of output file paths.
    
    Args:
        sc: List of subclips as tuples (filename, start_time, end_time, speed)
        
    Returns:
        List of output file paths
    """
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


def make_video(files: List[str], fname: str) -> int:
    """Combine video files into a single video.
    
    Args:
        files: List of input video file paths
        fname: Output file path
        
    Returns:
        Return code from ffmpeg process
    """
    base_name = os.path.basename(fname)
    bname, ext = os.path.splitext(base_name)
    out_file = "{}_mylist.txt".format(bname)
    # Slow the videos using ffmpeg
    sfiles = []
    for f in files:
        if f.endswith("_s.mp4"):
            outf = "{0}-s.mp4".format(f)
            cmdline = "ffmpeg -i {0} -an -filter:v 'setpts=2.0*PTS' {1}".format(f, outf)
            logger.debug(f"Executing: {cmdline}")
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
    logger.debug(f"Executing: {cmdline}")
    iret = os.system(cmdline)
    logger.info(f"Combined {len(files)} videos into {fname}")
    return iret


def generate_video_hl(
    vc, new_audioclip, outfile, fps=30, fadeout=1, afadeout=2, clip=None
):
    """Generate final video with highlights and audio.
    
    Args:
        vc: List of video clips
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
            logger.debug("Concatenating video clips")
            clip = mpy.concatenate_videoclips(vc, method="compose")
            clip = clip.with_effects([vfx.FadeOut(fadeout)])

        # Extract audio from video clip
        try:
            audiofile = os.path.join(os.path.dirname(outfile), "out_audio.mp3")
            clip.audio.write_audiofile(audiofile, fps=44100)
            logger.debug(f"Extracted audio to {audiofile}")
        except Exception as ex:
            logger.warning(f"Failed to extract audio: {ex}")
            # Continue without extracted audio

        clipduration = clip.duration
        min, sec = divmod(clipduration, 60)
        logger.info(
            f"Duration of generated clip is {clipduration:.2f} seconds or {min:.0f}:{sec:.0f}"
        )
        
        # Handle audio duration matching
        if new_audioclip.duration < clipduration:
            logger.debug(f"Looping audio to match video duration ({clipduration:.2f}s)")
            naudio = new_audioclip.with_effects([afx.AudioLoop(duration=clipduration)])
        else:
            logger.debug(f"Truncating audio to match video duration ({clipduration:.2f}s)")
            naudio = new_audioclip.with_duration(clipduration)

        naudio = naudio.with_effects([afx.AudioFadeOut(afadeout)])
        clip_withsound = clip.with_audio(naudio)
        
        logger.info(f"Writing final video to {outfile} with fps={fps}")
        clip_withsound.write_videofile(
            outfile,
            temp_audiofile="out.m4a",
            audio_codec="aac",
            fps=fps,
        )
        clip.close()
        logger.info(f"Successfully generated video: {outfile}")
        return clip_withsound
        
    except Exception as e:
        logger.error(f"Error in generate_video_hl: {e}")
        raise VideoProcessingError(f"Failed to generate video: {e}") from e


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
            # Validate input files
            if not os.path.exists(args.filename):
                raise FileNotFoundError(f"Input file {args.filename} not found")
            
            audfiles = args.audfile
            startats = args.startat if args.startat is not None else []
            temp_audio_filepath = None

            # Validate audio files
            if audfiles:
                for audfile in audfiles:
                    if not os.path.exists(audfile):
                        raise FileNotFoundError(f"Audio file {audfile} not found")

            # Pad startats with 0.0 if it's shorter than audfiles
            if len(startats) < len(audfiles):
                startats.extend([0.0] * (len(audfiles) - len(startats)))

            logger.info(f"Processing {len(audfiles)} audio files with start times: {startats}")

            clips = []
            for f, st in zip(audfiles, startats):
                try:
                    clip = mpy.AudioFileClip(f)
                    if st > 0:
                        clip = clip.subclipped(st)
                    clips.append(clip)
                    logger.debug(f"Loaded audio clip {f} with start time {st}")
                except Exception as e:
                    raise VideoProcessingError(f"Failed to load audio file {f}: {e}") from e

            final_clip = mpy.concatenate_audioclips(clips)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_f:
                final_clip.write_audiofile(temp_f.name)
                temp_audio_filepath = temp_f.name
                logger.debug(f"Created temporary audio file: {temp_audio_filepath}")

            audfile = temp_audio_filepath
            startat = 0.0

            # Extract parameters
            threshold = args.threshold
            vtype = args.vtype
            clip_time = args.clip_time
            fps = args.fps
            fadeout = args.fadeout
            afadeout = args.afadeout
            prefix = args.prefix
            skipheader = args.skipheader
            skipfooter = args.skipfooter

            logger.info(f"Starting highlights generation with parameters: vtype={vtype}, threshold={threshold}, clip_time={clip_time}")

            # 1. Read inputs
            mov = read_orderfile(args.filename, skipheader, skipfooter)
            vdir = os.path.dirname(os.path.abspath(args.filename))
            
            # Validate video files
            for video_file in mov:
                if not os.path.exists(video_file):
                    raise FileNotFoundError(f"Video file {video_file} not found")
            
            logger.info(f"Processing {len(mov)} video files")
            
            # Get video durations with error handling
            vdursd = {}
            for f in mov:
                try:
                    vdursd[f] = get_length(f)
                    logger.debug(f"Video {f} duration: {vdursd[f]:.2f}s")
                except VideoDurationError as e:
                    logger.warning(f"Skipping video {f} due to duration error: {e}")
                    continue
            
            if not vdursd:
                raise VideoProcessingError("No valid video files found")
            
            clip_time = clip_time or len(vdursd) + 1

            # 2. Analyze audio
            logger.info("Analyzing audio beats...")
            beats = detect_beats(audfile, startat)
            times = extract_beat_times(beats, threshold)
            durations = compute_segment_durations(times)
            logger.debug(f"Found {len(times)} beats, {len(durations)} segments")

            _, new_audio = beats_clip(audfile, startat)

            # 3. Select clips
            logger.info("Selecting video clips...")
            subclips = create_subclips(vtype, list(vdursd.keys()), vdursd, durations, clip_time)
            logger.info(f"Generated {len(subclips)} subclips")

            # save json
            argsdict = copy.copy(args.__dict__)
            del argsdict["func"]
            json_file = os.path.join(vdir, "highlights.json")
            write_subclips_json(json_file, {"args": argsdict, "subclips": subclips})
            logger.debug(f"Saved subclips metadata to {json_file}")

            cwd = os.getcwd()
            with tempfile.TemporaryDirectory(prefix="highlights") as tempdir:
                logger.debug(f"Working in temporary directory: {tempdir}")
                os.chdir(tempdir)

                try:
                    logger.info("Trimming video clips...")
                    trimmed = trim_and_get_outfiles(subclips)
                    logger.info(f"Successfully trimmed {len(trimmed)} clips")
                    
                    logger.info("Creating combined video...")
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
                    logger.info(f"Successfully generated highlights video: {out_name}")
                    
                finally:
                    os.chdir(cwd)
                    logger.debug(f"Restored working directory to {cwd}")

        except Exception as e:
            logger.error(f"Error in highlights generation: {e}")
            raise
        finally:
            if temp_audio_filepath and os.path.exists(temp_audio_filepath):
                try:
                    os.remove(temp_audio_filepath)
                    logger.debug(f"Cleaned up temporary audio file: {temp_audio_filepath}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_audio_filepath}: {e}")

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
        """Main execution method for stitch plugin."""
        try:
            # Validate input files
            if not os.path.exists(args.filename):
                raise FileNotFoundError(f"Input file {args.filename} not found")
            
            audfiles = args.audfile
            startats = args.startat if args.startat is not None else []
            temp_audio_filepath = None

            # Validate audio files
            if audfiles:
                for audfile in audfiles:
                    if not os.path.exists(audfile):
                        raise FileNotFoundError(f"Audio file {audfile} not found")

            # Pad startats with 0.0 if it's shorter than audfiles
            if len(startats) < len(audfiles):
                startats.extend([0.0] * (len(audfiles) - len(startats)))

            logger.info(f"Processing {len(audfiles)} audio files with start times: {startats}")

            clips = []
            for f, st in zip(audfiles, startats):
                try:
                    clip = mpy.AudioFileClip(f)
                    if st > 0:
                        clip = clip.subclipped(st)
                    clips.append(clip)
                    logger.debug(f"Loaded audio clip {f} with start time {st}")
                except Exception as e:
                    raise VideoProcessingError(f"Failed to load audio file {f}: {e}") from e

            final_clip = mpy.concatenate_audioclips(clips)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_f:
                final_clip.write_audiofile(temp_f.name)
                temp_audio_filepath = temp_f.name
                logger.debug(f"Created temporary audio file: {temp_audio_filepath}")

            audfile = temp_audio_filepath
            startat = 0.0

            # Extract parameters
            threshold = args.threshold
            howmany = args.howmany

            logger.info(f"Starting stitch generation with parameters: threshold={threshold}, howmany={howmany}")

            # 1. Read inputs
            mov = read_orderfile(args.filename)
            vdir = os.path.dirname(os.path.abspath(args.filename))
            
            # Validate video files
            for video_file in mov:
                if not os.path.exists(video_file):
                    raise FileNotFoundError(f"Video file {video_file} not found")
            
            logger.info(f"Processing {len(mov)} video files")
            
            # Get video durations with error handling
            vdursd = {}
            for f in mov:
                try:
                    vdursd[f] = get_length(f)
                    logger.debug(f"Video {f} duration: {vdursd[f]:.2f}s")
                except VideoDurationError as e:
                    logger.warning(f"Skipping video {f} due to duration error: {e}")
                    continue
            
            if not vdursd:
                raise VideoProcessingError("No valid video files found")

            # 2. Analyze audio
            logger.info("Analyzing audio beats...")
            beats = detect_beats(audfile, startat)
            times = extract_beat_times(beats, threshold)
            durations = compute_segment_durations(times)
            logger.debug(f"Found {len(times)} beats, {len(durations)} segments")

            # 3. Select clips
            logger.info("Generating video cuts...")
            subclips = generate_video_cuts(vdursd, durations, howmany)
            logger.info(f"Generated cuts for {len(subclips)} videos")
            
            basename = os.path.basename(args.filename)
            # save json
            vtype = "linear"
            argsdict = copy.copy(args.__dict__)
            del argsdict["func"]
            fname = f"{basename[:10]}_{Path(audfile).stem[:10]}_{vtype}_stitch_t_{threshold}_hm_{howmany}"
            json_file = os.path.join(vdir, f"{fname}.json")
            write_subclips_json(json_file, {"args": argsdict, "subclips": subclips})
            logger.debug(f"Saved subclips metadata to {json_file}")
            
            logger.info("Creating final stitched video...")
            create_video_using_subclips_json(json_file)
            logger.info(f"Successfully generated stitched video")
            
        except Exception as e:
            logger.error(f"Error in stitch generation: {e}")
            raise
        finally:
            if temp_audio_filepath and os.path.exists(temp_audio_filepath):
                try:
                    os.remove(temp_audio_filepath)
                    logger.debug(f"Cleaned up temporary audio file: {temp_audio_filepath}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_audio_filepath}: {e}")

    def hello(self, args):
        # this routine will be called when "vidtoolz "highlights is called."
        print("Hello! This is an example ``vidtoolz`` plugin.")


highlights_plugin = ViztoolzPlugin()
stitch_plugin = ViztoolzPluginStitch()
