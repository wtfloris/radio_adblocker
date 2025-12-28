import sys
import tty
import select
import librosa
import asyncio
import termios
import numpy as np
import subprocess as sp

from time import sleep
from rich.status import Status
from datetime import datetime as dt
from datetime import timedelta
from scipy.spatial.distance import cosine

from pyatv import scan as ap_scan
from pyatv import connect as ap_connect
from pyatv.interface import BaseConfig

STREAM_URL = "https://icecast.omroep.nl/radio2-bb-mp3"
REFERENCE_START_FILES = ["audio/jingle_start.wav", "audio/jingle_start_2.wav"]
REFERENCE_END_FILES = ["audio/jingle_end.wav", "audio/jingle_end_2.wav"]
SAMPLING_RATE = 16000
SIMILARITY_THRESHOLD = 0.9

JINGLE_COUNTER_OFFSET = 0
AD_WINDOW_START_MINUTE = 52
AD_WINDOW_END_MINUTE = 10
DETECTION_ADJUST_DELAY = 5

VOLUME_ADS = 0
VOLUME_DEFAULT = 2
AIRPLAY_TARGETS = {
    "Kitchen": 66.6,
    # "Office": 100.0,
    "Living Room": 100.0,
    "Living Room HomePod R": 100.0,
}

HELP_STR = "Actions: [D] Default volume  [L] Lowest volume  [M] Mute  [0-9] Set volume"


def get_file_mfcc(file_path: str, target_sr: int = SAMPLING_RATE, n_mfcc: int = 13) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=target_sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return mfcc


def get_stream_chunk_mfcc(stream_url: str, duration: int = 10) -> np.ndarray:
    cmd = [
        'ffmpeg', '-i', stream_url, '-acodec', 'pcm_s16le', '-f',
        's16le', '-ac', '1', '-ar', str(SAMPLING_RATE), '-t', str(duration), '-'
    ]
    result = sp.run(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL)
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLING_RATE, n_mfcc=13)
    return mfcc


def compare_fragment(fragment_mfcc: np.ndarray, reference_mfccs: list[np.ndarray]) -> bool:
    for reference_mfcc in reference_mfccs:
        fragment_len = fragment_mfcc.shape[1]
        reference_len = reference_mfcc.shape[1]
        reference_flat = reference_mfcc.flatten()

        for offset in range(1, fragment_len - reference_len + 1):
            fragment_flat = fragment_mfcc[:, offset:offset+reference_len]
            similarity = 1 - cosine(reference_flat, fragment_flat.flatten())
            if similarity > SIMILARITY_THRESHOLD:
                return True
    return False


async def scan_airplay_devices(status: Status) -> list[BaseConfig]:
    loop = asyncio.get_running_loop()
    airplay_configs = await ap_scan(loop)

    log = f"Found {len(airplay_configs)} AirPlay devices:"
    if not airplay_configs:
        log = "Warning: no AirPlay devices found"
    for config in airplay_configs:
        log = log + f"\n  - {config.name}"
    status.console.log(log)
    return airplay_configs


async def set_volume(volume: int | float, airplay_configs: list[BaseConfig], status: Status):
    loop = asyncio.get_running_loop()
    for config in airplay_configs:
        if config.name in AIRPLAY_TARGETS.keys():
            volume_abs = AIRPLAY_TARGETS[config.name] * (volume/9)
            status.update(f"Connecting to {config.name}")
            conn = await ap_connect(config, loop)

            try:
                status.update(f"Setting volume on {config.name}")
                await conn.audio.set_volume(round(volume_abs, 1))
            except:
                pass
            finally:
                conn.close()


def wait_for_key_with_timeout(sleep_time: int, sleep_msg: str, status: Status, timeout: int = 1):
    sleep_until_dt = dt.now() + timedelta(seconds=sleep_time)
    sleep_msg = sleep_msg + "\n  " + HELP_STR
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while dt.now() < sleep_until_dt:
            diff = (sleep_until_dt - dt.now()).seconds
            status.update(sleep_msg % diff)
            r, _, _ = select.select([sys.stdin], [], [], float(timeout))
            if r:
                ch = sys.stdin.read(1)
                if ch == "d":
                    status.console.log("Keypress: D -> setting default volume")
                    asyncio.run(set_volume(VOLUME_DEFAULT, airplay_configs, status))
                elif ch == "m" or ch == "0":
                    status.console.log("Keypress: M -> muting")
                    asyncio.run(set_volume(0, airplay_configs, status))
                elif ch == "l":
                    status.console.log("Keypress: L -> setting lowest audible volume")
                    asyncio.run(set_volume(0.01, airplay_configs, status))
                elif ch in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    status.console.log(f"Keypress: {ch} -> setting volume")
                    asyncio.run(set_volume(int(ch), airplay_configs, status))
            else:
                pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


with Status("Initializing") as status:
    status.update("Getting AirPlay devices")
    airplay_configs = asyncio.run(scan_airplay_devices(status))
    
    status.update("Processing reference audio")
    reference_start_mfccs: list[np.ndarray] = []
    reference_end_mfccs: list[np.ndarray] = []
    for file in REFERENCE_START_FILES:
        reference_start_mfccs.append(get_file_mfcc(file))
        status.console.log(f"Reference (start) audio from {file} loaded")
    reference_mfccs = reference_start_mfccs
    for file in REFERENCE_END_FILES:
        reference_end_mfccs.append(get_file_mfcc(file))
        status.console.log(f"Reference (end) audio from {file} loaded")
    
    jingle_counter = JINGLE_COUNTER_OFFSET
    cooldown = 60
    ads_finished = False
    state_msg = "Inside ad start window"

    while True:
        if dt.now().minute > AD_WINDOW_START_MINUTE or dt.now().minute < AD_WINDOW_END_MINUTE and not ads_finished:
            status.update("Getting stream chunk")
            fragment_mfcc = get_stream_chunk_mfcc(STREAM_URL, 10)
            if len(fragment_mfcc) > 0:
                status.update("Comparing stream chunk to reference")
                if compare_fragment(fragment_mfcc, reference_mfccs):
                    status.console.log(f"Reference audio detected at {dt.now().strftime("%H:%M:%S")}")
                    if jingle_counter == 0:
                        status.console.log("Ads starting")
                        wait_for_key_with_timeout(DETECTION_ADJUST_DELAY, "Ads starting, adjusting in %s seconds", status)
                        asyncio.run(set_volume(VOLUME_ADS, airplay_configs, status))
                        sleep(1)
                        asyncio.run(set_volume(VOLUME_ADS, airplay_configs, status))
                        jingle_counter += 1
                        state_msg = "Ads running, detected 1/3 jingles"
                        cooldown = 60 * (60 - dt.now().minute) + 60
                    elif jingle_counter == 1:
                        jingle_counter += 1
                        state_msg = "Ads running, detected 2/3 jingles"
                        reference_mfccs = reference_end_mfccs
                        cooldown = 60
                    elif jingle_counter == 2:
                        status.console.log("Ads ending")
                        wait_for_key_with_timeout(DETECTION_ADJUST_DELAY, "Ads ending, adjusting in %s seconds", status)
                        asyncio.run(set_volume(VOLUME_DEFAULT/2, airplay_configs, status))
                        sleep(1)
                        asyncio.run(set_volume(VOLUME_DEFAULT, airplay_configs, status))
                        jingle_counter = 0
                        state_msg = "Inside ad window"
                        reference_mfccs = reference_start_mfccs
                        ads_finished = True
                        cooldown = 0
                    wait_for_key_with_timeout(cooldown, "Detected reference audio, cooldown for %s seconds", status)
            wait_for_key_with_timeout(5, f"{state_msg}, sleeping for %s seconds", status)
        elif jingle_counter > 0 and not ads_finished:
            status.console.log("Ads window ended with missing jingles (likely failed detection), adjusting volume up")
            asyncio.run(set_volume(VOLUME_DEFAULT/2, airplay_configs, status))
            sleep(1)
            asyncio.run(set_volume(VOLUME_DEFAULT, airplay_configs, status))
            jingle_counter = 0
            state_msg = "Inside ad window"
            reference_mfccs = reference_start_mfccs
            ads_finished = True
        else:
            ads_finished = False
            time_until_window = (60 - dt.now().second) + 60 * (AD_WINDOW_START_MINUTE - dt.now().minute)
            wait_for_key_with_timeout(time_until_window, "Outside ad window, sleeping for %s seconds", status)
