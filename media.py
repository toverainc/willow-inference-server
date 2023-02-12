import asyncio
import errno
import fractions
import logging
import threading
import time
from typing import Dict, Optional, Set, Union

import av
from av import AudioFrame, VideoFrame
from av.audio import AudioStream
from av.frame import Frame
from av.packet import Packet
from av.video.stream import VideoStream

from aiortc.mediastreams import AUDIO_PTIME, MediaStreamError, MediaStreamTrack

logger = logging.getLogger(__name__)

class MediaRecorderLiteContext:
    def __init__(self, stream):
        self.started = False
        self.stream = stream
        self.task = None

class MediaRecorderLite:
    """
    A media sink that writes audio and/or video to a file.

    Examples:

    .. code-block:: python

        # Write to a video file.
        player = MediaRecorder('/path/to/file.mp4')

        # Write to a set of images.
        player = MediaRecorder('/path/to/file-%3d.png')

    :param file: The path to a file, or a file-like object.
    :param format: The format to use, defaults to autodect.
    :param options: Additional options to pass to FFmpeg.
    """

    def __init__(self, file, format=None, options={}):
        self.__container = av.open(file=file, format=format, mode="w", options=options)
        self.__tracks = {}

    def addTrack(self, track):
        """
        Add a track to be recorded.

        :param track: A :class:`aiortc.MediaStreamTrack`.
        """
        if track.kind == "audio":
            if self.__container.format.name in ("wav", "alsa"):
                codec_name = "pcm_s16le"
            elif self.__container.format.name == "mp3":
                codec_name = "mp3"
            else:
                codec_name = "aac"
            stream = self.__container.add_stream(codec_name)
        else:
            if self.__container.format.name == "image2":
                stream = self.__container.add_stream("png", rate=30)
                stream.pix_fmt = "rgb24"
            else:
                stream = self.__container.add_stream("libx264", rate=30)
                stream.pix_fmt = "yuv420p"
        self.__tracks[track] = MediaRecorderLiteContext(stream)

    async def start(self):
        """
        Start recording.
        """
        for track, context in self.__tracks.items():
            if context.task is None:
                context.task = asyncio.ensure_future(self.__run_track(track, context))

    async def stop(self):
        """
        Stop recording.
        """
        if self.__container:
            for track, context in self.__tracks.items():
                if context.task is not None:
                    context.task.cancel()
                    context.task = None
                    for packet in context.stream.encode(None):
                        self.__container.mux(packet)
            self.__tracks = {}

            if self.__container:
                self.__container.close()
                self.__container = None

    async def __run_track(self, track: MediaStreamTrack, context: MediaRecorderLiteContext):
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return

            if not context.started:
                # adjust the output size to match the first frame
                if isinstance(frame, VideoFrame):
                    context.stream.width = frame.width
                    context.stream.height = frame.height
                context.started = True

            for packet in context.stream.encode(frame):
                self.__container.mux(packet)
