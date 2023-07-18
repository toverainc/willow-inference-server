import asyncio
import logging
import io

# FFMPEG (for now)
import av

from aiortc.mediastreams import MediaStreamError, MediaStreamTrack


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

    def __init__(self, file=None, format='wav', options={}):
        if file is None:
            file = io.BytesIO()
        self.file = file
        logger.debug(f'RTC MEDIA: MediaRecorderLite using file {file} format {format} options {options}')
        self.__container = av.open(file=file, format=format, mode="w", options=options)
        self.__tracks = {}

    def addTrack(self, track):
        """
        Add a track to be recorded.

        :param track: A :class:`aiortc.MediaStreamTrack`.
        """
        # WAV
        codec_name = "pcm_s16le"
        stream = self.__container.add_stream(codec_name, rate=16000)
        self.__tracks[track] = MediaRecorderLiteContext(stream)

    def start(self):
        logger.debug('RTC MEDIA: Called start')
        """
        Start recording.
        """
        for track, context in self.__tracks.items():
            if context.task is None:
                context.task = asyncio.ensure_future(self.__run_track(track, context))

    def stop(self):
        logger.debug('RTC MEDIA: Called stop')
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
                logger.debug('RTC MEDIA: Closing container')
                self.__container.close()
                self.__container = None

    async def __run_track(self, track: MediaStreamTrack, context: MediaRecorderLiteContext):
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError as e:
                logger.error("RTC MEDIA: STREAM ERROR", e)
                return

            if not context.started:
                context.started = True

            for packet in context.stream.encode(frame):
                self.__container.mux(packet)
