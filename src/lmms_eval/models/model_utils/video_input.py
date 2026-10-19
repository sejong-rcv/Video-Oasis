import math
from dataclasses import dataclass
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class VideoSpec:
    path: str
    start_time: float | None = None
    end_time: float | None = None

    @property
    def has_time_bound(self) -> bool:
        return self.start_time is not None


def parse_video_input(value: Any) -> VideoSpec | None:
    """Normalize a video path or a (path, start, end) tuple."""
    if isinstance(value, str):
        return VideoSpec(path=value)

    if not isinstance(value, tuple):
        return None
    if len(value) != 3:
        if value and isinstance(value[0], str):
            raise ValueError(f"Expected video tuple (path, start_time, end_time), got {value!r}")
        return None

    path, start_time, end_time = value
    if not isinstance(path, str):
        return None
    if isinstance(start_time, bool) or not isinstance(start_time, Real):
        raise TypeError(f"Video start_time must be numeric, got {type(start_time).__name__}")
    if isinstance(end_time, bool) or not isinstance(end_time, Real):
        raise TypeError(f"Video end_time must be numeric, got {type(end_time).__name__}")

    start_time = float(start_time)
    end_time = float(end_time)
    if not math.isfinite(start_time) or not math.isfinite(end_time) or start_time < 0 or start_time >= end_time:
        raise ValueError(f"Invalid video time range: start_time={start_time}, end_time={end_time}")

    return VideoSpec(path=path, start_time=start_time, end_time=end_time)


def validate_video_bounds(start_time, end_time, total_frames, fps):
    """Check the requested interval before clamping frame indices."""
    if total_frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video metadata: {total_frames} frames @ {fps} fps")
    duration = total_frames / fps
    start = 0.0 if start_time is None else start_time
    end = duration if end_time is None else end_time
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= end:
        raise ValueError(f"Invalid video time range: [{start}, {end}], duration={duration:.6f}s")
    # Allow at most one frame of rounding error at the video end.
    if start >= duration or end > duration + 1 / fps + 1e-6:
        raise ValueError(f"Video segment [{start}, {end}] exceeds duration={duration:.6f}s")


def add_video_bounds(video_dict: dict, spec: VideoSpec) -> dict:
    """Attach canonical Qwen video bounds without changing unbounded inputs."""
    if spec.has_time_bound:
        video_dict["video_start"] = spec.start_time
        video_dict["video_end"] = spec.end_time
    return video_dict
