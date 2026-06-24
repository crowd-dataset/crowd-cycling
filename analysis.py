from tqdm import tqdm
from typing import Optional, Set
import math
from custom_logger import CustomLogger
from logmod import logs
from urllib.parse import urljoin, urlparse
from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
import requests
import os
import subprocess
import common
import pathlib
from bs4 import BeautifulSoup
import polars as pl
import cv2
from types import SimpleNamespace                # lightweight config container
from utils.bicyclist_detect import Algorithm
from utils.bicyclist_following import (
    CyclistFollowing,
    FollowingParams,
    TRAFFIC_CONTROL_PROXY_CLASS_IDS,
    YOLO_STOP_SIGN_CLASS,
    YOLO_TRAFFIC_LIGHT_CLASS,
)
from utils.analytics.io import IO
from utils.core.tools import Tools
from utils.analytics.geo import Geo


tools = Tools()
algo = Algorithm()
cf = CyclistFollowing(algo)
analytics_IO = IO()
geo = Geo()

# Common junk files/folders to ignore.
MISC_FILES: Set[str] = {"DS_Store"}


logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


# -----------------------------------------------------------------------------
# USER PARAMETERS (edit these variables; no argparse / no CLI parser is used)
# -----------------------------------------------------------------------------
def _cfg_bool(key: str, default: bool) -> bool:
    """Best-effort boolean config lookup with a safe fallback.

    Handles real booleans from JSON as well as string values such as
    "true"/"false", "yes"/"no", and "1"/"0".
    """
    try:
        v = common.get_configs(key)
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1", "yes", "y", "on"}:
                return True
            if s in {"false", "0", "no", "n", "off", ""}:
                return False
            return default
        if isinstance(v, (int, float)):
            return bool(v)
        return default
    except Exception:
        return default


def _cfg_float(key: str, default: float) -> float:
    """Best-effort float config lookup with a safe fallback."""
    try:
        return float(common.get_configs(key))
    except Exception:
        return default


def _cfg_float_opt(key: str, default: Optional[float] = None) -> Optional[float]:
    """Best-effort optional float config lookup.

    Returns default when config is missing, None, or an empty string.
    """
    try:
        v = common.get_configs(key)
        if v is None:
            return default
        if isinstance(v, str) and v.strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def _cfg_int(key: str, default: int) -> int:
    """Best-effort integer config lookup with a safe fallback."""
    try:
        v = common.get_configs(key)
        if v is None:
            return default
        if isinstance(v, str) and v.strip() == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _cfg_int_list(key: str, default: list[int]) -> list[int]:
    """Best-effort integer-list config lookup with a safe fallback.

    Supports lists from JSON as well as strings such as "9,11" or "[9, 11]".
    """
    try:
        v = common.get_configs(key)
        if v is None:
            return list(default)

        raw_values = []
        if isinstance(v, (list, tuple, set)):
            raw_values = list(v)
        elif isinstance(v, str):
            s = v.strip()
            if not s:
                return list(default)
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            raw_values = [part.strip() for part in s.split(",") if part.strip()]
        else:
            raw_values = [v]

        out: list[int] = []
        for item in raw_values:
            try:
                out.append(int(float(item)))
            except Exception:
                continue

        return out if out else list(default)
    except Exception:
        return list(default)


# When True: if cyclist-following episodes are detected in a CSV, the script will
# download the corresponding source video and generate an annotated video showing
# the involved cyclists (and their bicycles) across the entire CSV segment.
DOWNLOAD_AND_ANNOTATE: bool = _cfg_bool("DOWNLOAD_AND_ANNOTATE", True)

# Output folders
videos_dir: str = common.get_configs("videos")
DOWNLOADED_VIDEOS_DIR = os.path.join(videos_dir, "downloaded_video")
ANNOTATED_VIDEOS_DIR = os.path.join(videos_dir, "annotated_video")
TRIMMED_CLIPS_DIR = os.path.join(videos_dir, "trimmed_video")

# If True, keep the intermediate trimmed clip on disk. If False, it will be deleted
# after the annotated video is written successfully.
KEEP_TRIMMED_CLIP: bool = _cfg_bool("KEEP_TRIMMED_CLIP", False)

# If True, delete the full downloaded source video after all annotation jobs for a CSV
# finish or fail. This only deletes files inside DOWNLOADED_VIDEOS_DIR.
DELETE_DOWNLOADED_VIDEO_ON_COMPLETE: bool = _cfg_bool("DELETE_DOWNLOADED_VIDEO_ON_COMPLETE", False)

# Annotation controls
# If True, always draw IDs for ALL detected bicyclists (not only those involved in following episodes).
ANNOTATE_ALL_BICYCLISTS: bool = _cfg_bool("ANNOTATE_ALL_BICYCLISTS", True)
# If True, render a text overlay each frame showing active follower->leader pairs.
ANNOTATE_PAIR_OVERLAY: bool = _cfg_bool("ANNOTATE_PAIR_OVERLAY", True)
# If True, always write a full-segment annotated video for each CSV segment that has bicyclists,
# regardless of whether a following episode is detected. Cropped clips (if enabled) are still
# produced only when following exists.
ANNOTATE_WHOLE_SEGMENT: bool = _cfg_bool("ANNOTATE_WHOLE_SEGMENT", False)


# Cropping controls (optional)
# If True: when following episodes exist, write cropped annotated clips around each follower->leader pair.
# Crop window definition:
#   - start: CROP_PRE_SECONDS before the first encounter (episode start_frame)
#   - end  : CROP_POST_GONE_SECONDS after BOTH leader and follower have disappeared from the frame
#           (within the CSV segment; clamped to segment bounds).
CROP_AROUND_FOLLOWING: bool = _cfg_bool("CROP_AROUND_FOLLOWING", False)
CROP_PRE_SECONDS: float = _cfg_float("CROP_PRE_SECONDS", 5.0)
CROP_POST_GONE_SECONDS: float = _cfg_float("CROP_POST_GONE_SECONDS", 6.0)
# If set (>0): require leader & follower track overlap for at least this many seconds.
# If empty/missing: include all.
MIN_CO_VISIBLE_SECONDS: Optional[float] = _cfg_float_opt("MIN_CO_VISIBLE_SECONDS", None)
# If True and MIN_CO_VISIBLE_SECONDS is set (>0): do not skip crop clips that fail the co-visibility
# requirement. Instead, still annotate them and save them into a separate output folder.
ANNOTATE_BELOW_MIN_CO_VISIBLE_SECONDS: bool = _cfg_bool("ANNOTATE_BELOW_MIN_CO_VISIBLE_SECONDS", False)
# If True and CROP_AROUND_FOLLOWING is enabled, also write the full-segment annotated video.
ALSO_WRITE_FULL_SEGMENT_WHEN_CROPPING: bool = _cfg_bool("ALSO_WRITE_FULL_SEGMENT_WHEN_CROPPING", False)


# Crossing proxy controls (traffic light / stop sign)
# COCO YOLO class ids: traffic light=9, stop sign=11. If your model uses different
# ids, set TRAFFIC_CONTROL_PROXY_CLASSES in config accordingly.
YOLO_TRAFFIC_LIGHT_CLASS = 9
YOLO_STOP_SIGN_CLASS = 11
FILTER_FOLLOWING_BY_TRAFFIC_CONTROL_PROXY: bool = _cfg_bool(
    "FILTER_FOLLOWING_BY_TRAFFIC_CONTROL_PROXY",
    True,
)
TRAFFIC_CONTROL_PROXY_CLASSES: list[int] = _cfg_int_list(
    "TRAFFIC_CONTROL_PROXY_CLASSES",
    TRAFFIC_CONTROL_PROXY_CLASS_IDS,
)
# Keep an episode if at least TRAFFIC_CONTROL_PROXY_MIN_DETECTIONS traffic-control
# detections occur between episode start/end, expanded by this many seconds.
TRAFFIC_CONTROL_PROXY_FRAME_BUFFER_SECONDS: float = _cfg_float(
    "TRAFFIC_CONTROL_PROXY_FRAME_BUFFER_SECONDS",
    3.0,
)
# Stricter reviewer-aligned crossing event gates. These are used when filtering
# following episodes before annotation. The old temporal proxy is kept for
# backwards compatibility, while these defaults reject common false positives:
# ordinary platooning, unrelated traffic lights/signs elsewhere in the frame,
# and cases where the follower crosses too late after the leader.
CROSSING_EVENT_PROXY_FRAME_BUFFER_SECONDS: float = _cfg_float(
    "CROSSING_EVENT_PROXY_FRAME_BUFFER_SECONDS",
    0.75,
)
CROSSING_EVENT_PROXY_MAX_PAIR_DISTANCE: Optional[float] = _cfg_float_opt(
    "CROSSING_EVENT_PROXY_MAX_PAIR_DISTANCE",
    0.35,
)
CROSSING_EVENT_PROXY_SAME_FRAME_TOLERANCE_SECONDS: float = _cfg_float(
    "CROSSING_EVENT_PROXY_SAME_FRAME_TOLERANCE_SECONDS",
    0.50,
)
MAX_MEAN_TIME_HEADWAY_SECONDS: Optional[float] = _cfg_float_opt(
    "MAX_MEAN_TIME_HEADWAY_SECONDS",
    3.0,
)
MAX_P90_TIME_HEADWAY_SECONDS: Optional[float] = _cfg_float_opt(
    "MAX_P90_TIME_HEADWAY_SECONDS",
    5.0,
)
TRAFFIC_CONTROL_PROXY_MIN_DETECTIONS: int = max(
    1,
    _cfg_int("TRAFFIC_CONTROL_PROXY_MIN_DETECTIONS", 1),
)
# Optional extra confidence threshold for traffic lights / stop signs. Leave empty
# or missing to rely on the global CSV confidence filter.
TRAFFIC_CONTROL_PROXY_MIN_CONFIDENCE: Optional[float] = _cfg_float_opt(
    "TRAFFIC_CONTROL_PROXY_MIN_CONFIDENCE",
    None,
)
# Draw traffic light / stop sign boxes in the annotated videos for manual checking.
ANNOTATE_TRAFFIC_CONTROL_PROXY: bool = _cfg_bool("ANNOTATE_TRAFFIC_CONTROL_PROXY", True)

TRAFFIC_CONTROL_PROXY_CLASS_LABELS: dict[int, str] = {
    YOLO_TRAFFIC_LIGHT_CLASS: "traffic-light",
    YOLO_STOP_SIGN_CLASS: "stop-sign",
}


# -----------------------------------------------------------------------------
# ANNOTATION COLORS (BGR)
# -----------------------------------------------------------------------------
# follower  : cyclist that is following another cyclist
# leader    : cyclist that is being followed (leader in episodes table)
# normal    : cyclist that is not in a following episode
COLOR_CYCLIST_FOLLOWER = (0, 0, 255)   # red
COLOR_CYCLIST_FOLLOWING = (0, 255, 0)  # green
COLOR_CYCLIST_LEADER = COLOR_CYCLIST_FOLLOWING  # alias for clarity
COLOR_CYCLIST_NORMAL = (0, 215, 255)   # orange/yellow-ish (visible on most backgrounds)
COLOR_BICYCLE = (255, 0, 0)            # blue
COLOR_TRAFFIC_LIGHT = (0, 255, 255)    # yellow/cyan-ish in BGR
COLOR_STOP_SIGN = (255, 255, 255)      # white
COLOR_TRAFFIC_CONTROL_PROXY = (255, 255, 0)


class Analysis():

    def download_videos_from_ftp(self, filename: str, base_url: Optional[str] = None, out_dir: str = ".",
                                 username: Optional[str] = None, password: Optional[str] = None,
                                 token: Optional[str] = None, timeout: int = 20, debug: bool = True,
                                 max_pages: int = 500) -> Optional[tuple[str, str, str, float]]:
        """
        Search and download a specific .mp4 file from a multi-directory FastAPI-based
        HTTP file server (e.g., files.mobility-squad.com). This function attempts direct
        download from known /files/ paths (tue1/tue2/tue3), and if not found, recursively
        crawls the /browse pages to locate the video file. Progress is shown with tqdm.
        Args: filename (str): Target file name (with or without .mp4 extension).
        base_url (str, optional): Base URL of the file server.
        Must include protocol, e.g. "https://files.mobility-squad.com/".
        out_dir (str, optional): Local output directory to save the video.
        Defaults to current directory ".".
        username (str, optional): Username for HTTP Basic Auth.
        password (str, optional): Password for HTTP Basic Auth.
        token (str, optional): Token string for token-based authentication.
        Sent as a query parameter ?token=.... timeout (int, optional):
        Request timeout in seconds. Default is 20. max_pages (int, optional):
        Safety limit for crawl depth/pages. Default is 500.
        Returns: Optional[Tuple[str, str, str, float]]: Returns a tuple
        (local_path, filename, resolution_label, fps) if the download succeeds,
        or None if the file is not found or download fails.
        Logging: - logger.info: start, success summaries.
        - logger.debug: HTTP requests, crawl steps, file matches.
        - logger.warning: non-fatal issues (metadata failures, skipped pages).
        - logger.error: fatal errors (network/IO exceptions). Example:

            result = self.download_videos_from_http_fileserver(
                filename="3ai7SUaPoHM",
                base_url="https://files.mobility-squad.com/",
                out_dir="./downloads",
                username="mobility",
                password="your_password"
            )
            if result:
                path, name, res, fps = result
                print(f"Downloaded {name} ({res}, {fps} fps) to {path}")
            else:
                print("File not found or failed.")
        """
        # -------------------- Input Preparation --------------------
        if not base_url:
            logger.error("Base URL is missing.")
            return None

        base = base_url if base_url.endswith("/") else base_url + "/"

        if username == "":
            username = None
        if password == "":
            password = None

        filename_with_ext = filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"
        filename_lower = filename_with_ext.lower()

        # Local cache: if the file is already present on disk, reuse it.
        os.makedirs(out_dir, exist_ok=True)
        cached_path = os.path.join(out_dir, filename_with_ext)
        if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
            resolution, fps_meta = "unknown", 0.0
            try:
                fps_meta = float(Analysis.get_video_fps(cached_path))
                resolution = Analysis.get_video_resolution_label(cached_path)
            except Exception:
                pass
            logger.info(f"Using cached video: {cached_path}")
            return cached_path, filename_with_ext, resolution, fps_meta
        aliases = ["tue1", "tue2", "tue3", "tue4"]

        req_params = {"token": token} if token else None

        logger.info(f"Starting download for '{filename_with_ext}'")
        logger.debug(
            f"Base URL: {base} | Auth: {'Basic' if username and password else 'None'} | Token: {'Yes' if token else 'No'}"  # noqa:E501
        )  # noqa: E501

        # ---------- Session ----------
        with requests.Session() as session:
            if username and password:
                session.auth = (username, password)
            session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

            def fetch(url: str, stream: bool = False) -> Optional[requests.Response]:
                """GET with logging and safe error handling."""
                try:
                    r = session.get(url, timeout=timeout, params=req_params, stream=stream)
                    logger.debug(f"GET {url} -> {r.status_code}")
                    if r.status_code == 401:
                        logger.error(f"Authentication failed for {url}")
                    r.raise_for_status()
                    return r
                except requests.RequestException as e:
                    logger.warning(f"Request failed [{url}]: {e}")
                    return None

            # ---------- 1. Try direct /files paths ----------
            for alias in aliases:
                direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
                logger.debug(f"Trying direct URL: {direct_url}")

                r = fetch(direct_url, stream=True)
                if r is None:
                    continue

                logger.info(f"Found file via direct URL: {direct_url}")
                content_len = int(r.headers.get("content-length", 0))
                logger.debug(f"Content-Length: {content_len or 'unknown'} bytes")

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                # Avoid overwriting
                if os.path.exists(local_path):
                    stem, suf = os.path.splitext(local_path)
                    i = 1
                    while os.path.exists(f"{stem} ({i}){suf}"):
                        i += 1
                    local_path = f"{stem} ({i}){suf}"
                    logger.warning(f"File exists, saving as: {local_path}")

                # ---------- Download ----------
                try:
                    total = content_len or None
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading from ftp: {filename_with_ext}",
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if total:
                                    bar.update(len(chunk))
                    logger.info(f"Download complete: {local_path} ({written} bytes)")
                except Exception as e:
                    logger.error(f"Download failed for {filename_with_ext}: {e}")
                    return None

                # ---------- Metadata ----------
                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path))  # type: ignore
                    resolution = Analysis.get_video_resolution_label(local_path)
                    logger.debug(f"Metadata extracted: fps={fps}, resolution={resolution}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                logger.info(f"✅ Saved '{filename_with_ext}' (res={resolution}, fps={fps})")
                return local_path, filename_with_ext, resolution, fps

            # ---------- 2. Crawl /browse fallback ----------
            visited: Set[str] = set()

            def is_dir_link(href: str) -> bool:
                return href.startswith("/v/") and "/browse" in href

            def is_file_link(href: str) -> bool:
                return "/files/" in href

            def crawl(start_url: str) -> Optional[str]:
                """Recursively traverse /browse pages."""
                stack = [start_url]
                pages_seen = 0

                while stack:
                    url = stack.pop()

                    if url in visited:
                        continue

                    visited.add(url)
                    pages_seen += 1
                    if pages_seen > max_pages:
                        logger.warning(f"Crawl aborted after {max_pages} pages.")
                        return None

                    resp = fetch(url)
                    if resp is None:
                        continue

                    try:
                        soup = BeautifulSoup(resp.text, "html.parser")
                    except Exception as e:
                        logger.warning(f"HTML parse failed at {url}: {e}")
                        continue

                    for a in soup.find_all("a"):
                        href = (a.get("href") or "").strip()  # type: ignore
                        if not href:
                            continue

                        full = urljoin(url, href)

                        if is_file_link(href):
                            anchor_text = (a.text or "").strip().lower()
                            tail = pathlib.PurePosixPath(urlparse(full).path).name.lower()
                            if anchor_text == filename_lower or tail == filename_lower:
                                logger.info(f"File located via crawl: {full}")
                                return full

                        if is_dir_link(href):
                            stack.append(full)

                logger.debug("Crawl finished — no file found.")
                return None

            for alias in aliases:
                start_url = urljoin(base, f"v/{alias}/browse")
                logger.debug(f"Crawling alias: {alias} -> {start_url}")

                found = crawl(start_url)
                if not found:
                    continue

                r = fetch(found, stream=True)
                if not r:
                    continue

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                try:
                    total = int(r.headers.get("content-length", 0)) or None
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {filename_with_ext}",
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if total:
                                    bar.update(len(chunk))
                    logger.info(f"Downloaded via crawl: {local_path} ({written} bytes)")
                except Exception as e:
                    logger.error(f"Download during crawl failed: {e}")
                    return None

                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path))  # type: ignore
                    resolution = Analysis.get_video_resolution_label(local_path)
                    logger.debug(f"Metadata: fps={fps}, resolution={resolution}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                return local_path, filename_with_ext, resolution, fps

            logger.warning(f"File '{filename_with_ext}' not found in any alias.")
            return None

    @staticmethod
    def _parse_fps_fraction(value: str) -> Optional[float]:
        """Parse ffprobe frame-rate strings such as '30000/1001' or '29.97'."""
        try:
            text = str(value).strip()
            if not text or text == "0/0":
                return None
            if "/" in text:
                num_s, den_s = text.split("/", 1)
                num = float(num_s)
                den = float(den_s)
                if den == 0:
                    return None
                fps = num / den
            else:
                fps = float(text)
            if fps > 0:
                return float(fps)
        except Exception:
            return None
        return None

    @staticmethod
    def get_video_fps(video_path: str) -> float:
        """Return FPS for a local video file.

        Prefer ffprobe's avg_frame_rate because it preserves values such as
        30000/1001 = 29.970... that are often rounded to 30 elsewhere. Fall
        back to OpenCV when ffprobe is unavailable.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate,r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    fps = Analysis._parse_fps_fraction(line)
                    if fps is not None and fps > 0:
                        return fps
        except Exception:
            pass

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return fps

    @staticmethod
    def infer_frame_count_base(df: pl.DataFrame) -> int:
        """Infer whether CSV frame-count is zero-based or one-based.

        The YOLO CSVs in this project normally start at 1, meaning CSV frame 1
        corresponds to the first video frame of the mapped segment. This helper
        keeps the code safe if an older zero-based CSV appears.
        """
        try:
            if df.height == 0 or "frame-count" not in df.columns:
                return 1
            min_frame = int(df.select(pl.min("frame-count")).item())
            return 0 if min_frame == 0 else 1
        except Exception:
            return 1

    @staticmethod
    def csv_frame_start_seconds(
        segment_start_seconds: float,
        csv_frame: int,
        fps: float,
        frame_count_base: int,
    ) -> float:
        """Map a CSV frame-count to the source-video timestamp at that frame start."""
        if float(fps) <= 0:
            raise ValueError("fps must be positive when converting frame-count to seconds.")
        return float(segment_start_seconds) + (float(int(csv_frame) - int(frame_count_base)) / float(fps))

    @staticmethod
    def csv_frame_end_seconds(
        segment_start_seconds: float,
        csv_frame: int,
        fps: float,
        frame_count_base: int,
    ) -> float:
        """Map an inclusive CSV frame-count to the exclusive source-video end timestamp."""
        if float(fps) <= 0:
            raise ValueError("fps must be positive when converting frame-count to seconds.")
        return float(segment_start_seconds) + (float(int(csv_frame) - int(frame_count_base) + 1) / float(fps))

    def _prepare_annotation_data(
        self,
        *,
        df: pl.DataFrame,
        cyclist_map: pl.DataFrame,
        episodes: pl.DataFrame,
        involved_cyclist_ids: set[int],
        draw_all_bicyclists: bool,
    ):
        """Prepare frame-indexed rows and role lookup for annotation drawing."""
        required_df_cols = {"frame-count", "yolo-id", "unique-id", "x-center", "y-center", "width", "height"}
        missing_df_cols = required_df_cols - set(df.columns)
        if missing_df_cols:
            raise ValueError(f"df is missing required columns: {sorted(missing_df_cols)}")

        has_cyclist_map = cyclist_map is not None and cyclist_map.height > 0
        cyclist_map_has_cols = has_cyclist_map and {"cyclist_id", "bicycle_id"}.issubset(set(cyclist_map.columns))

        cyclist_ids: set[int] = set()
        if draw_all_bicyclists and cyclist_map_has_cols:
            cyclist_ids |= set(map(int, cyclist_map.get_column("cyclist_id").to_list()))
        if involved_cyclist_ids:
            cyclist_ids |= set(map(int, involved_cyclist_ids))
        if not cyclist_ids and cyclist_map_has_cols:
            cyclist_ids = set(map(int, cyclist_map.get_column("cyclist_id").to_list()))

        bicycle_ids: set[int] = set()
        if cyclist_ids and cyclist_map_has_cols:
            bicycle_ids = set(
                map(
                    int,
                    cyclist_map.filter(pl.col("cyclist_id").is_in(list(cyclist_ids)))
                    .get_column("bicycle_id")
                    .to_list(),
                )
            )

        bike_to_cyclist: dict[int, int] = {}
        if cyclist_map_has_cols:
            for r in cyclist_map.select(["bicycle_id", "cyclist_id"]).iter_rows(named=True):
                try:
                    bike_to_cyclist[int(r["bicycle_id"])] = int(r["cyclist_id"])
                except Exception:
                    continue

        leader_col: Optional[str] = None
        if episodes is not None and episodes.height > 0:
            if "leader_id" in episodes.columns:
                leader_col = "leader_id"
            elif "following_id" in episodes.columns:
                leader_col = "following_id"

        intervals: list[tuple[int, int, int, int]] = []
        if leader_col is not None:
            needed = {"start_frame", "end_frame", "follower_id", leader_col}
            missing = needed - set(episodes.columns)
            if missing:
                raise ValueError(f"episodes is missing required columns for roles: {sorted(missing)}")

            for r in episodes.select(["start_frame", "end_frame", "follower_id", leader_col]).iter_rows(named=True):
                intervals.append(
                    (
                        int(r["start_frame"]),
                        int(r["end_frame"]),
                        int(r["follower_id"]),
                        int(r[leader_col]),
                    )
                )

        def roles_for_frame(frame_count: int) -> tuple[dict[int, str], dict[int, int]]:
            roles: dict[int, str] = {}
            active_pairs: dict[int, int] = {}
            for s, e, fid, lid in intervals:
                if s <= frame_count <= e:
                    roles[fid] = "follower"
                    roles[lid] = "leader"
                    active_pairs[fid] = lid
            return roles, active_pairs

        frame_to_rows: dict[int, list[dict]] = {}
        traffic_proxy_class_ids = [int(x) for x in TRAFFIC_CONTROL_PROXY_CLASSES]
        draw_traffic_proxy = bool(ANNOTATE_TRAFFIC_CONTROL_PROXY) and bool(traffic_proxy_class_ids)

        if df.height > 0 and (cyclist_ids or bicycle_ids or draw_traffic_proxy):
            object_filter = (
                ((pl.col("yolo-id") == 0) & (pl.col("unique-id").is_in(list(cyclist_ids)))) |
                ((pl.col("yolo-id") == 1) & (pl.col("unique-id").is_in(list(bicycle_ids))))
            )
            if draw_traffic_proxy:
                object_filter = object_filter | (
                    pl.col("yolo-id").cast(pl.Int64, strict=False).is_in(traffic_proxy_class_ids)
                )

            wanted = (
                df.filter(object_filter)
                .select(["frame-count", "yolo-id", "unique-id", "x-center", "y-center", "width", "height"])
            )
            for row in wanted.iter_rows(named=True):
                fc = int(row["frame-count"])
                frame_to_rows.setdefault(fc, []).append(row)

        return frame_to_rows, roles_for_frame, bike_to_cyclist

    @staticmethod
    def _draw_annotation_frame(
        *,
        frame,
        csv_frame: int,
        frame_to_rows: dict[int, list[dict]],
        roles_for_frame,
        bike_to_cyclist: dict[int, int],
        draw_pair_overlay: bool,
        draw_labels: bool,
    ):
        """Draw annotations for one CSV frame-count on an already decoded image."""
        height, width = frame.shape[:2]
        coords_normalised = True
        roles, active_pairs = roles_for_frame(int(csv_frame))

        for row in frame_to_rows.get(int(csv_frame), []):
            yolo_id = int(row["yolo-id"])
            obj_id = int(row["unique-id"])

            xc = float(row["x-center"])
            yc = float(row["y-center"])
            w = float(row["width"])
            h = float(row["height"])

            if coords_normalised:
                xc *= width
                yc *= height
                w *= width
                h *= height

            x1 = int(round(xc - w / 2.0))
            y1 = int(round(yc - h / 2.0))
            x2 = int(round(xc + w / 2.0))
            y2 = int(round(yc + h / 2.0))

            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))

            if yolo_id == 0:
                role = roles.get(obj_id, "normal")
                if role == "follower":
                    color = COLOR_CYCLIST_FOLLOWER
                elif role == "leader":
                    color = COLOR_CYCLIST_LEADER
                else:
                    color = COLOR_CYCLIST_NORMAL
                label = f"{role}:{obj_id}"
            elif yolo_id == 1:
                color = COLOR_BICYCLE
                rider_id = bike_to_cyclist.get(obj_id)
                label = f"bicycle:{obj_id}" if rider_id is None else f"bicycle:{obj_id} rider:{rider_id}"
            elif yolo_id in set(int(x) for x in TRAFFIC_CONTROL_PROXY_CLASSES):
                if yolo_id == YOLO_TRAFFIC_LIGHT_CLASS:
                    color = COLOR_TRAFFIC_LIGHT
                elif yolo_id == YOLO_STOP_SIGN_CLASS:
                    color = COLOR_STOP_SIGN
                else:
                    color = COLOR_TRAFFIC_CONTROL_PROXY
                class_label = TRAFFIC_CONTROL_PROXY_CLASS_LABELS.get(yolo_id, f"traffic-control-{yolo_id}")
                label = f"{class_label}:{obj_id}"
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if draw_labels:
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        if draw_pair_overlay and active_pairs:
            x0, y0 = 12, 28
            for i, (fid, lid) in enumerate(sorted(active_pairs.items())):
                txt = f"Follower {fid} -> Leader {lid}"
                cv2.putText(
                    frame,
                    txt,
                    (x0, y0 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return frame

    def annotate_following_segment_from_source(
        self,
        *,
        input_video_path: str,
        output_video_path: str,
        df: pl.DataFrame,
        cyclist_map: pl.DataFrame,
        episodes: pl.DataFrame,
        involved_cyclist_ids: set[int],
        source_segment_start_seconds: float,
        csv_start_frame: int,
        csv_end_frame: int,
        csv_fps: float,
        frame_count_base: int = 1,
        draw_all_bicyclists: bool = True,
        draw_pair_overlay: bool = True,
        draw_labels: bool = True,
    ) -> None:
        """Render an annotated clip using CSV frame-count as frame ordinal.

        Important alignment rule:
          CSV frame-count is treated as the ordinal decoded frame number inside
          the mapped segment. For example, with one-based CSVs, frame-count 1 is
          the first decoded frame of the mapped segment.

        The function seeks once to the mapped segment start time, skips decoded
        frames sequentially until ``csv_start_frame``, then reads exactly one
        decoded video frame for each CSV frame. This avoids drift caused by
        rounded or nominal FPS values such as 30 versus 30000/1001.
        """
        out_dir = os.path.dirname(output_video_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        csv_start_frame = int(csv_start_frame)
        csv_end_frame = int(csv_end_frame)
        frame_count_base = int(frame_count_base)
        if csv_end_frame < csv_start_frame:
            raise ValueError(f"Invalid CSV frame range: {csv_start_frame}..{csv_end_frame}")

        frame_to_rows, roles_for_frame, bike_to_cyclist = self._prepare_annotation_data(
            df=df,
            cyclist_map=cyclist_map,
            episodes=episodes,
            involved_cyclist_ids=involved_cyclist_ids,
            draw_all_bicyclists=draw_all_bicyclists,
        )

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        opencv_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError(f"Invalid video dimensions for: {input_video_path} (w={width}, h={height})")

        writer_fps = 0.0
        try:
            writer_fps = float(Analysis.get_video_fps(input_video_path))
        except Exception:
            writer_fps = 0.0
        if writer_fps <= 0:
            writer_fps = opencv_fps if opencv_fps > 0 else float(csv_fps or 0.0)
        if writer_fps <= 0:
            writer_fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(output_video_path, fourcc, float(writer_fps), (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open VideoWriter: {output_video_path}")

        # Seek once to the mapped segment start. After that, do not convert every
        # CSV frame through FPS. Frame-count is an ordinal frame number, so
        # sequential reads are the least drift-prone alignment method.
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(source_segment_start_seconds) * 1000.0))

        frames_to_skip = max(0, int(csv_start_frame) - int(frame_count_base))
        skipped = 0

        try:
            while skipped < frames_to_skip:
                ok, _ = cap.read()
                if not ok:
                    logger.warning(
                        f"Could not skip to csv_frame={csv_start_frame}. "
                        f"Stopped after skipping {skipped}/{frames_to_skip} decoded frames."
                    )
                    break
                skipped += 1

            for csv_frame in range(csv_start_frame, csv_end_frame + 1):
                ok, frame = cap.read()
                if not ok:
                    logger.warning(
                        f"Could not read source frame for csv_frame={csv_frame}. "
                        f"Wrote frames through csv_frame={csv_frame - 1}."
                    )
                    break

                frame = self._draw_annotation_frame(
                    frame=frame,
                    csv_frame=csv_frame,
                    frame_to_rows=frame_to_rows,
                    roles_for_frame=roles_for_frame,
                    bike_to_cyclist=bike_to_cyclist,
                    draw_pair_overlay=draw_pair_overlay,
                    draw_labels=draw_labels,
                )
                out.write(frame)
        finally:
            cap.release()
            out.release()

    def annotate_following_segment(
        self,
        *,
        input_video_path: str,
        output_video_path: str,
        df: pl.DataFrame,
        cyclist_map: pl.DataFrame,
        episodes: pl.DataFrame,
        involved_cyclist_ids: set[int],
        frame_offset: int = 0,
        fps_override: Optional[float] = None,
        draw_all_bicyclists: bool = True,
        draw_pair_overlay: bool = True,
        draw_labels: bool = True,
    ) -> None:
        """
        Render an annotated video for a CSV segment.

        - Reads frames from `input_video_path`.
        - Draws bboxes for cyclists (yolo-id==0) and bicycles (yolo-id!=0, typically 1)
          for the IDs selected from `cyclist_map` and/or `involved_cyclist_ids`.
        - If `episodes` contains intervals, cyclists are labelled as follower/leader when active.
        - Writes output WITHOUT audio via OpenCV VideoWriter.
        - `frame_offset` aligns video frame index -> CSV frame-count.
        """
        out_dir = os.path.dirname(output_video_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        # -----------------------------
        # Defensive column checks
        # -----------------------------
        required_df_cols = {"frame-count", "yolo-id", "unique-id", "x-center", "y-center", "width", "height"}
        missing_df_cols = required_df_cols - set(df.columns)
        if missing_df_cols:
            raise ValueError(f"df is missing required columns: {sorted(missing_df_cols)}")

        # cyclist_map is optional-ish, but if present we expect these cols
        has_cyclist_map = cyclist_map is not None and cyclist_map.height > 0
        cyclist_map_has_cols = has_cyclist_map and {"cyclist_id", "bicycle_id"}.issubset(set(cyclist_map.columns))

        # -----------------------------
        # Select which IDs to draw
        # -----------------------------
        cyclist_ids: set[int] = set()

        if draw_all_bicyclists and cyclist_map_has_cols:
            cyclist_ids |= set(map(int, cyclist_map.get_column("cyclist_id").to_list()))

        if involved_cyclist_ids:
            cyclist_ids |= set(map(int, involved_cyclist_ids))

        # If we ended up with nothing but cyclist_map exists, fall back to cyclist_map cyclists.
        if not cyclist_ids and cyclist_map_has_cols:
            cyclist_ids = set(map(int, cyclist_map.get_column("cyclist_id").to_list()))

        bicycle_ids: set[int] = set()
        if cyclist_ids and cyclist_map_has_cols:
            bicycle_ids = set(
                map(
                    int,
                    cyclist_map.filter(pl.col("cyclist_id").is_in(list(cyclist_ids)))
                    .get_column("bicycle_id")
                    .to_list(),
                )
            )

        object_ids_to_draw: set[int] = cyclist_ids | bicycle_ids  # noqa: F841

        # Map bicycle_id -> cyclist_id for nicer labels
        bike_to_cyclist: dict[int, int] = {}
        if cyclist_map_has_cols:
            for r in cyclist_map.select(["bicycle_id", "cyclist_id"]).iter_rows(named=True):
                try:
                    bike_to_cyclist[int(r["bicycle_id"])] = int(r["cyclist_id"])
                except Exception:
                    continue

        # -----------------------------
        # Build episode intervals (roles)
        # -----------------------------
        leader_col: Optional[str] = None
        if episodes is not None and episodes.height > 0:
            if "leader_id" in episodes.columns:
                leader_col = "leader_id"
            elif "following_id" in episodes.columns:
                leader_col = "following_id"

        intervals: list[tuple[int, int, int, int]] = []  # (start, end, follower_id, leader_id)
        if leader_col is not None:
            needed = {"start_frame", "end_frame", "follower_id", leader_col}
            missing = needed - set(episodes.columns)
            if missing:
                raise ValueError(f"episodes is missing required columns for roles: {sorted(missing)}")

            for r in episodes.select(["start_frame", "end_frame", "follower_id", leader_col]).iter_rows(named=True):
                intervals.append(
                    (
                        int(r["start_frame"]),
                        int(r["end_frame"]),
                        int(r["follower_id"]),
                        int(r[leader_col]),
                    )
                )

        def roles_for_frame(frame_count: int) -> tuple[dict[int, str], dict[int, int]]:
            """roles: cyclist_id -> role; active_pairs: follower_id -> leader_id"""
            roles: dict[int, str] = {}
            active_pairs: dict[int, int] = {}
            for s, e, fid, lid in intervals:
                if s <= frame_count <= e:
                    roles[fid] = "follower"
                    roles[lid] = "leader"
                    active_pairs[fid] = lid
            return roles, active_pairs

        # -----------------------------
        # Pre-index bboxes by frame
        # -----------------------------
        # IMPORTANT: many trackers reuse `unique-id` across classes. If we filter by `unique-id` only,
        # we can accidentally pull a CAR row (yolo-id=2) that shares the same unique-id as a BICYCLE row (yolo-id=1),
        # and then it gets drawn with the wrong label. To avoid that, we filter by BOTH class and id.
        frame_to_rows: dict[int, list[dict]] = {}
        traffic_proxy_class_ids = [int(x) for x in TRAFFIC_CONTROL_PROXY_CLASSES]
        draw_traffic_proxy = bool(ANNOTATE_TRAFFIC_CONTROL_PROXY) and bool(traffic_proxy_class_ids)

        if df.height > 0 and (cyclist_ids or bicycle_ids or draw_traffic_proxy):
            object_filter = (
                ((pl.col("yolo-id") == 0) & (pl.col("unique-id").is_in(list(cyclist_ids)))) |
                ((pl.col("yolo-id") == 1) & (pl.col("unique-id").is_in(list(bicycle_ids))))
            )
            if draw_traffic_proxy:
                object_filter = object_filter | (
                    pl.col("yolo-id").cast(pl.Int64, strict=False).is_in(traffic_proxy_class_ids)
                )

            wanted = (
                df.filter(object_filter)
                .select(["frame-count", "yolo-id", "unique-id", "x-center", "y-center", "width", "height"])
            )
            for row in wanted.iter_rows(named=True):
                fc = int(row["frame-count"])
                frame_to_rows.setdefault(fc, []).append(row)

# Coordinates are YOLO-normalized [0,1] in this project
        coords_normalised = True

        # -----------------------------
        # Open input + output video
        # -----------------------------
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError(f"Invalid video dimensions for: {input_video_path} (w={width}, h={height})")

        if fps_override is not None and float(fps_override) > 0:
            fps = float(fps_override)
        else:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open VideoWriter: {output_video_path}")

        # -----------------------------
        # Main loop
        # -----------------------------
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                csv_frame = frame_idx + int(frame_offset)
                roles, active_pairs = roles_for_frame(csv_frame)

                # Draw all bboxes for this CSV frame
                for row in frame_to_rows.get(csv_frame, []):
                    yolo_id = int(row["yolo-id"])
                    obj_id = int(row["unique-id"])

                    xc = float(row["x-center"])
                    yc = float(row["y-center"])
                    w = float(row["width"])
                    h = float(row["height"])

                    if coords_normalised:
                        xc *= width
                        yc *= height
                        w *= width
                        h *= height

                    x1 = int(round(xc - w / 2.0))
                    y1 = int(round(yc - h / 2.0))
                    x2 = int(round(xc + w / 2.0))
                    y2 = int(round(yc + h / 2.0))

                    # clip to bounds
                    x1 = max(0, min(width - 1, x1))
                    x2 = max(0, min(width - 1, x2))
                    y1 = max(0, min(height - 1, y1))
                    y2 = max(0, min(height - 1, y2))

                    if yolo_id == 0:  # cyclist/person
                        role = roles.get(obj_id, "normal")
                        if role == "follower":
                            color = COLOR_CYCLIST_FOLLOWER
                        elif role == "leader":
                            color = COLOR_CYCLIST_LEADER
                        else:
                            color = COLOR_CYCLIST_NORMAL
                        label = f"{role}:{obj_id}"
                    elif yolo_id == 1:  # bicycle
                        color = COLOR_BICYCLE
                        rider_id = bike_to_cyclist.get(obj_id)
                        label = f"bicycle:{obj_id}" if rider_id is None else f"bicycle:{obj_id} rider:{rider_id}"
                    elif yolo_id in set(int(x) for x in TRAFFIC_CONTROL_PROXY_CLASSES):
                        if yolo_id == YOLO_TRAFFIC_LIGHT_CLASS:
                            color = COLOR_TRAFFIC_LIGHT
                        elif yolo_id == YOLO_STOP_SIGN_CLASS:
                            color = COLOR_STOP_SIGN
                        else:
                            color = COLOR_TRAFFIC_CONTROL_PROXY
                        class_label = TRAFFIC_CONTROL_PROXY_CLASS_LABELS.get(yolo_id, f"traffic-control-{yolo_id}")
                        label = f"{class_label}:{obj_id}"
                    else:
                        # Skip all other object classes (e.g., cars/buses/trucks).
                        continue

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    if draw_labels:
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

                # Optional overlay: active pairs
                if draw_pair_overlay and active_pairs:
                    x0, y0 = 12, 28
                    for i, (fid, lid) in enumerate(sorted(active_pairs.items())):
                        txt = f"Follower {fid} -> Leader {lid}"
                        cv2.putText(
                            frame,
                            txt,
                            (x0, y0 + 18 * i),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                out.write(frame)
                frame_idx += 1

        finally:
            cap.release()
            out.release()

    @staticmethod
    def get_video_resolution_label(video_path: str) -> str:
        """
        Return a resolution label for a local video file using an "exact, truthful" policy.

        Policy
        -----------------
        - Read the frame height (pixels) from the file via OpenCV.
        - If the height matches a known standard, return its label (e.g., "720p", "1080p").
        - If the height is close to a known standard within a small tolerance (to account for
          encoder/container padding such as 1088 instead of 1080), return the nearest
          standard label.
        - Otherwise, return the exact height in the form "<height>p" (e.g., "540p", "768p").

        This approach remains compatible with the updated download selection logic, which
        may select non-standard heights when they are the best available option.

        Parameters
        ----------
        video_path : str
            Path to the video file on disk.

        Returns
        -------
        str
            A resolution label (e.g., "144p", "360p", "720p", "1080p") or "<height>p" for
            non-standard heights.

        Raises
        ------
        FileNotFoundError
            If `video_path` does not exist.
        RuntimeError
            If the video cannot be opened or the frame height cannot be determined.

        Notes
        -----
        - The label is derived from frame height only (not bitrate, codec, aspect ratio, etc.).
        - Some videos may report padded heights (e.g., 544, 736, 1088). These are mapped to
          the nearest standard label only when within the configured tolerance.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

        if height <= 0:
            raise RuntimeError(f"Could not determine frame height for video: {video_path}")

        # Canonical/common heights (add more if we want "named" labels, but non-matches still fall back to "<height>p")
        labels = {
            144: "144p",
            240: "240p",
            360: "360p",
            480: "480p",
            540: "540p",
            576: "576p",
            720: "720p",
            900: "900p",
            1080: "1080p",
            1440: "1440p",
            2160: "2160p",  # 4K UHD
            4320: "4320p",  # 8K UHD
        }

        # Small tolerance to normalize padded encodes (e.g., 1088 -> 1080).
        tolerance_px = 16

        # Exact match
        if height in labels:
            return labels[height]

        # Nearest label within tolerance (padding normalization only)
        closest_h = min(labels.keys(), key=lambda h: abs(height - h))
        if abs(height - closest_h) <= tolerance_px:
            return labels[closest_h]

        # Truthful fallback for truly non-standard heights
        return f"{height}p"

    def trim_video(self, input_path, output_path, start_time, end_time):
        """
        Trims a segment from a video and saves the result to a specified file.
        Parameters: input_path (str): The file path to the original video.
        output_path (str): The destination file path where the trimmed video will be saved.
        start_time (float or str): The start time for the trimmed segment.
        This can be specified in seconds or in a time format recognised by MoviePy.
        nd_time (float or str): The end time for the trimmed segment.
        Similar to start_time, it can be in seconds or another supported time format.
        Returns: None The function performs the following steps:
        1. Loads the original video using MoviePy's VideoFileClip.
        2. Creates a subclip from the original video based on the provided start_time and end_time.
        3. Writes the subclip to the output_path using the H.264 video codec and AAC audio codec.
        4. Closes the video file to free up resources.
        """
        # Load the video and create a subclip using the provided start and end times.
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)  # type: ignore

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the subclip to the specified output file using the 'libx264' codec for video and 'aac' for audio.
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Close the video clip to release any resources used.
        video_clip.close()


if __name__ == "__main__":
    logger.info("Analysis started.")

    # ---------------------------------------------------------------------
    # Load secrets (email + FTP credentials)
    # ---------------------------------------------------------------------
    secret = SimpleNamespace(
        ftp_username=common.get_secrets("ftp_username"),
        ftp_password=common.get_secrets("ftp_password"),
    )

    mapping_path = common.get_configs("mapping")
    df_mapping = pl.read_csv(
        mapping_path,
        schema_overrides={
            "literacy_rate": pl.Float64,
            "gmp": pl.Float64,   # <-- add this
        },
    )

    countries_analyse: list[str] = common.get_configs("countries_analyse")

    if countries_analyse:  # non-empty -> filter
        df_mapping = df_mapping.filter(pl.col("iso3").is_in(countries_analyse))
    # else: empty -> do nothing (keep all rows)

    min_conf = common.get_configs("min_confidence")

    # Precompute fast lookup once (place this immediately before the loops)
    id_to_place: dict[int, tuple[str, str, str]] = {
        int(row_id): (city, state, country)
        for row_id, city, state, country in df_mapping.select(["id", "city", "state", "country"]).iter_rows()
    }

    analysis = Analysis()

    for folder_path in common.get_configs("data"):  # Iterable[str]
        if not os.path.exists(folder_path):
            logger.warning(f"Folder does not exist: {folder_path}.")
            continue

        for file_name in tqdm(os.listdir(folder_path), desc=f"Processing files in {folder_path}"):
            filtered: Optional[str] = analytics_IO.filter_csv_files(
                file=file_name, df_mapping=df_mapping
            )
            if filtered is None:
                continue

            file_str: str = os.fspath(filtered)

            if file_str in MISC_FILES:
                continue

            filename_no_ext = os.path.splitext(file_str)[0]
            logger.debug(f"{filename_no_ext}: fetching values.")

            file_path = os.path.join(folder_path, file_str)

            # Polars read + filter
            df = pl.read_csv(file_path)

            df = (
                df
                # 1) drop invalid IDs early
                .filter(pl.col("unique-id") != -1)
                # 2) enforce join-key dtype compatibility (avoid crashes if some rows look like 15.0)
                .with_columns(
                    pl.col("unique-id").cast(pl.Int64, strict=False)
                )
                # 3) drop rows that couldn't be cast to int
                .filter(pl.col("unique-id").is_not_null())
                # 4) your existing confidence filter
                .filter(pl.col("confidence") >= min_conf)
            )

            # After reading the file, clean up the filename
            base_name = tools.clean_csv_filename(file_str)
            filename_no_ext = os.path.splitext(base_name)[0]

            try:
                video_id, start_index, fps = filename_no_ext.rsplit("_", 2)
            except ValueError:
                logger.warning(f"Unexpected filename format: {filename_no_ext}")
                continue

            video_city_id = geo.find_city_id(df_mapping, video_id, int(start_index))

            place = id_to_place.get(int(video_city_id)) if video_city_id is not None else None
            if place is None:
                logger.warning(f"{file_str}: no mapping row found for id={video_city_id}.")
                continue

            video_city, video_state, video_country = place
            logger.info(f"{file_str}: found values {video_city}, {video_state}, {video_country}.")

            fps_csv = float(fps)  # from filename suffix
            # Require a continuous person-bicycle association of about 2.0 seconds.
            # This avoids short accidental overlaps being treated as cyclists.
            min_continuous_shared_frames = max(1, int(math.ceil(2.0 * fps_csv)))
            shared_run_gap_allow = max(0, int(math.ceil(0.1 * fps_csv)))
            cyclist_map = cf.identify_bicyclists(
                df,
                min_shared_frames=30,
                min_continuous_shared_frames=min_continuous_shared_frames,
                shared_run_gap_allow=shared_run_gap_allow,
                min_vehicle_width_ratio=0.50,
                min_vehicle_width_ratio_frames=0.65,
                score_thresh=0.0,
            )
            states = cf.build_cyclist_states(df, cyclist_map, prefer_vehicle_center=True)
            episodes = cf.detect_following_episodes(
                states,
                params=FollowingParams(
                    speed_min=8e-4,
                    dir_cos_thresh=0.2,
                    rel_speed_thresh=0.003,
                    long_min=0.03,
                    long_max=0.3,
                    lat_max=0.1,
                    min_follow_frames=10,
                    gap_allow=10,
                    min_co_visible_seconds=MIN_CO_VISIBLE_SECONDS,
                    include_pairs_below_min_co_visible=ANNOTATE_BELOW_MIN_CO_VISIBLE_SECONDS,
                    max_mean_time_headway_seconds=MAX_MEAN_TIME_HEADWAY_SECONDS,
                    max_p90_time_headway_seconds=MAX_P90_TIME_HEADWAY_SECONDS,
                ),
                fps=fps_csv,
            )
            logger.info(str(cyclist_map))

            if bool(FILTER_FOLLOWING_BY_TRAFFIC_CONTROL_PROXY):
                raw_episode_count = int(episodes.height)
                episodes = cf.filter_following_episodes_by_traffic_control_proxy(
                    episodes=episodes,
                    df=df,
                    fps=fps_csv,
                    class_ids=TRAFFIC_CONTROL_PROXY_CLASSES,
                    frame_buffer_seconds=CROSSING_EVENT_PROXY_FRAME_BUFFER_SECONDS,
                    min_detections=TRAFFIC_CONTROL_PROXY_MIN_DETECTIONS,
                    min_confidence=TRAFFIC_CONTROL_PROXY_MIN_CONFIDENCE,
                    states=states,
                    max_pair_distance=CROSSING_EVENT_PROXY_MAX_PAIR_DISTANCE,
                    same_frame_tolerance_seconds=CROSSING_EVENT_PROXY_SAME_FRAME_TOLERANCE_SECONDS,
                    max_mean_time_headway_seconds=MAX_MEAN_TIME_HEADWAY_SECONDS,
                    max_p90_time_headway_seconds=MAX_P90_TIME_HEADWAY_SECONDS,
                )
                logger.info(
                    f"{file_str}: traffic-control crossing proxy kept {episodes.height}/{raw_episode_count} "
                    f"following episodes using classes={TRAFFIC_CONTROL_PROXY_CLASSES}, "
                    f"buffer={CROSSING_EVENT_PROXY_FRAME_BUFFER_SECONDS}s, "
                    f"max_pair_distance={CROSSING_EVENT_PROXY_MAX_PAIR_DISTANCE}, "
                    f"max_mean_thw={MAX_MEAN_TIME_HEADWAY_SECONDS}s, "
                    f"max_p90_thw={MAX_P90_TIME_HEADWAY_SECONDS}s, "
                    f"min_detections={TRAFFIC_CONTROL_PROXY_MIN_DETECTIONS}."
                )

            logger.info(str(episodes))

            # Human-readable summary of unique pairs
            try:
                pair_summary = cf.summarize_following_pairs(episodes, fps=fps_csv)
                logger.info(str(pair_summary))
            except Exception:
                pass

            # If enabled, download the source video and generate an annotated video for this CSV segment
            logger.info(f"{file_str}: annotation pipeline triggered (DOWNLOAD_AND_ANNOTATE={DOWNLOAD_AND_ANNOTATE},bicyclists={cyclist_map.height})")  # noqa: E501
            should_annotate = bool(DOWNLOAD_AND_ANNOTATE) and (episodes.height > 0)

            # If cropping is enabled and we're NOT writing full segments, only process segments where following exists.

            if CROP_AROUND_FOLLOWING and (not ANNOTATE_WHOLE_SEGMENT):

                should_annotate = should_annotate and (episodes.height > 0)

            if should_annotate:
                local_video_path: Optional[str] = None
                try:
                    involved_cyclists: set[int] = set()
                    if "follower_id" in episodes.columns:
                        involved_cyclists |= set(episodes.get_column("follower_id").to_list())
                    if "leader_id" in episodes.columns:
                        involved_cyclists |= set(episodes.get_column("leader_id").to_list())

                    # Download base video (cached if already present on disk)
                    dl = analysis.download_videos_from_ftp(
                        filename=video_id,
                        base_url=common.get_configs("ftp_server"),
                        out_dir=DOWNLOADED_VIDEOS_DIR,
                        username=getattr(secret, "ftp_username", None),
                        password=getattr(secret, "ftp_password", None),
                        token=getattr(secret, "ftp_token", None),
                    )
                    if dl is None:
                        logger.warning(f"{file_str}: could not download video for video_id='{video_id}'.")
                        continue

                    local_video_path, downloaded_name, resolution, downloaded_fps = dl
                    logger.info(f"{file_str}: downloaded video '{downloaded_name}' ({resolution}, fps={downloaded_fps}).")  # noqa: E501

                    # Compute segment [start_seconds, end_seconds] from CSV filename + frame-count
                    try:
                        start_seconds = float(start_index)
                    except Exception:
                        logger.warning(f"{file_str}: could not parse start time from '{start_index}'. Skipping video.")
                        continue

                    # CSV fps (from filename) is the best alignment hint; fall back to downloaded fps
                    fps_value = 0.0
                    try:
                        fps_value = float(fps)
                    except Exception:
                        fps_value = float(downloaded_fps or 0.0)

                    if fps_value <= 0:
                        fps_value = 25.0

                    min_frame = int(df.select(pl.min("frame-count")).item()) if df.height > 0 else 1
                    max_frame = int(df.select(pl.max("frame-count")).item()) if df.height > 0 else 1
                    frame_count_base = Analysis.infer_frame_count_base(df)
                    # CSV frame-count is usually one-based in these YOLO outputs:
                    # CSV frame 1 corresponds to the first video frame of the mapped segment.
                    # Therefore frame N starts at (N - frame_count_base) / fps, not N / fps.
                    end_seconds = Analysis.csv_frame_end_seconds(
                        float(start_seconds), int(max_frame), float(fps_value), int(frame_count_base)
                    ) if fps_value > 0 else float(start_seconds)
                    logger.debug(
                        f"{file_str}: frame-count base={frame_count_base}, "
                        f"csv frames=[{min_frame}, {max_frame}], "
                        f"segment seconds=[{start_seconds:.3f}, {end_seconds:.3f}]"
                    )

                    # Decide which window(s) to annotate
                    os.makedirs(TRIMMED_CLIPS_DIR, exist_ok=True)
                    os.makedirs(ANNOTATED_VIDEOS_DIR, exist_ok=True)

                    def _sec_tag(val: float) -> str:
                        # File system friendly tag, eg 12.5 -> 12p5
                        s = f"{float(val):g}"
                        return s.replace('.', 'p')

                    # Bucket size for co visible time folders when keeping clips below MIN_CO_VISIBLE_SECONDS.
                    # Default is 1 second buckets: coviz_2_3s, coviz_3_4s, ...
                    CO_VISIBLE_BUCKET_SECONDS: float = 1.0

                    coviz_ge_dir: Optional[str] = None
                    coviz_bucket_cache: dict[tuple[str, str], str] = {}

                    if (
                        ANNOTATE_BELOW_MIN_CO_VISIBLE_SECONDS
                        and MIN_CO_VISIBLE_SECONDS is not None
                        and float(MIN_CO_VISIBLE_SECONDS) > 0
                    ):
                        tag = _sec_tag(float(MIN_CO_VISIBLE_SECONDS))
                        coviz_ge_dir = os.path.join(ANNOTATED_VIDEOS_DIR, f"coviz_ge_{tag}s")
                        os.makedirs(coviz_ge_dir, exist_ok=True)

                    def _bucket_dir(seconds: float) -> str:
                        # seconds is the co visible time within the crop window
                        b = float(CO_VISIBLE_BUCKET_SECONDS) if CO_VISIBLE_BUCKET_SECONDS is not None else 1.0
                        if b <= 0:
                            b = 1.0
                        s = max(0.0, float(seconds))
                        low = math.floor(s / b) * b
                        high = low + b
                        low_tag = _sec_tag(low)
                        high_tag = _sec_tag(high)
                        key = (low_tag, high_tag)
                        if key not in coviz_bucket_cache:
                            d = os.path.join(ANNOTATED_VIDEOS_DIR, f"coviz_{low_tag}_{high_tag}s")
                            os.makedirs(d, exist_ok=True)
                            coviz_bucket_cache[key] = d
                        return coviz_bucket_cache[key]

                    jobs: list[dict] = []

                    def _add_job(*, clip_start_frame: int, clip_end_frame: int,
                                 clip_name: str, annotated_name: str, annotated_dir: Optional[str] = None) -> None:
                        clip_start_frame = int(clip_start_frame)
                        clip_end_frame = int(clip_end_frame)
                        clip_start_s = Analysis.csv_frame_start_seconds(
                            float(start_seconds), clip_start_frame, float(fps_value), int(frame_count_base)
                        )
                        clip_end_s = Analysis.csv_frame_end_seconds(
                            float(start_seconds), clip_end_frame, float(fps_value), int(frame_count_base)
                        )
                        jobs.append({
                            'clip_start_s': float(clip_start_s),
                            'clip_end_s': float(clip_end_s),
                            'clip_start_frame': int(clip_start_frame),
                            'clip_end_frame': int(clip_end_frame),
                            'frame_offset': int(clip_start_frame),
                            'clip_name': str(clip_name),
                            'annotated_name': str(annotated_name),
                            'annotated_dir': str(annotated_dir) if annotated_dir is not None else str(ANNOTATED_VIDEOS_DIR),  # noqa: E501
                        })

                    # Default (full CSV segment) job definition
                    full_annotated_suffix = 'following_annotated' if episodes.height > 0 else 'annotated'
                    full_clip_name = f"{filename_no_ext}.mp4"
                    full_annotated_name = f"{filename_no_ext}_{full_annotated_suffix}.mp4"

                    if CROP_AROUND_FOLLOWING and episodes.height > 0:
                        # Produce a cropped annotated clip per detected follower->leader pair.
                        pre_frames = int(math.ceil(CROP_PRE_SECONDS * fps_value))
                        post_frames = int(math.ceil(CROP_POST_GONE_SECONDS * fps_value))

                        try:
                            pairs = episodes.select(['follower_id', 'leader_id']).unique()
                        except Exception:
                            pairs = pl.DataFrame(schema={'follower_id': pl.Int64, 'leader_id': pl.Int64})

                        for fid, lid in pairs.select(['follower_id', 'leader_id']).iter_rows():
                            fid_i = int(fid)
                            lid_i = int(lid)
                            annotated_dir_for_pair: str = ANNOTATED_VIDEOS_DIR

                            pair_eps = episodes.filter((pl.col('follower_id') == fid_i
                                                        ) & (pl.col('leader_id') == lid_i))
                            if pair_eps.height == 0:
                                continue

                            # Crop window based on FOLLOWING EPISODES (not full visibility), per follower->leader pair.
                            try:
                                pair_start = int(pair_eps.select(pl.min('start_frame')).item())
                                pair_end = int(pair_eps.select(pl.max('end_frame')).item())
                            except Exception:
                                continue

                            crop_start_frame = max(min_frame, int(pair_start) - pre_frames)
                            crop_end_frame = min(max_frame, int(pair_end) + post_frames)

                            # Optional: enforce that BOTH cyclist tracks are visible together for at least
                            # MIN_CO_VISIBLE_SECONDS *within the cropped output*.
                            # This prevents clips where "following" is detected briefly but one track is mostly absent.
                            if MIN_CO_VISIBLE_SECONDS is not None and float(MIN_CO_VISIBLE_SECONDS) > 0 and fps_value > 0:  # noqa: E501
                                min_coviz_frames = int(math.ceil(float(MIN_CO_VISIBLE_SECONDS) * float(fps_value)))
                                try:
                                    coviz = (
                                        states
                                        .filter(
                                            (pl.col('frame-count') >= crop_start_frame) &
                                            (pl.col('frame-count') <= crop_end_frame) &
                                            (pl.col('cyclist_id').is_in([fid_i, lid_i]))
                                        )
                                        .group_by('frame-count')
                                        .agg(pl.col('cyclist_id').n_unique().alias('n'))
                                        .filter(pl.col('n') >= 2)
                                        .height
                                    )
                                except Exception:
                                    coviz = 0
                                coviz_frames_in_crop = int(coviz)
                                coviz_seconds_in_crop = float(coviz_frames_in_crop) / float(fps_value) if fps_value > 0 else 0.0  # noqa: E501
                                meets_min_coviz = coviz_frames_in_crop >= int(min_coviz_frames)

                                if bool(meets_min_coviz):
                                    if coviz_ge_dir is not None:
                                        annotated_dir_for_pair = coviz_ge_dir
                                else:
                                    if not bool(ANNOTATE_BELOW_MIN_CO_VISIBLE_SECONDS):
                                        logger.info(
                                            f"{file_str}: skipping crop for follower {fid_i} -> leader {lid_i}: "
                                            f"co visible frames in crop {coviz_frames_in_crop} < required {min_coviz_frames}"  # noqa: E501
                                        )
                                        continue

                                    annotated_dir_for_pair = _bucket_dir(coviz_seconds_in_crop)
                                    logger.info(
                                        f"{file_str}: keeping crop for follower {fid_i} -> leader {lid_i} even though "
                                        f"co visible frames in crop {coviz_frames_in_crop} < required {min_coviz_frames}."  # noqa: E501
                                        f"Co visible time in crop is {coviz_seconds_in_crop:.2f}s. "
                                        f"Output will be written under {annotated_dir_for_pair}"
                                    )

                            # Ensure at least ~1 second of output (VideoFileClip can error on empty subclips).
                            if crop_end_frame <= crop_start_frame:
                                crop_end_frame = min(max_frame, crop_start_frame + max(1, int(math.ceil(fps_value))))

                            clip_start_s = Analysis.csv_frame_start_seconds(
                                float(start_seconds), int(crop_start_frame), float(fps_value), int(frame_count_base)
                            )
                            clip_end_s = Analysis.csv_frame_end_seconds(
                                float(start_seconds), int(crop_end_frame), float(fps_value), int(frame_count_base)
                            )

                            clip_name = f"{filename_no_ext}_f{fid_i}_l{lid_i}_crop.mp4"
                            annotated_name = f"{filename_no_ext}_f{fid_i}_l{lid_i}_following_crop_annotated.mp4"
                            _add_job(
                                clip_start_frame=crop_start_frame,
                                clip_end_frame=crop_end_frame,
                                clip_name=clip_name,
                                annotated_name=annotated_name,
                                annotated_dir=annotated_dir_for_pair,
                            )
                            logger.info(
                                f"{file_str}: crop window for follower {fid_i} -> leader {lid_i}: "
                                f"frames [{crop_start_frame}, {crop_end_frame}] "
                                f"(t=[{clip_start_s:.2f}s, {clip_end_s:.2f}s])"
                            )

                        if ANNOTATE_WHOLE_SEGMENT or ALSO_WRITE_FULL_SEGMENT_WHEN_CROPPING:
                            _add_job(
                                clip_start_frame=int(min_frame),
                                clip_end_frame=int(max_frame),
                                clip_name=full_clip_name,
                                annotated_name=full_annotated_name,
                            )
                    else:
                        # Default behavior: annotate the full CSV segment.
                        _add_job(
                            clip_start_frame=int(min_frame),
                            clip_end_frame=int(max_frame),
                            clip_name=full_clip_name,
                            annotated_name=full_annotated_name,
                        )

                    # Execute jobs
                    if not jobs:
                        logger.info(f"{file_str}: no valid crop jobs produced; skipping video trim/annotation.")
                    else:
                        for job in jobs:
                            clip_path = os.path.join(TRIMMED_CLIPS_DIR, job['clip_name'])
                            out_dir = job.get('annotated_dir', ANNOTATED_VIDEOS_DIR)
                            os.makedirs(out_dir, exist_ok=True)
                            annotated_path = os.path.join(out_dir, job['annotated_name'])

                            # Keep a plain trimmed source clip only when explicitly requested. Annotation itself
                            # is rendered directly from the source video using CSV frame-count as an ordinal frame
                            # number. This avoids one-frame offsets and drift from nominal FPS values such as 30
                            # versus 30000/1001.
                            if KEEP_TRIMMED_CLIP:
                                analysis.trim_video(local_video_path, clip_path, job['clip_start_s'], job['clip_end_s'])

                            analysis.annotate_following_segment_from_source(
                                input_video_path=local_video_path,
                                output_video_path=annotated_path,
                                df=df,
                                cyclist_map=cyclist_map,
                                episodes=episodes,
                                involved_cyclist_ids=involved_cyclists,
                                source_segment_start_seconds=float(start_seconds),
                                csv_start_frame=int(job['clip_start_frame']),
                                csv_end_frame=int(job['clip_end_frame']),
                                csv_fps=float(fps_value),
                                frame_count_base=int(frame_count_base),
                                draw_all_bicyclists=ANNOTATE_ALL_BICYCLISTS,
                                draw_pair_overlay=ANNOTATE_PAIR_OVERLAY,
                                draw_labels=True,
                            )
                            logger.info(f"{file_str}: annotated video written to {annotated_path}")

                except Exception as e:
                    logger.error(f"{file_str}: download/annotate failed: {e}")
                finally:
                    if bool(DELETE_DOWNLOADED_VIDEO_ON_COMPLETE) and local_video_path:
                        try:
                            downloaded_root = os.path.abspath(DOWNLOADED_VIDEOS_DIR)
                            candidate = os.path.abspath(local_video_path)
                            # Guard against accidentally deleting anything outside the download cache.
                            if os.path.isfile(candidate) and os.path.commonpath([downloaded_root, candidate]) == downloaded_root:
                                os.remove(candidate)
                                logger.info(f"{file_str}: deleted downloaded source video {candidate}")
                            elif os.path.exists(candidate):
                                logger.warning(
                                    f"{file_str}: not deleting downloaded source video outside download directory: {candidate}"
                                )
                        except Exception as cleanup_error:
                            logger.warning(f"{file_str}: could not delete downloaded source video {local_video_path}: {cleanup_error}")
