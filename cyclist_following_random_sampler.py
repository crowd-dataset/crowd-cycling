from __future__ import annotations

"""
Random sampler for cyclist-following cases.

Place this file in the repository root, next to analysis.py, common.py, config,
and the utils/ directory. Run with:

    python3 cyclist_following_random_sampler.py

This script intentionally does not use a command-line parser. It asks for the
number of cases when it starts.
"""

import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import polars as pl

# -----------------------------------------------------------------------------
# Project-root imports
# -----------------------------------------------------------------------------
# Your structure has analysis.py at the repository root, not inside
# utils.analytics. This block makes that root import explicit and prevents the
# wrong import:
#     from utils.analytics import analysis as analysis_module
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402
from custom_logger import CustomLogger  # noqa: E402
from logmod import logs  # noqa: E402
import analysis as analysis_module  # noqa: E402
from utils.analytics.geo import Geo  # noqa: E402
from utils.analytics.io import IO  # noqa: E402
from utils.bicyclist_detect import Algorithm  # noqa: E402
from utils.bicyclist_following import CyclistFollowing, FollowingParams  # noqa: E402

try:  # noqa: E402
    from utils.bicyclist_following import TRAFFIC_CONTROL_PROXY_CLASS_IDS  # noqa: E402
except ImportError:  # compatibility fallback for older cyclist_following.py
    TRAFFIC_CONTROL_PROXY_CLASS_IDS = [9, 11]
from utils.core.tools import Tools  # noqa: E402

Analysis = analysis_module.Analysis


# =============================================================================
# USER SETTINGS
# =============================================================================
# Keep this as None if you want the script to ask you when it starts.
NUMBER_OF_CASES: Optional[int] = None

# Set to an integer, for example 42, when you want the same random order again.
# Keep as None for a fresh random order every run.
RANDOM_SEED: Optional[int] = None

# Output folder inside common.get_configs("videos").
OUTPUT_FOLDER_NAME = "random_following_samples"

# One fixed OpenCV BGR colour for every cyclist-related bounding box.
# This is applied to follower, following/leader, normal bicyclist, and bicycle.
SAME_CYCLIST_COLOUR_BGR = (0, 255, 0)

# No text in the video saying follower, leader, following, or pair arrows.
DRAW_LABELS = False
DRAW_PAIR_OVERLAY = False

# True = same visual style as your main analysis pipeline, but with one fixed
# colour. False = draw only the sampled follower/following pair.
DRAW_ALL_BICYCLISTS_FROM_CONFIG = False

# Maximum detection gap still treated as the same visibility spell for the sampled pair.
# This prevents one reused/very long track ID from making 30 minute clips.
# Example: at 30 fps and 1.0 second, gaps up to 30 missing frames are still joined.
MAX_PAIR_VISIBILITY_GAP_SECONDS = 1.0

# Bicyclist association gate: a person and bicycle must stay matched for at
# least this many near-continuous seconds. This removes short accidental overlaps,
# such as person ID 1947 with bicycle ID 2083 in 1SM-LqOAFOo_0_30.csv.
MIN_BICYCLIST_SHARED_SECONDS = 2.0
BICYCLIST_SHARED_GAP_ALLOW_SECONDS = 0.1

# Rider sanity check: the matched bicycle should not be much narrower than
# the person for most matched frames. This suppresses pedestrians walking in
# front of smaller/background bicycles.
MIN_BICYCLE_TO_PERSON_WIDTH_RATIO = 0.50
MIN_BICYCLE_WIDTH_RATIO_FRAMES = 0.65

# Optional: pause after each saved sample.
PAUSE_AFTER_EACH_SAMPLE = False

# Regression guard for reviewer-confirmed true positives.
# Format: <csv_filename_without_.csv>_f<follower_id>_l<leader_id>
# Optional sampler prefixes such as sample_0011_ are also accepted.
# The sampler checks these cases first; random sampling starts only if every
# required true-positive pair still survives the updated detector and filters.
TRUE_POSITIVE_REGRESSION_CASES = [
    "32LIoZpuvqA_0_30_f5740_l5658",
]

MISC_FILES = {"DS_Store"}


# =============================================================================
# LOGGER
# =============================================================================
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)


# =============================================================================
# SMALL CONFIG HELPERS, SAME STYLE AS analysis.py
# =============================================================================
def _cfg_bool(key: str, default: bool) -> bool:
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
    try:
        return float(common.get_configs(key))
    except Exception:
        return default


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(common.get_configs(key))
    except Exception:
        return default


def _cfg_int_list(key: str, default: list[int]) -> list[int]:
    """Best-effort config lookup for a list of integer class ids.

    Accepts real lists/tuples/sets and simple strings such as:
      "9,11", "[9, 11]", or "9".
    """
    try:
        value = common.get_configs(key)
    except Exception:
        return list(default)

    if value is None:
        return list(default)

    items = []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default)
        text = text.strip("[](){}")
        items = [part.strip().strip("'\"") for part in text.split(",")]
    else:
        items = [value]

    out: list[int] = []
    for item in items:
        if item is None or str(item).strip() == "":
            continue
        try:
            out.append(int(item))
        except Exception:
            continue

    return out if out else list(default)


def _cfg_str_list(key: str, default: list[str]) -> list[str]:
    """Best-effort config lookup for a list of strings.

    Accepts real lists/tuples/sets and simple strings such as:
      "a,b", "[a, b]", or one item per line.
    """
    try:
        value = common.get_configs(key)
    except Exception:
        return list(default)

    if value is None:
        return list(default)

    if isinstance(value, (list, tuple, set)):
        out = [str(v).strip().strip("'\"") for v in value if str(v).strip()]
        return out if out else list(default)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default)
        text = text.strip("[](){}")
        normalised = text.replace("\n", ",").replace(";", ",")
        out = [part.strip().strip("'\"") for part in normalised.split(",") if part.strip()]
        return out if out else list(default)

    text = str(value).strip()
    return [text] if text else list(default)


def _cfg_float_opt(key: str, default: Optional[float] = None) -> Optional[float]:
    try:
        v = common.get_configs(key)
        if v is None:
            return default
        if isinstance(v, str) and v.strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def _get_secret(key: str) -> Optional[str]:
    try:
        value = common.get_secrets(key)
        if value == "":
            return None
        return value
    except Exception:
        return None


def _normalise_data_folders(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except Exception:
        return []


def _ask_number_of_cases() -> int:
    if NUMBER_OF_CASES is not None:
        return max(1, int(NUMBER_OF_CASES))

    while True:
        raw = input("How many cyclist following cases do you want to sample? ").strip()
        try:
            n = int(raw)
            if n > 0:
                return n
        except Exception:
            pass
        print("Please enter a positive integer.")


@dataclass
class SampleContext:
    file_name: str
    file_path: str
    filename_no_ext: str
    video_id: str
    start_index: str
    fps_from_name: float
    city: str
    state: str
    country: str
    df: pl.DataFrame
    cyclist_map: pl.DataFrame
    states: pl.DataFrame
    episodes: pl.DataFrame


@dataclass(frozen=True)
class TruePositiveCase:
    raw_name: str
    filename_no_ext: str
    follower_id: int
    leader_id: int


MANIFEST_FIELDS = [
    "sample_index",
    "source_csv",
    "video_id",
    "city",
    "state",
    "country",
    "traffic_control_proxy",
    "traffic_control_proxy_count",
    "traffic_control_pair_min_distance",
    "follower_id",
    "leader_id",
    "episode_start_frame",
    "episode_end_frame",
    "crop_start_frame",
    "crop_end_frame",
    "fps",
    "output_path",
]


TRUE_POSITIVE_REGRESSION_FIELDS = [
    "case_name",
    "source_csv",
    "follower_id",
    "leader_id",
    "status",
    "reason",
    "detected_pairs",
    "episode_start_frame",
    "episode_end_frame",
]


# =============================================================================
# MAIN SAMPLER
# =============================================================================
class CyclistFollowingRandomSampler:
    def __init__(self) -> None:
        self.tools = Tools()
        self.geo = Geo()
        self.io = IO()
        self.analysis = Analysis()
        self.cf = CyclistFollowing(Algorithm())

        try:
            self.min_conf = float(common.get_configs("min_confidence"))
        except Exception:
            self.min_conf = 0.0

        self.videos_dir = str(common.get_configs("videos"))
        self.downloaded_videos_dir = os.path.join(self.videos_dir, "downloaded_video")
        self.output_dir = os.path.join(self.videos_dir, OUTPUT_FOLDER_NAME)
        os.makedirs(self.output_dir, exist_ok=True)

        self.crop_pre_seconds = _cfg_float("CROP_PRE_SECONDS", 5.0)
        self.crop_post_seconds = _cfg_float("CROP_POST_GONE_SECONDS", 6.0)
        self.max_pair_visibility_gap_seconds = _cfg_float(
            "MAX_PAIR_VISIBILITY_GAP_SECONDS",
            MAX_PAIR_VISIBILITY_GAP_SECONDS,
        )
        self.min_co_visible_seconds = _cfg_float_opt("MIN_CO_VISIBLE_SECONDS", None)
        self.annotate_below_min_coviz = _cfg_bool("ANNOTATE_BELOW_MIN_CO_VISIBLE_SECONDS", False)
        self.delete_downloaded_video = _cfg_bool("DELETE_DOWNLOADED_VIDEO_ON_COMPLETE", False)
        self.draw_all_bicyclists = (
            _cfg_bool("ANNOTATE_ALL_BICYCLISTS", True)
            if DRAW_ALL_BICYCLISTS_FROM_CONFIG
            else False
        )

        # Keep the random sampler aligned with the main following pipeline:
        # when enabled, only sample follower->leader episodes that occur near
        # a traffic light or stop sign detection. The actual filtering method
        # lives in utils.bicyclist_following.CyclistFollowing.
        self.filter_following_by_traffic_proxy = _cfg_bool(
            "FILTER_FOLLOWING_BY_TRAFFIC_CONTROL_PROXY",
            True,
        )
        self.traffic_control_proxy_classes = _cfg_int_list(
            "TRAFFIC_CONTROL_PROXY_CLASSES",
            TRAFFIC_CONTROL_PROXY_CLASS_IDS,
        )
        self.traffic_control_proxy_frame_buffer_seconds = _cfg_float(
            "TRAFFIC_CONTROL_PROXY_FRAME_BUFFER_SECONDS",
            3.0,
        )
        self.traffic_control_proxy_min_detections = max(
            1,
            _cfg_int("TRAFFIC_CONTROL_PROXY_MIN_DETECTIONS", 1),
        )
        self.traffic_control_proxy_min_confidence = _cfg_float_opt(
            "TRAFFIC_CONTROL_PROXY_MIN_CONFIDENCE",
            None,
        )

        # Reviewer-aligned crossing-event proxy settings. These make the sampler
        # stricter than generic cyclist platooning: the traffic-control object
        # must be close in time and, when coordinates are available, spatially
        # close to the sampled pair. Time-headway gates remove cases where the
        # follower crosses much later than the leader.
        self.crossing_event_proxy_frame_buffer_seconds = _cfg_float(
            "CROSSING_EVENT_PROXY_FRAME_BUFFER_SECONDS",
            0.75,
        )
        self.crossing_event_proxy_max_pair_distance = _cfg_float_opt(
            "CROSSING_EVENT_PROXY_MAX_PAIR_DISTANCE",
            0.35,
        )
        self.crossing_event_proxy_same_frame_tolerance_seconds = _cfg_float(
            "CROSSING_EVENT_PROXY_SAME_FRAME_TOLERANCE_SECONDS",
            0.50,
        )
        self.max_mean_time_headway_seconds = _cfg_float_opt(
            "MAX_MEAN_TIME_HEADWAY_SECONDS",
            3.0,
        )
        self.max_p90_time_headway_seconds = _cfg_float_opt(
            "MAX_P90_TIME_HEADWAY_SECONDS",
            5.0,
        )

        # Reviewer-confirmed positives are used as a regression test before new
        # random samples are generated. This prevents stricter filters from
        # accidentally removing the small set of known valid crossing-following
        # examples.
        self.true_positive_regression_cases = self.parse_true_positive_cases(
            _cfg_str_list("TRUE_POSITIVE_REGRESSION_CASES", TRUE_POSITIVE_REGRESSION_CASES)
        )
        self.require_true_positive_regression_pass = _cfg_bool(
            "REQUIRE_TRUE_POSITIVE_REGRESSION_PASS",
            True,
        )
        self.skip_true_positive_cases_during_random_sampling = _cfg_bool(
            "SKIP_TRUE_POSITIVE_CASES_DURING_RANDOM_SAMPLING",
            True,
        )

        self.secret = SimpleNamespace(
            ftp_username=_get_secret("ftp_username"),
            ftp_password=_get_secret("ftp_password"),
            ftp_token=_get_secret("ftp_token"),
        )

        self.force_one_colour_for_all_boxes()

    @staticmethod
    def force_one_colour_for_all_boxes() -> None:
        """Make every cyclist-related box use exactly the same colour."""
        analysis_module.COLOR_CYCLIST_FOLLOWER = SAME_CYCLIST_COLOUR_BGR
        analysis_module.COLOR_CYCLIST_FOLLOWING = SAME_CYCLIST_COLOUR_BGR
        analysis_module.COLOR_CYCLIST_LEADER = SAME_CYCLIST_COLOUR_BGR
        analysis_module.COLOR_CYCLIST_NORMAL = SAME_CYCLIST_COLOUR_BGR
        analysis_module.COLOR_BICYCLE = SAME_CYCLIST_COLOUR_BGR

    def load_mapping(self) -> pl.DataFrame:
        mapping_path = common.get_configs("mapping")
        df_mapping = pl.read_csv(
            mapping_path,
            schema_overrides={
                "literacy_rate": pl.Float64,
                "gmp": pl.Float64,
            },
        )

        countries_analyse = common.get_configs("countries_analyse")
        if countries_analyse:
            df_mapping = df_mapping.filter(pl.col("iso3").is_in(countries_analyse))

        return df_mapping

    @staticmethod
    def build_place_lookup(df_mapping: pl.DataFrame) -> dict[int, tuple[str, str, str]]:
        return {
            int(row_id): (str(city), str(state), str(country))
            for row_id, city, state, country in df_mapping.select(
                ["id", "city", "state", "country"]
            ).iter_rows()
        }

    def read_detection_csv(self, file_path: str) -> pl.DataFrame:
        df = pl.read_csv(file_path)
        return (
            df
            .filter(pl.col("unique-id") != -1)
            .with_columns(pl.col("unique-id").cast(pl.Int64, strict=False))
            .filter(pl.col("unique-id").is_not_null())
            .filter(pl.col("confidence") >= self.min_conf)
        )

    def detect_cases_in_csv(
        self,
        *,
        folder_path: str,
        file_name: str,
        df_mapping: pl.DataFrame,
        id_to_place: dict[int, tuple[str, str, str]],
    ) -> Optional[SampleContext]:
        filtered = self.io.filter_csv_files(file=file_name, df_mapping=df_mapping)
        if filtered is None:
            return None

        file_str = os.fspath(filtered)
        if file_str in MISC_FILES:
            return None

        base_name = self.tools.clean_csv_filename(file_str)
        filename_no_ext = os.path.splitext(base_name)[0]

        try:
            video_id, start_index, fps_text = filename_no_ext.rsplit("_", 2)
            fps_from_name = float(fps_text)
        except Exception:
            logger.warning(f"Unexpected filename format: {filename_no_ext}")
            return None

        video_city_id = self.geo.find_city_id(df_mapping, video_id, int(start_index))
        place = id_to_place.get(int(video_city_id)) if video_city_id is not None else None
        if place is None:
            logger.warning(f"{file_str}: no mapping row found for id={video_city_id}.")
            return None

        file_path = os.path.join(folder_path, file_str)
        df = self.read_detection_csv(file_path)

        # Same cyclist detection logic as analysis.py, with one extra continuity gate:
        # a person-bicycle pair must be continuously associated for about 2.0 seconds.
        min_continuous_shared_frames = max(1, int(math.ceil(MIN_BICYCLIST_SHARED_SECONDS * fps_from_name)))
        shared_run_gap_allow = max(0, int(math.ceil(BICYCLIST_SHARED_GAP_ALLOW_SECONDS * fps_from_name)))
        cyclist_map = self.cf.identify_bicyclists(
            df,
            min_shared_frames=30,
            min_continuous_shared_frames=min_continuous_shared_frames,
            shared_run_gap_allow=shared_run_gap_allow,
            min_vehicle_width_ratio=MIN_BICYCLE_TO_PERSON_WIDTH_RATIO,
            min_vehicle_width_ratio_frames=MIN_BICYCLE_WIDTH_RATIO_FRAMES,
            score_thresh=0.0,
        )
        if cyclist_map.height == 0:
            return None

        states = self.cf.build_cyclist_states(df, cyclist_map, prefer_vehicle_center=True)
        if states.height == 0:
            return None

        # Same following detection thresholds as analysis.py.
        episodes = self.cf.detect_following_episodes(
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
                min_co_visible_seconds=self.min_co_visible_seconds,
                include_pairs_below_min_co_visible=self.annotate_below_min_coviz,
                max_mean_time_headway_seconds=self.max_mean_time_headway_seconds,
                max_p90_time_headway_seconds=self.max_p90_time_headway_seconds,
            ),
            fps=fps_from_name,
        )

        if episodes.height == 0:
            return None

        if bool(self.filter_following_by_traffic_proxy):
            filter_method = getattr(
                self.cf,
                "filter_following_episodes_by_traffic_control_proxy",
                None,
            )
            if filter_method is None:
                raise RuntimeError(
                    "FILTER_FOLLOWING_BY_TRAFFIC_CONTROL_PROXY is enabled, but "
                    "utils.bicyclist_following.CyclistFollowing does not have "
                    "filter_following_episodes_by_traffic_control_proxy(). "
                    "Please update utils/bicyclist_following.py first."
                )

            before_count = int(episodes.height)
            episodes = filter_method(
                episodes=episodes,
                df=df,
                fps=fps_from_name,
                class_ids=self.traffic_control_proxy_classes,
                frame_buffer_seconds=self.crossing_event_proxy_frame_buffer_seconds,
                min_detections=self.traffic_control_proxy_min_detections,
                min_confidence=self.traffic_control_proxy_min_confidence,
                states=states,
                max_pair_distance=self.crossing_event_proxy_max_pair_distance,
                same_frame_tolerance_seconds=self.crossing_event_proxy_same_frame_tolerance_seconds,
                max_mean_time_headway_seconds=self.max_mean_time_headway_seconds,
                max_p90_time_headway_seconds=self.max_p90_time_headway_seconds,
            )

            if episodes.height == 0:
                return None

            logger.info(
                f"{file_str}: kept {episodes.height}/{before_count} following episodes "
                f"with traffic-control proxy classes={self.traffic_control_proxy_classes}, "
                f"buffer={self.crossing_event_proxy_frame_buffer_seconds}s, "
                f"max_pair_distance={self.crossing_event_proxy_max_pair_distance}, "
                f"max_mean_thw={self.max_mean_time_headway_seconds}s, "
                f"max_p90_thw={self.max_p90_time_headway_seconds}s, "
                f"min_detections={self.traffic_control_proxy_min_detections}."
            )

        city, state, country = place
        return SampleContext(
            file_name=file_str,
            file_path=file_path,
            filename_no_ext=filename_no_ext,
            video_id=video_id,
            start_index=start_index,
            fps_from_name=fps_from_name,
            city=city,
            state=state,
            country=country,
            df=df,
            cyclist_map=cyclist_map,
            states=states,
            episodes=episodes,
        )

    @staticmethod
    def parse_true_positive_case(raw_name: str) -> Optional[TruePositiveCase]:
        """Parse a reviewer-confirmed true-positive case name.

        Accepted examples:
          32LIoZpuvqA_0_30_f5740_l5658
          sample_0011_32LIoZpuvqA_0_30_f5740_l5658
        """
        text = str(raw_name).strip()
        if not text:
            return None

        parts = text.split("_")
        if len(parts) >= 3 and parts[0] == "sample" and parts[1].isdigit():
            text = "_".join(parts[2:])

        try:
            filename_part, leader_part = text.rsplit("_l", 1)
            filename_no_ext, follower_part = filename_part.rsplit("_f", 1)
            follower_id = int(follower_part)
            leader_id = int(leader_part)
        except Exception:
            logger.warning(
                f"Could not parse TRUE_POSITIVE_REGRESSION_CASES entry: {raw_name}. "
                "Expected <csv_filename_without_.csv>_f<follower_id>_l<leader_id>."
            )
            return None

        if not filename_no_ext:
            logger.warning(f"Empty CSV stem in true-positive case: {raw_name}.")
            return None

        return TruePositiveCase(
            raw_name=str(raw_name),
            filename_no_ext=str(filename_no_ext),
            follower_id=int(follower_id),
            leader_id=int(leader_id),
        )

    @classmethod
    def parse_true_positive_cases(cls, raw_cases: list[str]) -> list[TruePositiveCase]:
        cases: list[TruePositiveCase] = []
        seen: set[tuple[str, int, int]] = set()

        for raw in raw_cases:
            case = cls.parse_true_positive_case(raw)
            if case is None:
                continue
            key = (case.filename_no_ext, case.follower_id, case.leader_id)
            if key in seen:
                continue
            seen.add(key)
            cases.append(case)

        return cases

    @staticmethod
    def _csv_job_index(csv_jobs: list[tuple[str, str]]) -> dict[str, tuple[str, str]]:
        """Map CSV stem to job, using the cleaned filename convention."""
        out: dict[str, tuple[str, str]] = {}
        tools = Tools()

        for folder_path, file_name in csv_jobs:
            try:
                clean_name = tools.clean_csv_filename(file_name)
                stem = os.path.splitext(clean_name)[0]
            except Exception:
                stem = os.path.splitext(str(file_name))[0]
            out.setdefault(stem, (folder_path, file_name))

        return out

    @staticmethod
    def _episode_pair_exists(episodes: pl.DataFrame, follower_id: int, leader_id: int) -> bool:
        try:
            return (
                episodes
                .filter(
                    (pl.col("follower_id").cast(pl.Int64, strict=False) == int(follower_id))
                    & (pl.col("leader_id").cast(pl.Int64, strict=False) == int(leader_id))
                )
                .height
                > 0
            )
        except Exception:
            return False

    @staticmethod
    def _pair_episode_frame_range(episodes: pl.DataFrame, follower_id: int, leader_id: int) -> tuple[str, str]:
        try:
            pair_eps = episodes.filter(
                (pl.col("follower_id").cast(pl.Int64, strict=False) == int(follower_id))
                & (pl.col("leader_id").cast(pl.Int64, strict=False) == int(leader_id))
            )
            if pair_eps.height == 0:
                return "", ""
            return (
                str(int(pair_eps.select(pl.min("start_frame")).item())),
                str(int(pair_eps.select(pl.max("end_frame")).item())),
            )
        except Exception:
            return "", ""

    @staticmethod
    def _detected_pair_summary(episodes: pl.DataFrame, limit: int = 20) -> str:
        try:
            if episodes is None or episodes.height == 0:
                return ""
            unique_pairs = episodes.select(["follower_id", "leader_id"]).unique()
            pairs = unique_pairs.sort(["follower_id", "leader_id"]).head(limit).iter_rows()
            out = [f"f{int(fid)}->l{int(lid)}" for fid, lid in pairs]
            if unique_pairs.height > limit:
                out.append("...")
            return "; ".join(out)
        except Exception:
            return ""

    def _known_true_positive_pair_keys(self) -> set[tuple[str, int, int]]:
        return {
            (case.filename_no_ext, int(case.follower_id), int(case.leader_id))
            for case in self.true_positive_regression_cases
        }

    def validate_true_positive_regression_cases(
        self,
        *,
        csv_jobs: list[tuple[str, str]],
        df_mapping: pl.DataFrame,
        id_to_place: dict[int, tuple[str, str, str]],
    ) -> bool:
        """Run known true positives through the full updated detector first.

        If a reviewer-confirmed positive is no longer detected, random sampling
        is stopped by default so thresholds can be optimised before collecting
        more candidate clips.
        """
        cases = list(self.true_positive_regression_cases)
        if not cases:
            return True

        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, "true_positive_regression_report.csv")
        job_index = self._csv_job_index(csv_jobs)
        rows: list[dict[str, object]] = []
        all_passed = True

        print(f"Checking {len(cases)} reviewer-confirmed true-positive case(s) before random sampling...")

        for case in cases:
            job = job_index.get(case.filename_no_ext)
            if job is None:
                all_passed = False
                reason = f"CSV not found in configured data folders: {case.filename_no_ext}.csv"
                print(f"✗ {case.raw_name}: {reason}")
                rows.append({
                    "case_name": case.raw_name,
                    "source_csv": f"{case.filename_no_ext}.csv",
                    "follower_id": int(case.follower_id),
                    "leader_id": int(case.leader_id),
                    "status": "FAIL",
                    "reason": reason,
                    "detected_pairs": "",
                    "episode_start_frame": "",
                    "episode_end_frame": "",
                })
                continue

            folder_path, file_name = job
            try:
                context = self.detect_cases_in_csv(
                    folder_path=folder_path,
                    file_name=file_name,
                    df_mapping=df_mapping,
                    id_to_place=id_to_place,
                )
            except Exception as exc:
                context = None
                all_passed = False
                reason = f"Detector raised an error: {exc}"
                print(f"✗ {case.raw_name}: {reason}")
                rows.append({
                    "case_name": case.raw_name,
                    "source_csv": file_name,
                    "follower_id": int(case.follower_id),
                    "leader_id": int(case.leader_id),
                    "status": "FAIL",
                    "reason": reason,
                    "detected_pairs": "",
                    "episode_start_frame": "",
                    "episode_end_frame": "",
                })
                continue

            if context is None:
                all_passed = False
                reason = "CSV produced no following episodes after the updated filters"
                print(f"✗ {case.raw_name}: {reason}")
                rows.append({
                    "case_name": case.raw_name,
                    "source_csv": file_name,
                    "follower_id": int(case.follower_id),
                    "leader_id": int(case.leader_id),
                    "status": "FAIL",
                    "reason": reason,
                    "detected_pairs": "",
                    "episode_start_frame": "",
                    "episode_end_frame": "",
                })
                continue

            detected_pairs = self._detected_pair_summary(context.episodes)
            if not self._episode_pair_exists(context.episodes, case.follower_id, case.leader_id):
                all_passed = False
                reason = "Expected follower->leader pair was not detected after the updated filters"
                print(f"✗ {case.raw_name}: {reason}. Detected pairs: {detected_pairs or 'none'}")
                rows.append({
                    "case_name": case.raw_name,
                    "source_csv": context.file_name,
                    "follower_id": int(case.follower_id),
                    "leader_id": int(case.leader_id),
                    "status": "FAIL",
                    "reason": reason,
                    "detected_pairs": detected_pairs,
                    "episode_start_frame": "",
                    "episode_end_frame": "",
                })
                continue

            start_frame, end_frame = self._pair_episode_frame_range(
                context.episodes, case.follower_id, case.leader_id
            )
            print(
                f"✓ {case.raw_name}: detected f{case.follower_id}->l{case.leader_id} "
                f"in frames {start_frame}-{end_frame}"
            )
            rows.append({
                "case_name": case.raw_name,
                "source_csv": context.file_name,
                "follower_id": int(case.follower_id),
                "leader_id": int(case.leader_id),
                "status": "PASS",
                "reason": "Expected pair detected",
                "detected_pairs": detected_pairs,
                "episode_start_frame": start_frame,
                "episode_end_frame": end_frame,
            })

        self.write_rows(report_path, TRUE_POSITIVE_REGRESSION_FIELDS, rows)
        print(f"True-positive regression report: {report_path}")

        if all_passed:
            print("All true-positive regression checks passed. Starting random sampling.")
        else:
            print("At least one true-positive regression check failed.")

        return bool(all_passed)

    @staticmethod
    def unique_pairs(episodes: pl.DataFrame, rng: random.Random) -> list[tuple[int, int]]:
        try:
            pairs = episodes.select(["follower_id", "leader_id"]).unique()
            out = [(int(fid), int(lid)) for fid, lid in pairs.iter_rows()]
        except Exception:
            out = []
        rng.shuffle(out)
        return out

    @staticmethod
    def _pair_bicycle_ids(cyclist_map: pl.DataFrame, pair_cyclist_ids: list[int]) -> list[int]:
        """Return mapped bicycle track IDs for the sampled follower and leader."""
        try:
            if cyclist_map is None or cyclist_map.height == 0:
                return []
            if not {"cyclist_id", "bicycle_id"}.issubset(set(cyclist_map.columns)):
                return []
            ids = (
                cyclist_map
                .filter(pl.col("cyclist_id").cast(pl.Int64, strict=False).is_in(pair_cyclist_ids))
                .get_column("bicycle_id")
                .to_list()
            )
            return [int(v) for v in ids if v is not None]
        except Exception:
            return []

    @staticmethod
    def _pair_visibility_frames(
        *,
        context: SampleContext,
        pair_cyclist_ids: list[int],
        pair_bicycle_ids: list[int],
    ) -> list[int]:
        """Visible frames for the sampled pair only.

        A pair member is treated as visible when the person track is visible. The
        mapped bicycle track also counts, because the person box can briefly drop
        while the cyclist is still on screen.
        """
        frames: list[int] = []

        try:
            person_visible = (
                (pl.col("yolo-id") == 0)
                & pl.col("unique-id").cast(pl.Int64, strict=False).is_in(pair_cyclist_ids)
            )
            bicycle_visible = (
                (pl.col("yolo-id") == 1)
                & pl.col("unique-id").cast(pl.Int64, strict=False).is_in(pair_bicycle_ids)
            )
            frames = [
                int(v)
                for v in (
                    context.df
                    .filter(person_visible | bicycle_visible)
                    .select("frame-count")
                    .drop_nulls()
                    .unique()
                    .sort("frame-count")
                    .get_column("frame-count")
                    .to_list()
                )
            ]
        except Exception:
            frames = []

        if frames:
            return sorted(set(frames))

        # Fallback to cyclist states. Episodes are built from these states, so this
        # keeps the sampler usable even if the raw CSV lookup fails.
        try:
            return sorted(
                set(
                    int(v)
                    for v in (
                        context.states
                        .filter(pl.col("cyclist_id").cast(pl.Int64, strict=False).is_in(pair_cyclist_ids))
                        .select("frame-count")
                        .drop_nulls()
                        .unique()
                        .sort("frame-count")
                        .get_column("frame-count")
                        .to_list()
                    )
                )
            )
        except Exception:
            return []

    @staticmethod
    def _visibility_spell_around_episode(
        *,
        visible_frames: list[int],
        episode_start: int,
        episode_end: int,
        max_gap_frames: int,
    ) -> Optional[tuple[int, int]]:
        """Return the visibility spell connected to the sampled following episode.

        The old patch used the first and last visible frame across the whole CSV.
        That can create 30 minute clips when a track ID is reused or when one
        cyclist appears elsewhere in the segment. This method instead splits
        visibility into spells. A spell continues while at least one of the two
        sampled cyclists is visible, allowing short detector gaps. The selected
        spell is the one that overlaps the actual following episode.
        """
        if not visible_frames:
            return None

        frames = sorted(set(int(f) for f in visible_frames))
        allowed_gap = max(0, int(max_gap_frames))

        spells: list[tuple[int, int]] = []
        spell_start = frames[0]
        previous = frames[0]

        for frame in frames[1:]:
            # frame - previous == 1 means no missing frame.
            # allowed_gap == 30 means up to 30 missing frames are still one spell.
            if int(frame) - int(previous) <= allowed_gap + 1:
                previous = int(frame)
                continue
            spells.append((int(spell_start), int(previous)))
            spell_start = int(frame)
            previous = int(frame)

        spells.append((int(spell_start), int(previous)))

        overlapping = [
            (s, e)
            for s, e in spells
            if int(s) <= int(episode_end) and int(e) >= int(episode_start)
        ]
        if overlapping:
            return min(s for s, _ in overlapping), max(e for _, e in overlapping)

        # Defensive fallback: choose the nearest visibility spell to the episode.
        def distance_to_episode(spell: tuple[int, int]) -> int:
            s, e = spell
            if e < int(episode_start):
                return int(episode_start) - e
            if s > int(episode_end):
                return s - int(episode_end)
            return 0

        return min(spells, key=distance_to_episode)

    def crop_window_for_pair(
        self,
        *,
        context: SampleContext,
        follower_id: int,
        leader_id: int,
    ) -> Optional[tuple[pl.DataFrame, int, int, int, int]]:
        pair_eps = context.episodes.filter(
            (pl.col("follower_id") == int(follower_id))
            & (pl.col("leader_id") == int(leader_id))
        )
        if pair_eps.height == 0:
            return None

        try:
            pair_start = int(pair_eps.select(pl.min("start_frame")).item())
            pair_end = int(pair_eps.select(pl.max("end_frame")).item())
            min_frame = int(context.df.select(pl.min("frame-count")).item())
            max_frame = int(context.df.select(pl.max("frame-count")).item())
        except Exception:
            return None

        fps = float(context.fps_from_name)
        pre_frames = int(math.ceil(float(self.crop_pre_seconds) * fps))
        post_frames = int(math.ceil(float(self.crop_post_seconds) * fps))
        max_gap_frames = int(math.ceil(float(self.max_pair_visibility_gap_seconds) * fps))

        pair_cyclist_ids = [int(follower_id), int(leader_id)]
        pair_bicycle_ids = self._pair_bicycle_ids(context.cyclist_map, pair_cyclist_ids)
        visible_frames = self._pair_visibility_frames(
            context=context,
            pair_cyclist_ids=pair_cyclist_ids,
            pair_bicycle_ids=pair_bicycle_ids,
        )

        visibility_spell = self._visibility_spell_around_episode(
            visible_frames=visible_frames,
            episode_start=int(pair_start),
            episode_end=int(pair_end),
            max_gap_frames=max_gap_frames,
        )

        if visibility_spell is None:
            # Last-resort fallback to the detected following episode itself.
            visibility_start = int(pair_start)
            visibility_end = int(pair_end)
        else:
            visibility_start, visibility_end = visibility_spell

        # Desired crop rule:
        #   start = 5 seconds before either sampled cyclist first becomes visible
        #           in the visibility spell connected to the following episode
        #   end   = 6 seconds after BOTH sampled cyclists are gone from the screen
        #
        # "Both gone" starts after the later final visible frame of the follower or
        # leader. Therefore visibility_end is the maximum frame where either one of
        # the two is still visible, not the first time one of them disappears.
        crop_start = max(min_frame, int(visibility_start) - pre_frames)
        crop_end = min(max_frame, int(visibility_end) + post_frames)

        if crop_end <= crop_start:
            crop_end = min(max_frame, crop_start + max(1, int(math.ceil(fps))))

        if (
            self.min_co_visible_seconds is not None
            and float(self.min_co_visible_seconds) > 0
            and fps > 0
            and not bool(self.annotate_below_min_coviz)
        ):
            min_coviz_frames = int(math.ceil(float(self.min_co_visible_seconds) * fps))
            try:
                coviz_frames = (
                    context.states
                    .filter(
                        (pl.col("frame-count") >= int(crop_start))
                        & (pl.col("frame-count") <= int(crop_end))
                        & (pl.col("cyclist_id").cast(pl.Int64, strict=False).is_in(pair_cyclist_ids))
                    )
                    .group_by("frame-count")
                    .agg(pl.col("cyclist_id").n_unique().alias("n"))
                    .filter(pl.col("n") >= 2)
                    .height
                )
            except Exception:
                coviz_frames = 0

            if int(coviz_frames) < int(min_coviz_frames):
                logger.info(
                    f"{context.file_name}: skipped pair {follower_id}->{leader_id}; "
                    f"co-visible frames {coviz_frames} < required {min_coviz_frames}."
                )
                return None

        duration_s = (int(crop_end) - int(crop_start) + 1) / max(fps, 1e-9)
        logger.info(
            f"{context.file_name}: sampled pair {follower_id}->{leader_id}; "
            f"episode=[{pair_start}, {pair_end}], "
            f"visibility_spell=[{visibility_start}, {visibility_end}], "
            f"crop=[{crop_start}, {crop_end}], duration={duration_s:.2f}s."
        )

        return pair_eps, int(pair_start), int(pair_end), int(crop_start), int(crop_end)

    def download_source_video(self, video_id: str):
        return self.analysis.download_videos_from_ftp(
            filename=video_id,
            base_url=common.get_configs("ftp_server"),
            out_dir=self.downloaded_videos_dir,
            username=getattr(self.secret, "ftp_username", None),
            password=getattr(self.secret, "ftp_password", None),
            token=getattr(self.secret, "ftp_token", None),
        )

    @staticmethod
    def traffic_proxy_count_for_episodes(episodes: pl.DataFrame) -> int:
        """Return the total traffic-control proxy detections for sampled episodes."""
        try:
            if episodes is None or episodes.height == 0:
                return 0
            if "traffic_control_proxy_count" not in episodes.columns:
                return 0
            return int(
                episodes
                .select(pl.col("traffic_control_proxy_count").fill_null(0).sum())
                .item()
            )
        except Exception:
            return 0

    @staticmethod
    def traffic_proxy_min_distance_for_episodes(episodes: pl.DataFrame) -> Optional[float]:
        """Return the nearest cyclist-pair to traffic-control proxy distance."""
        try:
            if episodes is None or episodes.height == 0:
                return None
            if "traffic_control_pair_min_distance" not in episodes.columns:
                return None
            value = (
                episodes
                .select(pl.col("traffic_control_pair_min_distance").drop_nulls().min())
                .item()
            )
            return None if value is None else float(value)
        except Exception:
            return None

    @staticmethod
    def write_rows(path: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        """Write a small CSV report, replacing any previous copy."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def write_manifest_row(manifest_path: str, row: dict) -> None:
        file_exists = os.path.exists(manifest_path)
        with open(manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in MANIFEST_FIELDS})

    def maybe_delete_downloaded_video(self, local_video_path: Optional[str]) -> None:
        if not bool(self.delete_downloaded_video) or not local_video_path:
            return
        try:
            downloaded_root = os.path.abspath(self.downloaded_videos_dir)
            candidate = os.path.abspath(local_video_path)
            if os.path.isfile(candidate) and os.path.commonpath([downloaded_root, candidate]) == downloaded_root:
                os.remove(candidate)
                logger.info(f"Deleted downloaded source video {candidate}")
        except Exception as e:
            logger.warning(f"Could not delete downloaded source video {local_video_path}: {e}")

    def save_sample(
        self,
        *,
        context: SampleContext,
        pair_eps: pl.DataFrame,
        follower_id: int,
        leader_id: int,
        episode_start: int,
        episode_end: int,
        crop_start: int,
        crop_end: int,
        sample_index: int,
        local_video_path: str,
        manifest_path: str,
    ) -> str:
        frame_count_base = Analysis.infer_frame_count_base(context.df)

        try:
            source_segment_start_seconds = float(context.start_index)
        except Exception:
            source_segment_start_seconds = 0.0

        output_name = (
            f"sample_{sample_index:04d}_"
            f"{context.filename_no_ext}_"
            f"f{int(follower_id)}_l{int(leader_id)}.mp4"
        )
        output_path = os.path.join(self.output_dir, output_name)

        involved = {int(follower_id), int(leader_id)}

        # Make absolutely sure the colour constants are still fixed immediately
        # before drawing. This prevents role-based colour changes even if another
        # imported module touched the constants.
        self.force_one_colour_for_all_boxes()

        self.analysis.annotate_following_segment_from_source(
            input_video_path=local_video_path,
            output_video_path=output_path,
            df=context.df,
            cyclist_map=context.cyclist_map,
            episodes=pair_eps,
            involved_cyclist_ids=involved,
            source_segment_start_seconds=float(source_segment_start_seconds),
            csv_start_frame=int(crop_start),
            csv_end_frame=int(crop_end),
            csv_fps=float(context.fps_from_name),
            frame_count_base=int(frame_count_base),
            draw_all_bicyclists=bool(self.draw_all_bicyclists),
            draw_pair_overlay=bool(DRAW_PAIR_OVERLAY),
            draw_labels=bool(DRAW_LABELS),
        )

        self.write_manifest_row(
            manifest_path,
            {
                "sample_index": int(sample_index),
                "source_csv": context.file_name,
                "video_id": context.video_id,
                "city": context.city,
                "state": context.state,
                "country": context.country,
                "traffic_control_proxy": bool(self.traffic_proxy_count_for_episodes(pair_eps) > 0),
                "traffic_control_proxy_count": int(self.traffic_proxy_count_for_episodes(pair_eps)),
                "traffic_control_pair_min_distance": self.traffic_proxy_min_distance_for_episodes(pair_eps),
                "follower_id": int(follower_id),
                "leader_id": int(leader_id),
                "episode_start_frame": int(episode_start),
                "episode_end_frame": int(episode_end),
                "crop_start_frame": int(crop_start),
                "crop_end_frame": int(crop_end),
                "fps": float(context.fps_from_name),
                "output_path": output_path,
            },
        )

        return output_path

    def run(self, requested_cases: int) -> None:
        df_mapping = self.load_mapping()
        id_to_place = self.build_place_lookup(df_mapping)

        rng = random.Random(RANDOM_SEED)
        folders = _normalise_data_folders(common.get_configs("data"))

        csv_jobs: list[tuple[str, str]] = []
        for folder_path in folders:
            if not os.path.exists(folder_path):
                logger.warning(f"Folder does not exist: {folder_path}.")
                continue
            for file_name in os.listdir(folder_path):
                if file_name in MISC_FILES:
                    continue
                if not str(file_name).lower().endswith(".csv"):
                    continue
                csv_jobs.append((folder_path, file_name))

        regression_ok = self.validate_true_positive_regression_cases(
            csv_jobs=csv_jobs,
            df_mapping=df_mapping,
            id_to_place=id_to_place,
        )
        if not regression_ok and bool(self.require_true_positive_regression_pass):
            print(
                "Stopped before random sampling because a reviewer-confirmed true positive "
                "was not detected. Loosen or optimise the new filters, then rerun."
            )
            return

        rng.shuffle(csv_jobs)

        manifest_path = os.path.join(self.output_dir, "samples_manifest.csv")
        saved_count = 0
        known_true_positive_pair_keys = self._known_true_positive_pair_keys()

        print(f"Looking for {requested_cases} new random cyclist following cases...")
        print(f"Outputs will be written to: {self.output_dir}")

        for folder_path, file_name in csv_jobs:
            if saved_count >= requested_cases:
                break

            context = self.detect_cases_in_csv(
                folder_path=folder_path,
                file_name=file_name,
                df_mapping=df_mapping,
                id_to_place=id_to_place,
            )
            if context is None:
                continue

            pairs = self.unique_pairs(context.episodes, rng)
            if not pairs:
                continue

            dl = self.download_source_video(context.video_id)
            if dl is None:
                logger.warning(f"{context.file_name}: could not download video for {context.video_id}.")
                continue

            local_video_path = None
            try:
                local_video_path, downloaded_name, resolution, downloaded_fps = dl
                logger.info(
                    f"{context.file_name}: using video {downloaded_name} "
                    f"({resolution}, fps={downloaded_fps})."
                )

                for follower_id, leader_id in pairs:
                    if saved_count >= requested_cases:
                        break

                    pair_key = (context.filename_no_ext, int(follower_id), int(leader_id))
                    if (
                        bool(self.skip_true_positive_cases_during_random_sampling)
                        and pair_key in known_true_positive_pair_keys
                    ):
                        logger.info(
                            f"{context.file_name}: skipping known true-positive regression pair "
                            f"{follower_id}->{leader_id} during new random sampling."
                        )
                        continue

                    crop_info = self.crop_window_for_pair(
                        context=context,
                        follower_id=follower_id,
                        leader_id=leader_id,
                    )
                    if crop_info is None:
                        continue

                    pair_eps, episode_start, episode_end, crop_start, crop_end = crop_info
                    sample_index = saved_count + 1

                    output_path = self.save_sample(
                        context=context,
                        pair_eps=pair_eps,
                        follower_id=follower_id,
                        leader_id=leader_id,
                        episode_start=episode_start,
                        episode_end=episode_end,
                        crop_start=crop_start,
                        crop_end=crop_end,
                        sample_index=sample_index,
                        local_video_path=local_video_path,
                        manifest_path=manifest_path,
                    )

                    saved_count += 1
                    print(f"Saved sample {saved_count}/{requested_cases}: {output_path}")

                    if PAUSE_AFTER_EACH_SAMPLE and saved_count < requested_cases:
                        input("Press Enter to continue to the next sample...")

            except Exception as e:
                logger.error(f"{context.file_name}: sample creation failed: {e}")
            finally:
                self.maybe_delete_downloaded_video(local_video_path)

        print(f"Done. Saved {saved_count}/{requested_cases} samples.")
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    n_cases = _ask_number_of_cases()
    sampler = CyclistFollowingRandomSampler()
    sampler.run(n_cases)
