from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

import math
import numpy as np
import polars as pl

# Reuse your class IDs
PERSON_CLASS = 0
BICYCLE_CLASS = 1


@dataclass
class FollowingParams:
    # Motion / geometry
    speed_min: float = 0.8          # pixels per frame (tune) – see auto-speed logic in detect_following_episodes
    motion_lag: int = 5             # multi-frame displacement to stabilize direction/speed (>=1)
    dir_cos_thresh: float = 0.75    # direction alignment between follower and leader
    rel_speed_thresh: float = 0.6   # |vL - vF| / vF

    # Distance gates in units of follower "size" (height)
    long_min: float = 0.8           # leader must be at least 0.8*h ahead
    long_max: float = 10.0          # and at most 10*h ahead
    lat_max: float = 0.9            # lateral offset at most 0.9*h

    # Persistence
    min_follow_frames: int = 10     # minimum frames to qualify as following episode
    gap_allow: int = 1              # allow up to this many missing frames inside an episode

    # Additional robustness gates (optional but very effective on perspective views)
    # Require leader to have started earlier than follower by at least this many frames.
    # If None, a conservative default of 4 * min_follow_frames is used.
    leader_min_lead_frames: Optional[int] = None

    # Perspective sanity: leader is usually farther away -> smaller bbox height.
    # Set to None to disable.
    leader_height_ratio_max: Optional[float] = 0.85

    # Observation gate (optional)
    # Require follower and leader tracks to overlap in time for at least this many seconds.
    # Overlap is measured using frame-count ranges (max(min_f,min_l) .. min(max_f,max_l)).
    # If None or <= 0, include all pairs.
    min_co_visible_seconds: Optional[float] = None
    # If True, do not filter out follower->leader pairs whose total co-visibility is below
    # min_co_visible_seconds. Instead, include them in the output episodes so downstream
    # code can optionally annotate and save them separately.
    include_pairs_below_min_co_visible: bool = False
    eps: float = 1e-9


class CyclistFollowing:
    """
    Pipeline:
      1) identify_bicyclists: person_id -> bicycle_id (using Algorithm.classify_rider_type)
      2) build_cyclist_states: per-frame cyclist state (x,y, speed, direction)
      3) detect_following_episodes: leader/follower episodes with metrics
    """

    def __init__(self, algorithm) -> None:
        self.algorithm = algorithm  # your Algorithm instance/class

    @staticmethod
    def _dedup_per_frame(df: pl.DataFrame) -> pl.DataFrame:
        # Use your dedup logic if present; fall back to simple unique
        if "confidence" not in df.columns:
            return df.unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")

        return (
            df.sort(
                ["yolo-id", "unique-id", "frame-count", "confidence"],
                descending=[False, False, False, True],
            )
            .unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
        )

    def identify_bicyclists(
        self,
        df: pl.DataFrame,
        *,
        min_shared_frames: int = 4,
        min_continuous_shared_frames: int = 60,
        shared_run_gap_allow: int = 2,
        min_vehicle_width_ratio: float = 0.50,
        min_vehicle_width_ratio_frames: float = 0.65,
        score_thresh: float = 0.0,
        person_class: int = PERSON_CLASS,
        bicycle_class: int = BICYCLE_CLASS,
        enforce_unique_bicycle: bool = True,
    ) -> pl.DataFrame:
        """
        Returns mapping table:
          person_id (cyclist_id), bicycle_id, score, shared_frames, longest_shared_run,
          vehicle_width_ratio, vehicle_width_ratio_pass_ratio
        """
        df = self._dedup_per_frame(df)

        person_ids = (
            df.filter(pl.col("yolo-id") == person_class)
              .select("unique-id")
              .unique()
              .to_series()
              .to_list()
        )

        rows = []
        for pid in person_ids:
            res = self.algorithm.classify_rider_type(
                df,
                pid,
                min_shared_frames=min_shared_frames,
                min_continuous_shared_frames=min_continuous_shared_frames,
                shared_run_gap_allow=shared_run_gap_allow,
                min_vehicle_width_ratio=min_vehicle_width_ratio,
                min_vehicle_width_ratio_frames=min_vehicle_width_ratio_frames,
                person_class=person_class,
                bicycle_class=bicycle_class,
            )
            if (
                bool(res.get("is_rider"))
                and res.get("rider_type") == "bicycle"
                and res.get("role") == "rider"
                and res.get("vehicle_id") is not None
                and float(res.get("score", 0.0)) >= score_thresh
            ):
                rows.append(
                    {
                        "cyclist_id": int(pid),
                        "bicycle_id": int(res["vehicle_id"]),
                        "score": float(res["score"]),
                        "shared_frames": int(res.get("shared_frames", 0)),
                        "longest_shared_run": int(res.get("longest_shared_run", 0)),
                        "vehicle_width_ratio": float(res.get("vehicle_width_ratio", 0.0)),
                        "vehicle_width_ratio_pass_ratio": float(res.get("vehicle_width_ratio_pass_ratio", 0.0)),
                    }
                )

        if not rows:
            return pl.DataFrame(
                {
                    "cyclist_id": pl.Series([], dtype=pl.Int64),
                    "bicycle_id": pl.Series([], dtype=pl.Int64),
                    "score": pl.Series([], dtype=pl.Float64),
                    "shared_frames": pl.Series([], dtype=pl.Int64),
                    "longest_shared_run": pl.Series([], dtype=pl.Int64),
                    "vehicle_width_ratio": pl.Series([], dtype=pl.Float64),
                    "vehicle_width_ratio_pass_ratio": pl.Series([], dtype=pl.Float64),
                }
            )

        out = pl.DataFrame(rows)

        # Often multiple person tracks latch to the same bicycle track (track fragmentation / ID switches).
        # Keeping a 1-to-1 mapping reduces duplicate cyclists and downstream false following pairs.
        if enforce_unique_bicycle and out.height > 0:
            out = (
                out.sort(["bicycle_id", "longest_shared_run", "shared_frames", "vehicle_width_ratio_pass_ratio", "score"], descending=[False, True, True, True, True])
                   .unique(subset=["bicycle_id"], keep="first")
            )

        return out

    @staticmethod
    def build_cyclist_states(
        df: pl.DataFrame,
        cyclist_map: pl.DataFrame,
        *,
        prefer_vehicle_center: bool = True,
        person_class: int = PERSON_CLASS,
        bicycle_class: int = BICYCLE_CLASS,
        motion_lag: int = FollowingParams().motion_lag,
        eps: float = 1e-9,
    ) -> pl.DataFrame:
        """
        Produces per-frame cyclist states with velocity and unit direction.

        Output columns:
          frame-count, cyclist_id, bicycle_id, x, y, w, h, speed, dirx, diry
        """
        df = CyclistFollowing._dedup_per_frame(df)

        if cyclist_map.height == 0:
            return pl.DataFrame()

        # Person detections for cyclist_ids
        p = (
            df.filter(pl.col("yolo-id") == person_class)
              .join(
                  cyclist_map.select(["cyclist_id"]),
                  left_on="unique-id",
                  right_on="cyclist_id",
                  how="inner",
              ).select(
                  [
                      pl.col("frame-count"),
                      pl.col("unique-id").alias("cyclist_id"),
                      pl.col("x-center").alias("px"),
                      pl.col("y-center").alias("py"),
                      pl.col("width").alias("pw"),
                      pl.col("height").alias("ph"),
                  ]
              )
        )

        # Bicycle detections for bicycle_ids
        b = (
            df.filter(pl.col("yolo-id") == bicycle_class)
              .join(
                  cyclist_map.select(["cyclist_id", "bicycle_id"]),
                  left_on="unique-id",
                  right_on="bicycle_id",
                  how="inner",
              ).select(
                  [
                      pl.col("frame-count"),
                      pl.col("cyclist_id"),
                      pl.col("unique-id").alias("bicycle_id"),  # detected bicycle id (may be missing some frames)
                      pl.col("x-center").alias("bx"),
                      pl.col("y-center").alias("by"),
                      pl.col("width").alias("bw"),
                      pl.col("height").alias("bh"),
                  ]
              )
        )

        # Join per frame. Keep both detected bicycle_id and mapped bicycle_id_right for fallback.
        j = (
            p.join(b, on=["frame-count", "cyclist_id"], how="left")
             .join(cyclist_map.select(["cyclist_id", "bicycle_id"]), on="cyclist_id", how="left")
        )

        if prefer_vehicle_center:
            # Use bicycle center when present, else person center
            x = pl.when(pl.col("bx").is_not_null()).then(pl.col("bx")).otherwise(pl.col("px")).alias("x")
            y = pl.when(pl.col("by").is_not_null()).then(pl.col("by")).otherwise(pl.col("py")).alias("y")

            # IMPORTANT CHANGE (Option 2):
            # Always normalize using PERSON size (pw/ph). If person size ever missing, fall back to bicycle size.
            w = pl.coalesce([pl.col("pw"), pl.col("bw")]).alias("w")
            h = pl.coalesce([pl.col("ph"), pl.col("bh")]).alias("h")
        else:
            x = pl.col("px").alias("x")
            y = pl.col("py").alias("y")
            w = pl.coalesce([pl.col("pw"), pl.col("bw")]).alias("w")
            h = pl.coalesce([pl.col("ph"), pl.col("bh")]).alias("h")

        # Stabilize direction using a multi-frame displacement.
        # NOTE: speed is reported as "per-frame" distance (disp / lag).
        lag = max(int(motion_lag), 1)

        states = (
            j.select(
                [
                    pl.col("frame-count"),
                    pl.col("cyclist_id"),

                    # Prefer detected bicycle_id when present; otherwise use mapped id from cyclist_map join.
                    pl.coalesce([pl.col("bicycle_id"), pl.col("bicycle_id_right")]).alias("bicycle_id"),

                    x, y, w, h,
                ]
            )
            .sort(["cyclist_id", "frame-count"])
            .with_columns(
                [
                    (pl.col("x") - pl.col("x").shift(lag)).over("cyclist_id").alias("dx"),
                    (pl.col("y") - pl.col("y").shift(lag)).over("cyclist_id").alias("dy"),
                ]
            )
            .with_columns(
                [
                    ((pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt()).alias("disp"),
                ]
            )
            .with_columns(
                [
                    (pl.col("disp") / float(lag)).alias("speed"),
                ]
            )
            .with_columns(
                [
                    (pl.col("dx") / (pl.col("disp") + eps)).alias("dirx"),
                    (pl.col("dy") / (pl.col("disp") + eps)).alias("diry"),
                ]
            )
            .drop(["dx", "dy", "disp"])
        )

        return states

    @staticmethod
    def detect_following_episodes(
        states: pl.DataFrame,
        *,
        params: FollowingParams = FollowingParams(),
        fps: Optional[float] = None,
    ) -> pl.DataFrame:
        """
        Returns following episodes with leader/follower labels.

        Output columns (episodes):
          follower_id, leader_id, start_frame, end_frame, n_frames,
          mean_long, mean_lat, mean_dist, mean_dir_cos, mean_rel_speed,
          mean_time_headway_frames, mean_time_headway_s (if fps provided)

        Notes on parameter interpretation (auto-mode):
          - If (long_max <= 1.0 and lat_max <= 1.0): interpret long_min/long_max/lat_max
            as ABSOLUTE distances in the same coordinate system as x,y (e.g., normalized [0,1]).
            Otherwise, interpret them in units of follower height (original behavior).
          - If rel_speed_thresh < 0.05: interpret it as ABSOLUTE |vL - vF| (same units as speed).
            Otherwise, interpret it as RELATIVE |vL - vF| / max(vL, vF).
        """

        def _empty_episodes_frame() -> pl.DataFrame:
            schema = {
                "follower_id": pl.Int64,
                "leader_id": pl.Int64,
                "start_frame": pl.Int64,
                "end_frame": pl.Int64,
                "n_frames": pl.Int64,
                "mean_long": pl.Float64,
                "mean_lat": pl.Float64,
                "mean_dist": pl.Float64,
                "mean_dir_cos": pl.Float64,
                "mean_rel_speed": pl.Float64,
                "mean_time_headway_frames": pl.Float64,
            }
            if fps is not None and fps > 0:
                schema["mean_time_headway_s"] = pl.Float64
            if params.min_co_visible_seconds is not None and float(params.min_co_visible_seconds) > 0:
                schema["co_visible_frames_total"] = pl.Int64
                schema["meets_min_co_visible"] = pl.Boolean
                if fps is not None and fps > 0:
                    schema["co_visible_seconds_total"] = pl.Float64
            return pl.DataFrame(schema=schema)

        if states.height == 0:
            return _empty_episodes_frame()

        # Track start frame per cyclist (use full states, not speed-filtered states)
        start_map = {
            int(cid): int(sf)
            for cid, sf in states.group_by("cyclist_id").agg(pl.col("frame-count").min()).rows()
        }

        # Optional co-visibility info and gate: follower->leader pairs whose track time overlap
        # is at least N seconds (converted to frames using fps), based on frame-count ranges.
        # When params.include_pairs_below_min_co_visible is True, pairs below the threshold are
        # still considered, but the overlap duration is attached to the output for downstream
        # filtering and annotation.
        min_coviz_frames: Optional[int] = None
        allowed_leaders_by_follower: Optional[dict[int, np.ndarray]] = None
        coviz_table: Optional[pl.DataFrame] = None
        if params.min_co_visible_seconds is not None and float(params.min_co_visible_seconds) > 0:
            if fps is None or fps <= 0:
                raise ValueError(
                    "min_co_visible_seconds is set but fps was not provided. "
                    "Pass fps to detect_following_episodes so seconds can be converted to frames."
                )
            # seconds -> frames (ceil ensures at least N seconds)
            min_coviz_frames = int(math.ceil(float(params.min_co_visible_seconds) * float(fps)))

            # compute min/max frame-count per cyclist
            rng_rows = (
                states.group_by("cyclist_id")
                .agg(
                    pl.col("frame-count").min().alias("min_f"),
                    pl.col("frame-count").max().alias("max_f"),
                )
                .rows()
            )
            rng = {int(cid): (int(mn), int(mx)) for cid, mn, mx in rng_rows}
            cyclist_ids = sorted(rng.keys())

            coviz_rows: list[dict] = []
            if not bool(params.include_pairs_below_min_co_visible):
                allowed_leaders_by_follower = {}

            for fid in cyclist_ids:
                fmn, fmx = rng[fid]
                allowed: list[int] = []
                for lid in cyclist_ids:
                    if lid == fid:
                        continue
                    lmn, lmx = rng[lid]
                    ov_start = max(fmn, lmn)
                    ov_end = min(fmx, lmx)
                    ov_len = (ov_end - ov_start + 1) if ov_end >= ov_start else 0
                    coviz_rows.append(
                        {
                            "follower_id": int(fid),
                            "leader_id": int(lid),
                            "co_visible_frames_total": int(ov_len),
                        }
                    )
                    if allowed_leaders_by_follower is not None and int(ov_len) >= int(min_coviz_frames):
                        allowed.append(int(lid))
                if allowed_leaders_by_follower is not None:
                    allowed_leaders_by_follower[int(fid)] = np.asarray(allowed, dtype=np.int64)

            if coviz_rows:
                coviz_table = pl.DataFrame(coviz_rows)
        # If the user did not specify a lead-in, use a conservative default.
        leader_min_lead_frames = (
            int(params.leader_min_lead_frames)
            if params.leader_min_lead_frames is not None
            else int(4 * params.min_follow_frames)
        )
        leader_min_lead_frames = max(int(leader_min_lead_frames), 0)

        # Auto speed_min for normalized coordinates (typical YOLO exports: x/y in [0,1]).
        speed_min = float(params.speed_min)
        try:
            x_max = float(states.select(pl.col("x").max()).to_series()[0])
            y_max = float(states.select(pl.col("y").max()).to_series()[0])
            if speed_min > 0.05 and (x_max <= 2.0) and (y_max <= 2.0):
                # Estimate a reasonable speed threshold from the data.
                # Use 20% of the median non-null speed, floored to a tiny epsilon.
                med_speed = float(
                    states.filter(pl.col("speed").is_not_null())
                    .select(pl.col("speed").median())
                    .to_series()[0]
                )
                speed_min = max(0.0005, 0.2 * med_speed)
        except Exception:
            pass

        # Keep only frames where direction is meaningful (speed >= speed_min)
        s = states.filter(pl.col("speed") >= speed_min)
        if s.height == 0:
            return _empty_episodes_frame()

        frames = (
            s.select("frame-count")
             .unique()
             .sort("frame-count")
             .to_series()
             .to_list()
        )

        # Auto-interpret distance thresholds:
        # - "absolute mode" is typical when x,y are normalized to [0,1] and thresholds are like 0.03, 0.3, 0.1
        use_abs_dist = (params.long_max <= 1.0) and (params.lat_max <= 1.0)

        # Auto-interpret speed threshold:
        # - small values (e.g., 0.003) are much more plausible as absolute speed deltas than relative fractions
        use_abs_speed = (params.rel_speed_thresh < 0.05)

        assign_rows: List[dict] = []

        for f in frames:
            sf = s.filter(pl.col("frame-count") == f)
            if sf.height < 2:
                continue

            follower_ids = sf.get_column("cyclist_id").to_numpy()
            x = sf.get_column("x").to_numpy()
            y = sf.get_column("y").to_numpy()
            h = sf.get_column("h").to_numpy()
            sp = sf.get_column("speed").to_numpy()
            dirx = sf.get_column("dirx").to_numpy()
            diry = sf.get_column("diry").to_numpy()

            for i in range(sf.height):
                di0 = float(dirx[i])
                di1 = float(diry[i])
                if not np.isfinite(di0) or not np.isfinite(di1):
                    continue

                rx = x - x[i]
                ry = y - y[i]
                dist = np.sqrt(rx * rx + ry * ry)

                longi = rx * di0 + ry * di1
                lati = np.abs(rx * di1 - ry * di0)  # |cross(rel, dir)| in 2D

                cos_dir = dirx * di0 + diry * di1

                # Speed difference gate (auto-mode)
                abs_sp_diff = np.abs(sp - sp[i])
                if use_abs_speed:
                    speed_metric = abs_sp_diff
                    speed_ok = abs_sp_diff <= params.rel_speed_thresh
                else:
                    denom = np.maximum.reduce([sp, np.full_like(sp, sp[i]), np.full_like(sp, params.eps)])
                    speed_metric = abs_sp_diff / denom
                    speed_ok = speed_metric <= params.rel_speed_thresh

                # Distance gates (auto-mode)
                if use_abs_dist:
                    dist_ok = (
                        (longi >= params.long_min) &
                        (longi <= params.long_max) &
                        (lati <= params.lat_max)
                    )
                    size_i = 1.0  # only for bookkeeping; not used in gating
                else:
                    size_i = max(float(h[i]), params.eps)
                    dist_ok = (
                        (longi >= params.long_min * size_i) &
                        (longi <= params.long_max * size_i) &
                        (lati <= params.lat_max * size_i)
                    )

                cand = (
                    (follower_ids != follower_ids[i]) &
                    (cos_dir >= params.dir_cos_thresh) &
                    dist_ok &
                    speed_ok
                )

                # Optional co-visibility gate
                if allowed_leaders_by_follower is not None:
                    allowed = allowed_leaders_by_follower.get(int(follower_ids[i]))  # type: ignore
                    if allowed is None or allowed.size == 0:  # type: ignore
                        continue
                    cand = cand & np.isin(follower_ids, allowed)

                # Optional: leader must start sufficiently earlier than follower
                if leader_min_lead_frames > 0:
                    f_start = start_map.get(int(follower_ids[i]), 0)
                    lead_ok = np.array(
                        [
                            start_map.get(int(lid), 0) <= (f_start - leader_min_lead_frames)
                            for lid in follower_ids
                        ],
                        dtype=bool,
                    )
                    cand = cand & lead_ok

                # Optional: perspective sanity (leader bbox smaller than follower bbox)
                if params.leader_height_ratio_max is not None:
                    ratio = h / max(float(h[i]), params.eps)
                    cand = cand & (ratio <= float(params.leader_height_ratio_max))

                if not np.any(cand):
                    continue

                # pick nearest leader ahead (smallest longitudinal gap)
                cand_idx = np.where(cand)[0]
                j_best = cand_idx[np.argmin(longi[cand_idx])]

                thw_frames = float(longi[j_best] / max(sp[i], params.eps))

                assign_rows.append(
                    {
                        "frame-count": int(f),
                        "follower_id": int(follower_ids[i]),
                        "leader_id": int(follower_ids[j_best]),
                        "long_gap": float(longi[j_best]),
                        "lat_gap": float(lati[j_best]),
                        "dist": float(dist[j_best]),
                        "dir_cos": float(cos_dir[j_best]),
                        # keep column name for compatibility; metric depends on auto-mode
                        "rel_speed": float(speed_metric[j_best]),
                        "thw_frames": thw_frames,
                    }
                )

        if not assign_rows:
            return _empty_episodes_frame()

        assigns = pl.DataFrame(assign_rows).sort(["follower_id", "frame-count"])

        episodes: List[dict] = []
        for follower_key, g in assigns.group_by("follower_id", maintain_order=True):
            follower_id = follower_key[0] if isinstance(follower_key, tuple) else follower_key

            g = g.sort("frame-count")
            frames_g = g.get_column("frame-count").to_numpy()
            leaders_g = g.get_column("leader_id").to_numpy()

            long_g = g.get_column("long_gap").to_numpy()
            lat_g = g.get_column("lat_gap").to_numpy()
            dist_g = g.get_column("dist").to_numpy()
            cos_g = g.get_column("dir_cos").to_numpy()
            relsp_g = g.get_column("rel_speed").to_numpy()
            thw_g = g.get_column("thw_frames").to_numpy()

            start = 0
            for k in range(1, len(frames_g) + 1):
                end_of_run = False
                if k == len(frames_g):
                    end_of_run = True
                else:
                    gap = frames_g[k] - frames_g[k - 1]
                    if (leaders_g[k] != leaders_g[k - 1]) or (gap > (params.gap_allow + 1)):
                        end_of_run = True

                if end_of_run:
                    seg_slice = slice(start, k)
                    n = k - start
                    if n >= params.min_follow_frames:
                        leader = int(leaders_g[start])
                        seg_frames = frames_g[seg_slice]
                        episodes.append(
                            {
                                "follower_id": int(follower_id),
                                "leader_id": leader,
                                "start_frame": int(seg_frames[0]),
                                "end_frame": int(seg_frames[-1]),
                                "n_frames": int(n),
                                "mean_long": float(np.mean(long_g[seg_slice])),
                                "mean_lat": float(np.mean(lat_g[seg_slice])),
                                "mean_dist": float(np.mean(dist_g[seg_slice])),
                                "mean_dir_cos": float(np.mean(cos_g[seg_slice])),
                                "mean_rel_speed": float(np.mean(relsp_g[seg_slice])),
                                "mean_time_headway_frames": float(np.mean(thw_g[seg_slice])),
                            }
                        )
                    start = k

        if not episodes:
            return _empty_episodes_frame()

        ep = pl.DataFrame(episodes).sort(["start_frame", "follower_id", "leader_id"])

        # Convenience label for quick inspection
        ep = ep.with_columns(
            (pl.col("follower_id").cast(pl.Utf8) + pl.lit("->") + pl.col("leader_id").cast(pl.Utf8)).alias("pair")
        )

        if coviz_table is not None:
            ep = (
                ep.join(coviz_table, on=["follower_id", "leader_id"], how="left")
                .with_columns(
                    pl.col("co_visible_frames_total").fill_null(0).cast(pl.Int64)
                )
            )
            if min_coviz_frames is not None:
                ep = ep.with_columns(
                    (pl.col("co_visible_frames_total") >= int(min_coviz_frames)).alias("meets_min_co_visible")
                )
            if fps is not None and fps > 0:
                ep = ep.with_columns(
                    (pl.col("co_visible_frames_total") / float(fps)).alias("co_visible_seconds_total")
                )
        if fps is not None and fps > 0:
            ep = ep.with_columns(
                (pl.col("mean_time_headway_frames") / float(fps)).alias("mean_time_headway_s")
            )

        return ep

    @staticmethod
    def summarize_following_pairs(
        episodes: pl.DataFrame,
        *,
        fps: Optional[float] = None,
    ) -> pl.DataFrame:
        """
        Summarize detected following relationships as unique (follower_id -> leader_id) pairs.

        Returns a table with:
          follower_id, leader_id, pair, n_episodes, total_frames,
          first_start_frame, last_end_frame, total_seconds (if fps provided),
          w_mean_long, w_mean_lat, w_mean_dist, w_mean_dir_cos, w_mean_rel_speed, w_mean_time_headway_frames,
          w_mean_time_headway_s (if fps provided)

        Weighted means are weighted by episode length (n_frames).
        """
        if episodes is None or episodes.height == 0:
            out = pl.DataFrame(
                {
                    "follower_id": pl.Series([], dtype=pl.Int64),
                    "leader_id": pl.Series([], dtype=pl.Int64),
                    "pair": pl.Series([], dtype=pl.Utf8),
                    "n_episodes": pl.Series([], dtype=pl.Int64),
                    "total_frames": pl.Series([], dtype=pl.Int64),
                    "first_start_frame": pl.Series([], dtype=pl.Int64),
                    "last_end_frame": pl.Series([], dtype=pl.Int64),
                }
            )
            if fps is not None and fps > 0:
                out = out.with_columns(pl.Series([], dtype=pl.Float64).alias("total_seconds"))
            return out

        # Ensure expected columns exist
        required = {"follower_id", "leader_id", "start_frame", "end_frame", "n_frames"}
        missing = required - set(episodes.columns)
        if missing:
            raise ValueError(f"episodes is missing required columns: {sorted(missing)}")

        # Weighted means (by n_frames) for useful diagnostics
        def wmean(col: str) -> pl.Expr:
            if col not in episodes.columns:
                return pl.lit(None).cast(pl.Float64).alias(f"w_{col}")
            return (pl.sum(pl.col(col) * pl.col("n_frames")) / pl.sum("n_frames")).alias(f"w_{col}")  # type: ignore

        agg = (
            episodes.group_by(["follower_id", "leader_id"])
            .agg(
                [
                    pl.count().alias("n_episodes"),
                    pl.sum("n_frames").alias("total_frames"),
                    pl.min("start_frame").alias("first_start_frame"),
                    pl.max("end_frame").alias("last_end_frame"),
                    wmean("mean_long"),
                    wmean("mean_lat"),
                    wmean("mean_dist"),
                    wmean("mean_dir_cos"),
                    wmean("mean_rel_speed"),
                    wmean("mean_time_headway_frames"),
                ]
            )
            .with_columns(
                (pl.col("follower_id").cast(pl.Utf8) + pl.lit("->") + pl.col("leader_id").cast(pl.Utf8)).alias("pair")
            )
            .sort("total_frames", descending=True)
        )

        if fps is not None and fps > 0:
            agg = agg.with_columns((pl.col("total_frames") / float(fps)).alias("total_seconds"))
            # If we have weighted THW in frames, also provide seconds.
            if "w_mean_time_headway_frames" in agg.columns:
                agg = agg.with_columns(
                    (pl.col("w_mean_time_headway_frames") / float(fps)).alias("w_mean_time_headway_s")
                )

        return agg
