from typing import Optional
import numpy as np
import polars as pl

# Class IDs (same values)
PERSON_CLASS = 0
BICYCLE_CLASS = 1
CAR_CLASS = 2
MOTORCYCLE_CLASS = 3
BUS_CLASS = 5
TRUCK_CLASS = 7


class Algorithm():
    def __init__(self) -> None:
        pass

    @staticmethod
    def _dedup_per_frame(df: pl.DataFrame) -> pl.DataFrame:
        """Keep the highest-confidence detection for each (yolo-id, unique-id, frame-count)."""
        if "confidence" not in df.columns:
            return df.unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")

        return (
            df.sort(
                ["yolo-id", "unique-id", "frame-count", "confidence"],
                descending=[False, False, False, True],
            )
            .unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
        )

    @staticmethod
    def _longest_frame_run(frames, *, gap_allow: int = 2) -> int:
        """Return the longest near-continuous run of frame numbers.

        gap_allow is the number of missing frames allowed inside one run.
        For example, gap_allow=2 means frames 10 and 13 are still considered
        part of the same run, because only 11 and 12 are missing.
        """
        try:
            values = sorted({int(f) for f in frames})
        except Exception:
            return 0

        if not values:
            return 0

        max_run = 1
        cur_run = 1
        max_gap = max(int(gap_allow), 0) + 1
        prev = values[0]

        for frame in values[1:]:
            if int(frame) - int(prev) <= max_gap:
                cur_run += 1
            else:
                max_run = max(max_run, cur_run)
                cur_run = 1
            prev = frame

        return max(max_run, cur_run)

    @staticmethod
    def classify_rider_type(
        df: pl.DataFrame,
        person_id,
        *,
        avg_height: Optional[float] = None,  # accepted for compatibility; not used for normalized scaling
        min_shared_frames: int = 4,
        min_continuous_shared_frames: int = 60,
        shared_run_gap_allow: int = 2,
        min_vehicle_width_ratio: float = 0.50,
        min_vehicle_width_ratio_frames: float = 0.65,
        dist_rel_thresh: float = 0.8,
        prox_req: float = 0.7,
        alpha_x: float = 1.0,
        beta_y: float = 0.03,
        gamma_y: float = 1.4,
        coloc_req: float = 0.7,
        sim_thresh: float = 0.4,
        sim_req: float = 0.5,
        min_motion_steps: int = 3,
        motion_coloc_min: float = 0.5,
        short_shared_frames: int = 8,
        short_sim_req: float = 0.8,
        short_disp_req: float = 0.12,
        eps: float = 1e-9,
        person_class: int = PERSON_CLASS,
        bicycle_class: int = BICYCLE_CLASS,
        motorcycle_class: int = MOTORCYCLE_CLASS,
        car_class: int = CAR_CLASS,
        bus_class: int = BUS_CLASS,
        truck_class: int = TRUCK_CLASS,
    ) -> dict:
        """
        Returns a dict with keys:
          - is_rider (bool): True if the person is associated with any supported vehicle type.
          - rider_type (str|None): "bicycle"|"motorcycle"|"car"|"bus"|"truck"|None
          - role (str|None): "rider"|"passenger"|None
          - vehicle_id
          - score
          - shared_frames
          - longest_shared_run
          - vehicle_width_ratio, vehicle_width_ratio_pass_ratio
          - prox_ratio, coloc_ratio, sim_ratio
        """
        if avg_height is not None:
            try:
                if float(avg_height) <= 0.0:
                    return {
                        "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                        "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
                    }
            except Exception:
                return {
                    "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                    "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
                }

        df = Algorithm._dedup_per_frame(df)

        p = (
            df.filter((pl.col("yolo-id") == person_class) & (pl.col("unique-id") == person_id))
              .sort("frame-count")
        )
        if p.height == 0:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        p_frames = p.get_column("frame-count").to_numpy()
        if p_frames.size < min_shared_frames:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        first_frame = int(p_frames.min())
        last_frame = int(p_frames.max())

        supported_vehicle_classes = [bicycle_class, motorcycle_class, car_class, bus_class, truck_class]

        vehicles = df.filter(
            (pl.col("frame-count") >= first_frame)
            & (pl.col("frame-count") <= last_frame)
            & (pl.col("yolo-id").is_in(supported_vehicle_classes))
        )
        if vehicles.height == 0:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        vehicle_ids = vehicles.select("unique-id").unique().to_series().to_list()
        p1 = p.unique(subset=["frame-count"], keep="first")

        best = None

        for vid in vehicle_ids:
            v = vehicles.filter(pl.col("unique-id") == vid).sort("frame-count")
            if v.height == 0:
                continue

            v_class = int(v.get_column("yolo-id")[0])
            vtype = (
                "bicycle" if v_class == bicycle_class else
                "motorcycle" if v_class == motorcycle_class else
                "car" if v_class == car_class else
                "bus" if v_class == bus_class else
                "truck" if v_class == truck_class else
                None
            )
            if vtype is None:
                continue

            role = "rider" if v_class in (bicycle_class, motorcycle_class) else "passenger"

            v1 = v.unique(subset=["frame-count"], keep="first")
            j = p1.join(v1, on="frame-count", how="inner", suffix="_v")
            shared = j.height
            if shared < min_shared_frames:
                continue

            longest_shared_run = Algorithm._longest_frame_run(
                j.get_column("frame-count").to_list(),
                gap_allow=shared_run_gap_allow,
            )
            if role == "rider" and longest_shared_run < int(min_continuous_shared_frames):
                continue

            p_xy = j.select(["x-center", "y-center"]).to_numpy()
            v_xy = j.select(["x-center_v", "y-center_v"]).to_numpy()

            p_w = j.get_column("width").to_numpy()
            p_h = j.get_column("height").to_numpy()
            v_w = j.get_column("width_v").to_numpy()
            v_h = j.get_column("height_v").to_numpy()

            if role == "rider":
                vehicle_width_ratio_arr = v_w / np.maximum(p_w, eps)
                vehicle_width_ratio = float(np.median(vehicle_width_ratio_arr))
                vehicle_width_ratio_pass_ratio = float(
                    (vehicle_width_ratio_arr >= float(min_vehicle_width_ratio)).mean()
                )
                if vehicle_width_ratio_pass_ratio < float(min_vehicle_width_ratio_frames):
                    continue
            else:
                vehicle_width_ratio = 0.0
                vehicle_width_ratio_pass_ratio = 0.0

            dist = np.linalg.norm(p_xy - v_xy, axis=1)
            if role == "rider":
                dist_rel = dist / np.maximum(p_h, eps)
            else:
                dist_rel = dist / np.maximum(v_h, eps)

            prox = dist_rel < dist_rel_thresh
            prox_ratio = float(prox.mean())
            if prox_ratio < prox_req:
                continue

            relx = v_xy[:, 0] - p_xy[:, 0]
            rely = v_xy[:, 1] - p_xy[:, 1]

            if role == "rider":
                spatial = (np.abs(relx) < alpha_x * p_w) & (rely > beta_y * p_h) & (rely < gamma_y * p_h)
            else:
                inside = (np.abs(relx) <= 0.5 * v_w) & (np.abs(rely) <= 0.5 * v_h)
                spatial = inside

            coloc = prox & spatial
            coloc_ratio = float(coloc.mean())

            p_mov = np.diff(p_xy, axis=0)
            v_mov = np.diff(v_xy, axis=0)

            sim_ratio = 0.0
            if p_mov.shape[0] > 0:
                na = np.linalg.norm(p_mov, axis=1)
                nb = np.linalg.norm(v_mov, axis=1)
                move_mask = (na > eps) & (nb > eps)

                cos = np.zeros_like(na, dtype=float)
                cos[move_mask] = (p_mov[move_mask] * v_mov[move_mask]).sum(axis=1) / (na[move_mask] * nb[move_mask])

                prox_steps = prox[1:]
                m = min(len(prox_steps), len(cos), len(move_mask))
                prox_steps = prox_steps[:m]
                cos = cos[:m]
                move_mask = move_mask[:m]

                denom_mask = prox_steps & move_mask
                denom = int(denom_mask.sum())
                if denom >= min_motion_steps:
                    sim_ratio = float(((cos > sim_thresh) & denom_mask).sum() / denom)

            if shared < short_shared_frames:
                if shared > 1:
                    p_disp = float(np.linalg.norm(p_xy[-1] - p_xy[0]))
                    p_disp_rel = p_disp / float(np.maximum(np.mean(p_h), eps))
                else:
                    p_disp_rel = 0.0

                if not (sim_ratio >= short_sim_req or p_disp_rel >= short_disp_req):
                    continue

            ok = (coloc_ratio >= coloc_req) or (sim_ratio >= sim_req and coloc_ratio >= motion_coloc_min)
            if not ok:
                continue

            score = 0.7 * coloc_ratio + 0.2 * prox_ratio + 0.1 * float(sim_ratio)
            cand = {
                "is_rider": True,
                "rider_type": vtype,
                "role": role,
                "vehicle_id": vid,
                "score": float(score),
                "shared_frames": int(shared),
                "longest_shared_run": int(longest_shared_run),
                "vehicle_width_ratio": float(vehicle_width_ratio),
                "vehicle_width_ratio_pass_ratio": float(vehicle_width_ratio_pass_ratio),
                "prox_ratio": prox_ratio,
                "coloc_ratio": coloc_ratio,
                "sim_ratio": float(sim_ratio),
            }

            if best is None or cand["score"] > best["score"]:
                best = cand

        if best is None:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        return best

    def is_bicyclist(
        self,
        df: pl.DataFrame,
        person_id,
        *,
        avg_height: Optional[float] = None,
        min_shared_frames: int = 4,
        min_continuous_shared_frames: int = 60,
        shared_run_gap_allow: int = 2,
        min_vehicle_width_ratio: float = 0.50,
        min_vehicle_width_ratio_frames: float = 0.65,
        dist_rel_thresh: float = 0.8,
        prox_req: float = 0.7,
        alpha_x: float = 1.0,
        beta_y: float = 0.03,
        gamma_y: float = 1.4,
        coloc_req: float = 0.7,
        sim_thresh: float = 0.4,
        sim_req: float = 0.5,
        min_motion_steps: int = 3,
        motion_coloc_min: float = 0.5,
        short_shared_frames: int = 8,
        short_sim_req: float = 0.8,
        short_disp_req: float = 0.12,
        eps: float = 1e-9,
        person_class: int = PERSON_CLASS,
        bicycle_class: int = BICYCLE_CLASS,
        motorcycle_class: int = MOTORCYCLE_CLASS,
        car_class: int = CAR_CLASS,
        bus_class: int = BUS_CLASS,
        truck_class: int = TRUCK_CLASS,
    ) -> bool:
        """
        True iff the person is classified as a bicyclist (i.e., best associated vehicle is a bicycle),
        using the exact same parameter defaults/values as the original code.
        """
        res = Algorithm.classify_rider_type(
            df,
            person_id,
            avg_height=avg_height,
            min_shared_frames=min_shared_frames,
            min_continuous_shared_frames=min_continuous_shared_frames,
            shared_run_gap_allow=shared_run_gap_allow,
            min_vehicle_width_ratio=min_vehicle_width_ratio,
            min_vehicle_width_ratio_frames=min_vehicle_width_ratio_frames,
            dist_rel_thresh=dist_rel_thresh,
            prox_req=prox_req,
            alpha_x=alpha_x,
            beta_y=beta_y,
            gamma_y=gamma_y,
            coloc_req=coloc_req,
            sim_thresh=sim_thresh,
            sim_req=sim_req,
            min_motion_steps=min_motion_steps,
            motion_coloc_min=motion_coloc_min,
            short_shared_frames=short_shared_frames,
            short_sim_req=short_sim_req,
            short_disp_req=short_disp_req,
            eps=eps,
            person_class=person_class,
            bicycle_class=bicycle_class,
            motorcycle_class=motorcycle_class,
            car_class=car_class,
            bus_class=bus_class,
            truck_class=truck_class,
        )
        return bool(res.get("is_rider")) and (res.get("rider_type") == "bicycle")
