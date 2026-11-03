from collections import defaultdict, deque
from pathlib import Path
import cv2

from core.detector import YOLOByteTrackDetector
from core.evidence import EvidenceWriter
from core.fall_logic import FallStateMachine
from core.preprocess import Preprocessor
from core.signal_logic import SignalEvaluator
from core.source import FrameSource

class Pipeline:
    """Core video pipeline; UI is kept outside this class."""
    def __init__(self, video_path, config, state):
        self.config = config or {}
        self.source = FrameSource(video_path)
        self.preprocess = Preprocessor()
        self.detector = YOLOByteTrackDetector(
            self.config.get("model", "yolo26n.pt"),
            self.config.get("conf", 0.25),
            self.config.get("iou", 0.5),
            self.config.get("imgsz", 640),
            self.config.get("device", ""),
        )
        self.fall_logic = FallStateMachine(self.config)
        self.signal_logic = SignalEvaluator(self.config)
        self.state = state
        self.evidence = EvidenceWriter(
            self.config.get("evidence_root", "evidence"),
            self.config.get("evidence_pre_seconds", 1.5),
            self.config.get("evidence_post_seconds", 2.0),
        )
        self.history = defaultdict(lambda: deque(maxlen=40))
        self.approach_counter = defaultdict(int)
        self.red_emitted = set()
        self.pending = []

    @staticmethod
    def _bottom(det):
        return tuple(map(float, det.get("bottom_center", det["center"])))

    @staticmethod
    def _dist(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    def _detect_amr_approach(self, detections, frame_index):
        amr_names = {str(v).lower() for v in self.config.get("amr_classes", ["car", "truck", "forklift"])}
        min_conf = float(self.config.get("amr_conf_min", 0.25))
        min_frames = int(self.config.get("amr_approach_min_frames", 3))
        decrease = float(self.config.get("amr_distance_decrease_pixels", 3))
        max_distance = float(self.config.get("amr_max_distance_pixels", 350))

        fallen_ids = [
            tid for tid in self.state.track_ids
            if self.state.get_track_state(tid).get("fallen_onto_ground")
        ]
        for det in detections:
            amr_id = det.get("track_id")
            if amr_id is None or str(det.get("class_name", "")).lower() not in amr_names:
                continue
            if float(det.get("confidence", 0)) < min_conf:
                continue
            amr_id = int(amr_id)
            amr_pt = self._bottom(det)
            for fallen_id in fallen_ids:
                if int(fallen_id) == amr_id:
                    continue
                fallen_hist = self.history.get(int(fallen_id), [])
                if not fallen_hist:
                    continue
                obj_pt = fallen_hist[-1][1]
                distance = self._dist(amr_pt, obj_pt)
                key = (int(fallen_id), amr_id)
                if distance > max_distance:
                    self.approach_counter[key] = 0
                    continue
                amr_hist = self.history.get(amr_id, [])
                if len(amr_hist) < 2:
                    continue
                prior_distance = self._dist(amr_hist[-2][1], obj_pt)
                if prior_distance - distance >= decrease:
                    self.approach_counter[key] += 1
                else:
                    self.approach_counter[key] = 0
                if self.approach_counter[key] >= min_frames and key not in self.red_emitted:
                    self.red_emitted.add(key)
                    return int(fallen_id), amr_id, round(distance, 2)
        return None

    def _queue_evidence(self, event, level, frame_index, frame, pre_frames, post_frames):
        prior = [img.copy() for _, img in self.frame_buffer]
        self.pending.append({
            "event": event, "level": level, "event_frame": frame_index,
            "snapshot": frame.copy(), "frames": prior,
            "remaining": post_frames,
        })

    def _flush_pending(self, frame, fps):
        done = []
        for item in self.pending:
            item["frames"].append(frame.copy())
            item["remaining"] -= 1
            if item["remaining"] <= 0:
                try:
                    path = self.evidence.write_event(
                        item["level"], item["event"], item["snapshot"],
                        item["frames"], fps
                    )
                    item["event"]["evidence_path"] = path
                except Exception as exc:
                    self.state.add_warning(f"Evidence save failed for {item['event'].get('event_id')}: {exc}")
                done.append(item)
        for item in done:
            self.pending.remove(item)

    def process_all(self, output_path=None, on_progress=None):
        fps = max(float(self.source.fps or 24.0), 1.0)
        frame_count = int(self.source.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        writer = None
        frame_index = 0
        pre_frames = int(float(self.config.get("evidence_pre_seconds", 1.5)) * fps)
        post_frames = int(float(self.config.get("evidence_post_seconds", 2.0)) * fps)
        self.frame_buffer = deque(maxlen=max(1, pre_frames + 1))
        try:
            while True:
                ok, frame = self.source.read()
                if not ok:
                    break
                frame = self.preprocess.process(frame)
                detections, annotated = self.detector.detect_and_track(frame)
                self.frame_buffer.append((frame_index, annotated.copy()))

                for det in detections:
                    tid = det.get("track_id")
                    if tid is None:
                        continue
                    tid = int(tid)
                    self.state.add_track_id(tid)
                    self.history[tid].append((frame_index, self._bottom(det)))

                for det in detections:
                    tid = det.get("track_id")
                    if tid is None:
                        continue
                    event = self.fall_logic.update_track(
                        det, frame_index, fps, frame.shape[0], frame.shape[1]
                    )
                    if event:
                        grade = self.signal_logic.evaluate_detection(
                            det["bbox"], frame.shape[1], frame.shape[0]
                        )
                        event.update(grade)
                        # Critical: this flag is independent of zone overlap or signal grade.
                        event["fallen_onto_ground"] = True
                        event["amr_approach"] = False
                        self.state.update_track_state(int(tid), {
                            "fallen_onto_ground": True,
                            "fall_event_id": event["event_id"],
                            "fall_frame": frame_index,
                            "signal_level": grade["signal_level"],
                        })
                        self.state.add_event(event)
                        self._queue_evidence(event, grade["signal_level"], frame_index, annotated, pre_frames, post_frames)

                approach = self._detect_amr_approach(detections, frame_index)
                if approach:
                    fallen_id, amr_id, distance = approach
                    track_state = self.state.get_track_state(fallen_id)
                    red = {
                        "event_id": f"RED-{fallen_id}-{amr_id}-{frame_index}",
                        "event": "AMR_Approach",
                        "Anomaly_Type": "AMR_Approaching_Fallen_Object",
                        "track_id": fallen_id,
                        "amr_id": amr_id,
                        "linked_to": track_state.get("fall_event_id"),
                        "amr_distance_pixels": distance,
                        "frame": frame_index,
                        "time_sec": round(frame_index / fps, 3),
                        "fallen_onto_ground": True,
                        "amr_approach": True,
                        "signal_level": "red",
                        "uncertain": False,
                        "coords_source": "image",
                    }
                    self.state.update_track_state(fallen_id, {
                        "fallen_onto_ground": True,
                        "amr_approach": True,
                        "red_event_id": red["event_id"],
                        "signal_level": "red",
                    })
                    self.state.add_event(red)
                    self._queue_evidence(red, "red", frame_index, annotated, pre_frames, post_frames)

                self._flush_pending(annotated, fps)

                if output_path:
                    if writer is None:
                        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot create annotated video: {output_path}")
                    writer.write(annotated)

                fallen_now = any(s.get("fallen_onto_ground") for s in self.state.track_states.values())
                approach_now = any(s.get("amr_approach") for s in self.state.track_states.values())
                risk_levels = [s.get("signal_level", "low") for s in self.state.track_states.values() if s.get("fallen_onto_ground")]
                level = "red" if approach_now else "high" if "high" in risk_levels else "medium" if "medium" in risk_levels else "low"
                self.state.update_frame(annotated, level, fallen_now, approach_now)

                frame_index += 1
                if on_progress:
                    fraction = frame_index / frame_count if frame_count > 0 else 0.0
                    on_progress(min(1.0, fraction), annotated)

            # Write pending evidence at end-of-video; clip may be shorter if video ends early.
            for item in self.pending:
                try:
                    path = self.evidence.write_event(item["level"], item["event"], item["snapshot"], item["frames"], fps)
                    item["event"]["evidence_path"] = path
                except Exception as exc:
                    self.state.add_warning(f"Evidence save failed: {exc}")
        finally:
            self.source.release()
            if writer is not None:
                writer.release()
        return {"frames_processed": frame_index, "events": len(self.state.events)}
