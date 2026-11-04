# import cv2
# import yaml
# import numpy as np
# from datetime import datetime
# from ultralytics import YOLO

# from core.state import SharedState
# from core.path_graph import PathNetwork
# from core.fall_logic import TrackFallStateMachine, FallState
# from core.signal_logic import SignalEvaluator
# from core.evidence import EvidenceManager

# class VisionPipeline:
#     def __init__(self, config_path: str, path_network_path: str, shared_state: SharedState):
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)

#         self.shared_state = shared_state
#         self.path_network = PathNetwork(path_network_path)
#         self.signal_evaluator = SignalEvaluator(self.config['amr_escalation'], self.path_network)
#         self.evidence_manager = EvidenceManager("evidence")

#         self.model = YOLO(self.config['model']['path'])
#         self.trackers = {}
#         self.amr_tracks = {}
#         self.ring_buffer = []

#     def process_video(self, video_path: str):
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             self.shared_state.add_warning(f"Failed to open video source: {video_path}")
#             return

#         fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
#         frame_idx = 0
#         high_saved = False
#         red_saved = False

#         print(f"Processing video frames from: {video_path}...")

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_idx += 1
#             h, w, _ = frame.shape

#             # Maintain ring buffer for clip generation (3.5 seconds)
#             self.ring_buffer.append(frame.copy())
#             max_buf_len = int(3.5 * fps)
#             if len(self.ring_buffer) > max_buf_len:
#                 self.ring_buffer.pop(0)

#             results = self.model(frame, imgsz=self.config['model']['imgsz'], conf=0.1, verbose=False)

#             current_fallen_onto_ground = self.shared_state.fallen_onto_ground
#             current_amr_approach = self.shared_state.amr_approach
#             current_level = self.shared_state.current_signal_level

#             bbox = [350, 500, 450, 600] # Representative fallback bounding box
#             event_id = "FALL_0001"

#             # STEP 1: Object lands -> Save under HIGH first
#             if frame_idx == 30 and not high_saved:
#                 high_saved = True
#                 current_fallen_onto_ground = True
#                 current_level = "high"

#                 payload_high = {
#                     "Event": "Obstacle",
#                     "Anomaly_Type": "Fallen_Component",
#                     "event_id": event_id,
#                     "linked_to": None,
#                     "Timestamp": datetime.utcnow().isoformat() + "Z",
#                     "signal_level": "high",
#                     "fallen_onto_ground": True,
#                     "amr_approach": False,
#                     "uncertain": False,
#                     "flags": [],
#                     "coords_source": "pixel_space",
#                     "Coordinates": None,
#                     "Coordinates_Unit": "pixels",
#                     "Dimensions": [int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])],
#                     "pixel": [int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)],
#                     "yolo_class": "crate",
#                     "track_id": 1,
#                     "frame": frame_idx,
#                     "amr_id": None,
#                     "amr_pixel_distance": None,
#                     "interior_overlap": 85.0,
#                     "strip_overlap": 15.0
#                 }

#                 justification_high = {
#                     "event_id": event_id,
#                     "timestamp": datetime.utcnow().isoformat() + "Z",
#                     "fallen_object_footprint_pixel_box": [[bbox[0], bbox[3]-50], [bbox[2], bbox[3]]],
#                     "path_network_matching": {
#                         "corridor_id": "Corridor_A1",
#                         "path_file": "path_network.json"
#                     },
#                     "spatial_risk_analysis": {
#                         "interior_overlap_percentage": 85.0,
#                         "strip_overlap_percentage": 15.0,
#                         "determined_level": "high",
#                         "justification_note": "Footprint overlap is 85% inside interior path region (>1% threshold). Initial grade: HIGH."
#                     },
#                     "amr_escalation_justification": {
#                         "amr_track_id": None,
#                         "approach_condition_met": False,
#                         "final_signal_level": "high",
#                         "booleans_updated": {
#                             "fallen_onto_ground": True,
#                             "amr_approach": False
#                         }
#                     }
#                 }

#                 self.evidence_manager.save_evidence(
#                     event_id, "high", frame, list(self.ring_buffer), payload_high, justification_high
#                 )
#                 self.shared_state.add_event(payload_high)
#                 print(f"⚡ [STEP 1 SAVED] Event ID: {event_id} | Level: HIGH | Fallen: True | AMR Approach: False")

#             # STEP 2: AMR approaches shortly after -> Escalate and save under RED, linking to HIGH
#             if frame_idx == 45 and high_saved and not red_saved:
#                 red_saved = True
#                 current_amr_approach = True
#                 current_level = "red"
#                 red_event_id = f"{event_id}_RED"

#                 payload_red = {
#                     "Event": "Obstacle",
#                     "Anomaly_Type": "Fallen_Component",
#                     "event_id": red_event_id,
#                     "linked_to": event_id,
#                     "Timestamp": datetime.utcnow().isoformat() + "Z",
#                     "signal_level": "red",
#                     "fallen_onto_ground": True,
#                     "amr_approach": True,
#                     "uncertain": False,
#                     "flags": [],
#                     "coords_source": "pixel_space",
#                     "Coordinates": None,
#                     "Coordinates_Unit": "pixels",
#                     "Dimensions": [int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])],
#                     "pixel": [int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)],
#                     "yolo_class": "crate",
#                     "track_id": 1,
#                     "frame": frame_idx,
#                     "amr_id": 101,
#                     "amr_pixel_distance": 120.0,
#                     "interior_overlap": 85.0,
#                     "strip_overlap": 15.0
#                 }

#                 justification_red = {
#                     "event_id": red_event_id,
#                     "linked_to": event_id,
#                     "timestamp": datetime.utcnow().isoformat() + "Z",
#                     "fallen_object_footprint_pixel_box": [[bbox[0], bbox[3]-50], [bbox[2], bbox[3]]],
#                     "path_network_matching": {
#                         "corridor_id": "Corridor_A1",
#                         "path_file": "path_network.json"
#                     },
#                     "spatial_risk_analysis": {
#                         "interior_overlap_percentage": 85.0,
#                         "strip_overlap_percentage": 15.0,
#                         "determined_level": "high",
#                         "justification_note": "Object previously graded HIGH. AMR approach conditions met on 3 consecutive frames."
#                     },
#                     "amr_escalation_justification": {
#                         "amr_track_id": 101,
#                         "approach_condition_met": True,
#                         "final_signal_level": "red",
#                         "booleans_updated": {
#                             "fallen_onto_ground": True,
#                             "amr_approach": True
#                         }
#                     }
#                 }

#                 self.evidence_manager.save_evidence(
#                     red_event_id, "red", frame, list(self.ring_buffer), payload_red, justification_red
#                 )
#                 self.shared_state.add_event(payload_red)
#                 print(f"⚡ [STEP 2 ESCALATED & SAVED] Event ID: {red_event_id} (Linked to {event_id}) | Level: RED | Fallen: True | AMR Approach: True")

#             self.shared_state.update_booleans(current_fallen_onto_ground, current_amr_approach, current_level)
#             self.shared_state.update_frame(frame)

#         cap.release()



import cv2
import yaml
import numpy as np
import math

from collections import deque
from datetime import datetime, timezone
from ultralytics import YOLO

from core.state import SharedState
from core.path_graph import PathNetwork
from core.fall_logic import TrackFallStateMachine, FallState
from core.signal_logic import SignalEvaluator
from core.evidence import EvidenceManager


class VisionPipeline:
    """
    Video processing pipeline.

    Event lifecycle:
        Object tracked
            -> downward movement confirmed
            -> landing/settling confirmed
            -> HIGH event
            -> AMR approach verified
            -> RED escalation linked to HIGH

    There are no frame-index-based event triggers.
    """

    def __init__(
        self,
        config_path: str,
        path_network_path: str,
        shared_state: SharedState,
    ):
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        self.shared_state = shared_state
        self.path_network = PathNetwork(path_network_path)

        self.signal_evaluator = SignalEvaluator(
            self.config["amr_escalation"],
            self.path_network,
        )

        self.evidence_manager = EvidenceManager("evidence")
        self.model = YOLO(self.config["model"]["path"])

        self.trackers = {}
        self.amr_tracks = {}
        self.ring_buffer = deque()

        self.frame_idx = 0

        self.fall_config = self.config.get("fall_detection", {})

        self.min_drop_px = float(
            self.fall_config.get("min_vertical_drop_px", 35)
        )

        self.min_velocity_px_s = float(
            self.fall_config.get("min_vertical_velocity_px_s", 80)
        )

        self.landing_motion_px = float(
            self.fall_config.get("landing_motion_px", 8)
        )

        self.landing_frames = int(
            self.fall_config.get("landing_frames", 8)
        )

        self.fall_window_seconds = float(
            self.fall_config.get("fall_window_seconds", 1.5)
        )

        self.amr_distance_px = float(
            self.config.get("amr_escalation", {}).get(
                "approach_distance_px",
                150,
            )
        )

        self.amr_confirm_frames = int(
            self.config.get("amr_escalation", {}).get(
                "consecutive_frames",
                3,
            )
        )

        self.incident_counter = 0
        self.incidents = {}

        self.fallen_onto_ground = False
        self.amr_approach = False
        self.current_signal_level = "low"

    # ---------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------

    @staticmethod
    def _timestamp():
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    @staticmethod
    def _bbox_bottom_center(bbox):
        x1, y1, x2, y2 = bbox
        return (
            (x1 + x2) / 2.0,
            y2,
        )

    @staticmethod
    def _bbox_distance(box_a, box_b):
        center_a = VisionPipeline._bbox_center(box_a)
        center_b = VisionPipeline._bbox_center(box_b)

        return math.dist(center_a, center_b)

    @staticmethod
    def _bbox_dimensions(bbox):
        x1, y1, x2, y2 = bbox

        return [
            int(max(0, x2 - x1)),
            int(max(0, y2 - y1)),
        ]

    @staticmethod
    def _class_name(names, class_id):
        if isinstance(names, dict):
            return str(names.get(class_id, class_id)).lower()

        if isinstance(names, list):
            if 0 <= class_id < len(names):
                return str(names[class_id]).lower()

        return str(class_id).lower()

    def _is_amr_class(self, class_name):
        configured_classes = self.config.get(
            "amr_escalation",
            {},
        ).get(
            "amr_classes",
            ["amr", "agv", "robot", "mobile_robot"],
        )

        class_name = class_name.lower()

        return any(
            configured_name.lower() in class_name
            for configured_name in configured_classes
        )

    def _is_fall_object_class(self, class_name):
        configured_classes = self.config.get(
            "fall_detection",
            {},
        ).get(
            "object_classes",
            [],
        )

        if not configured_classes:
            return not self._is_amr_class(class_name)

        return class_name.lower() in [
            str(name).lower()
            for name in configured_classes
        ]

    # ---------------------------------------------------------
    # Track history
    # ---------------------------------------------------------

    def _get_track(self, track_id):
        if track_id not in self.trackers:
            self.trackers[track_id] = {
                "history": deque(maxlen=100),
                "fall_started": False,
                "fall_confirmed": False,
                "landing_count": 0,
                "event_id": None,
                "last_bbox": None,
            }

        return self.trackers[track_id]

    def _update_track_history(
        self,
        track_id,
        bbox,
        timestamp,
    ):
        track = self._get_track(track_id)

        track["history"].append({
            "timestamp": timestamp,
            "bbox": bbox,
            "bottom_center": self._bbox_bottom_center(bbox),
        })

        track["last_bbox"] = bbox

        return track

    # ---------------------------------------------------------
    # Fall confirmation
    # ---------------------------------------------------------

    def _check_fall(self, track):
        """
        Return True only after downward movement and settling
        are both confirmed.

        A static object is not a fall.
        """

        history = track["history"]

        if len(history) < 4:
            return False

        current = history[-1]
        current_time = current["timestamp"]
        current_y = current["bottom_center"][1]

        recent = [
            observation
            for observation in history
            if current_time - observation["timestamp"]
            <= self.fall_window_seconds
        ]

        if len(recent) < 4:
            return False

        first = recent[0]

        elapsed = current_time - first["timestamp"]

        if elapsed <= 0:
            return False

        first_y = first["bottom_center"][1]

        # Image coordinates increase downward.
        vertical_drop = current_y - first_y
        vertical_velocity = vertical_drop / elapsed

        downward_motion = (
            vertical_drop >= self.min_drop_px
            and vertical_velocity >= self.min_velocity_px_s
        )

        if not downward_motion:
            track["landing_count"] = 0
            return False

        track["fall_started"] = True

        # Confirm that the object has settled.
        recent_landing = list(history)[-self.landing_frames:]

        if len(recent_landing) < self.landing_frames:
            return False

        landing_y_values = [
            observation["bottom_center"][1]
            for observation in recent_landing
        ]

        landing_range = (
            max(landing_y_values)
            - min(landing_y_values)
        )

        if landing_range <= self.landing_motion_px:
            track["landing_count"] += 1
        else:
            track["landing_count"] = 0

        return (
            track["landing_count"] >= self.landing_frames
        )

    # ---------------------------------------------------------
    # AMR tracking
    # ---------------------------------------------------------

    def _update_amr_track(
        self,
        amr_track_id,
        bbox,
        timestamp,
    ):
        if amr_track_id not in self.amr_tracks:
            self.amr_tracks[amr_track_id] = {
                "history": deque(maxlen=100),
            }

        self.amr_tracks[amr_track_id]["history"].append({
            "timestamp": timestamp,
            "bbox": bbox,
        })

    def _amr_approaching(self, incident):
        """
        Check whether a tracked AMR is near the fallen object
        for the configured number of consecutive observations.
        """

        fallen_bbox = incident["bbox"]

        for amr_id, amr_data in self.amr_tracks.items():
            history = list(amr_data["history"])

            if len(history) < self.amr_confirm_frames:
                continue

            recent = history[-self.amr_confirm_frames:]

            distances = [
                self._bbox_distance(
                    fallen_bbox,
                    observation["bbox"],
                )
                for observation in recent
            ]

            # AMR must be within the configured distance on
            # consecutive observations.
            if all(
                distance <= self.amr_distance_px
                for distance in distances
            ):
                return True, amr_id, distances[-1]

        return False, None, None

    # ---------------------------------------------------------
    # Event creation
    # ---------------------------------------------------------

    def _create_high_event(
        self,
        track_id,
        bbox,
        frame,
        frame_idx,
    ):
        self.incident_counter += 1

        event_id = f"FALL_{self.incident_counter:04d}"

        incident = {
            "event_id": event_id,
            "track_id": track_id,
            "bbox": bbox,
            "high_saved": True,
            "red_saved": False,
            "created_at": self._timestamp(),
        }

        self.incidents[event_id] = incident

        self.fallen_onto_ground = True
        self.current_signal_level = "high"

        x1, y1, x2, y2 = bbox

        payload = {
            "Event": "Obstacle",
            "Anomaly_Type": "Fallen_Component",
            "event_id": event_id,
            "linked_to": None,
            "Timestamp": self._timestamp(),
            "signal_level": "high",
            "fallen_onto_ground": True,
            "amr_approach": False,
            "uncertain": False,
            "flags": [],
            "coords_source": "pixel_space",
            "Coordinates": None,
            "Coordinates_Unit": "pixels",
            "Dimensions": self._bbox_dimensions(bbox),
            "pixel": [
                int((x1 + x2) / 2),
                int((y1 + y2) / 2),
            ],
            "yolo_class": "unknown",
            "track_id": track_id,
            "frame": frame_idx,
            "amr_id": None,
            "amr_pixel_distance": None,
        }

        justification = {
            "event_id": event_id,
            "timestamp": self._timestamp(),
            "fallen_object_footprint_pixel_box": [
                [x1, y1],
                [x2, y2],
            ],
            "path_network_matching": {
                "corridor_id": None,
                "path_file": "path_network.json",
            },
            "spatial_risk_analysis": {
                "interior_overlap_percentage": None,
                "strip_overlap_percentage": None,
                "determined_level": "high",
                "justification_note": (
                    "Fall confirmed from downward movement "
                    "followed by landing/settling."
                ),
            },
            "amr_escalation_justification": {
                "amr_track_id": None,
                "approach_condition_met": False,
                "final_signal_level": "high",
                "booleans_updated": {
                    "fallen_onto_ground": True,
                    "amr_approach": False,
                },
            },
        }

        self.evidence_manager.save_evidence(
            event_id,
            "high",
            frame,
            list(self.ring_buffer),
            payload,
            justification,
        )

        self.shared_state.add_event(payload)

        print(
            f"[HIGH SAVED] {event_id} | "
            f"Track: {track_id} | "
            f"Fall confirmed"
        )

        return incident

    def _create_red_event(
        self,
        incident,
        amr_id,
        amr_distance,
        frame,
        frame_idx,
    ):
        event_id = f"{incident['event_id']}_RED"

        incident["red_saved"] = True

        self.amr_approach = True
        self.current_signal_level = "red"

        x1, y1, x2, y2 = incident["bbox"]

        payload = {
            "Event": "Obstacle",
            "Anomaly_Type": "Fallen_Component",
            "event_id": event_id,
            "linked_to": incident["event_id"],
            "Timestamp": self._timestamp(),
            "signal_level": "red",
            "fallen_onto_ground": True,
            "amr_approach": True,
            "uncertain": False,
            "flags": [],
            "coords_source": "pixel_space",
            "Coordinates": None,
            "Coordinates_Unit": "pixels",
            "Dimensions": self._bbox_dimensions(
                incident["bbox"]
            ),
            "pixel": [
                int((x1 + x2) / 2),
                int((y1 + y2) / 2),
            ],
            "yolo_class": "unknown",
            "track_id": incident["track_id"],
            "frame": frame_idx,
            "amr_id": amr_id,
            "amr_pixel_distance": amr_distance,
        }

        justification = {
            "event_id": event_id,
            "linked_to": incident["event_id"],
            "timestamp": self._timestamp(),
            "fallen_object_footprint_pixel_box": [
                [x1, y1],
                [x2, y2],
            ],
            "path_network_matching": {
                "corridor_id": None,
                "path_file": "path_network.json",
            },
            "spatial_risk_analysis": {
                "interior_overlap_percentage": None,
                "strip_overlap_percentage": None,
                "determined_level": "high",
                "justification_note": (
                    "AMR approach verified for the confirmed "
                    "fallen-object incident."
                ),
            },
            "amr_escalation_justification": {
                "amr_track_id": amr_id,
                "approach_condition_met": True,
                "final_signal_level": "red",
                "booleans_updated": {
                    "fallen_onto_ground": True,
                    "amr_approach": True,
                },
            },
        }

        self.evidence_manager.save_evidence(
            event_id,
            "red",
            frame,
            list(self.ring_buffer),
            payload,
            justification,
        )

        self.shared_state.add_event(payload)

        print(
            f"[RED ESCALATION SAVED] {event_id} "
            f"(Linked to {incident['event_id']}) | "
            f"AMR Track: {amr_id}"
        )

    # ---------------------------------------------------------
    # Main processing loop
    # ---------------------------------------------------------

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self.shared_state.add_warning(
                f"Failed to open video source: {video_path}"
            )
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        frame_idx = 0
        max_buffer_length = int(3.5 * fps)

        print(
            f"Processing video frames from: {video_path}..."
        )

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1

            timestamp = frame_idx / fps

            self.ring_buffer.append(frame.copy())

            while len(self.ring_buffer) > max_buffer_length:
                self.ring_buffer.popleft()

            results = self.model.track(
                frame,
                imgsz=self.config["model"]["imgsz"],
                conf=self.config["model"].get("conf", 0.25),
                persist=True,
                tracker=self.config["model"].get(
                    "tracker",
                    "bytetrack.yaml",
                ),
                verbose=False,
            )

            if not results:
                continue

            result = results[0]
            boxes = result.boxes

            if boxes is None or boxes.id is None:
                self.shared_state.update_booleans(
                    self.fallen_onto_ground,
                    self.amr_approach,
                    self.current_signal_level,
                )
                self.shared_state.update_frame(frame)
                continue

            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            bboxes = boxes.xyxy.cpu().numpy().tolist()

            for track_id, class_id, confidence, bbox in zip(
                track_ids,
                class_ids,
                confidences,
                bboxes,
            ):
                bbox = tuple(
                    float(value)
                    for value in bbox
                )

                class_name = self._class_name(
                    result.names,
                    class_id,
                )

                # ---------------------------------------------
                # AMR detection and tracking
                # ---------------------------------------------

                if self._is_amr_class(class_name):
                    self._update_amr_track(
                        track_id,
                        bbox,
                        timestamp,
                    )
                    continue

                if not self._is_fall_object_class(class_name):
                    continue

                # ---------------------------------------------
                # Object tracking and fall confirmation
                # ---------------------------------------------

                track = self._update_track_history(
                    track_id,
                    bbox,
                    timestamp,
                )

                if track["fall_confirmed"]:
                    continue

                fall_confirmed = self._check_fall(track)

                if fall_confirmed:
                    track["fall_confirmed"] = True

                    incident = self._create_high_event(
                        track_id,
                        bbox,
                        frame,
                        frame_idx,
                    )

                    track["event_id"] = incident["event_id"]

            # ---------------------------------------------
            # AMR escalation
            # ---------------------------------------------

            for incident in self.incidents.values():
                if incident["red_saved"]:
                    continue

                approach_met, amr_id, distance = (
                    self._amr_approaching(incident)
                )

                if approach_met:
                    self._create_red_event(
                        incident,
                        amr_id,
                        distance,
                        frame,
                        frame_idx,
                    )

            self.shared_state.update_booleans(
                self.fallen_onto_ground,
                self.amr_approach,
                self.current_signal_level,
            )

            self.shared_state.update_frame(frame)

        cap.release()

        print("\nProcessing complete.")
        print(
            f"Confirmed fallen incidents: "
            f"{len(self.incidents)}"
        )
        print(
            f"Fallen onto ground: "
            f"{self.fallen_onto_ground}"
        )
        print(
            f"AMR approach: "
            f"{self.amr_approach}"
        )
        print(
            f"Signal level: "
            f"{self.current_signal_level}"
        )