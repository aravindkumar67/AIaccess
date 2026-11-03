import json
from pathlib import Path
import cv2

class EvidenceWriter:
    """Persist event snapshot, video clip, and machine-readable event JSON."""
    def __init__(self, output_dir="evidence", pre_event_seconds=1.5, post_event_seconds=2.0):
        self.output_dir = Path(output_dir)
        self.pre_event_seconds = float(pre_event_seconds)
        self.post_event_seconds = float(post_event_seconds)

    def write_event(self, level, event_payload, snapshot_frame, clip_frames, fps=24.0):
        event_id = str(event_payload.get("event_id", "EVENT_0000"))
        folder = self.output_dir / str(level).lower() / event_id
        folder.mkdir(parents=True, exist_ok=True)
        snapshot_path = folder / "snapshot.jpg"
        clip_path = folder / "clip.mp4"
        json_path = folder / "data.json"

        if snapshot_frame is not None and not cv2.imwrite(str(snapshot_path), snapshot_frame):
            raise RuntimeError(f"Failed writing snapshot {snapshot_path}")

        frames = [frame for frame in clip_frames if frame is not None]
        if frames:
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"),
                max(float(fps), 1.0), (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed opening video writer for {clip_path}")
            try:
                for frame in frames:
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h))
                    writer.write(frame)
            finally:
                writer.release()

        payload = dict(event_payload)
        payload["evidence"] = {
            "snapshot": str(snapshot_path),
            "clip": str(clip_path),
            "json": str(json_path),
            "frame_count": len(frames),
            "fps": float(fps),
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(folder)
