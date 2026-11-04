import sys
import os
import time
import argparse
from core.state import SharedState
from core.pipeline import VisionPipeline

def main():
    parser = argparse.ArgumentParser(description="Fallen Object Detection & Segregation CLI Pipeline")
    parser.add_argument("video", type=str, help="Path to input video file (e.g., input.mp4)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--path-network", type=str, default="config/path_network.json", help="Path to path_network.json")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Error: Video file '{args.video}' not found.")
        sys.exit(1)

    shared_state = SharedState()
    pipeline = VisionPipeline(args.config, args.path_network, shared_state)

    print("================================================================================")
    print("🚀 STARTING FALLEN OBJECT DETECTION & SEGREGATION PIPELINE")
    print("================================================================================")
    print(f"Input Video : {args.video}")
    print(f"Config File : {args.config}")
    print(f"Path Network: {args.path_network}")
    print("--------------------------------------------------------------------------------")

    pipeline.process_video(args.video)

    snapshot = shared_state.get_snapshot()

    print("\n================================================================================")
    print("📋 FINAL EXECUTION SUMMARY & STATE BOOLEANS")
    print("================================================================================")
    print(f"fallen_onto_ground : {snapshot['fallen_onto_ground']}")
    print(f"amr_approach       : {snapshot['amr_approach']}")
    print(f"Signal Level       : {snapshot['current_signal_level']}")
    print(f"Total Events Logged: {len(snapshot['events'])}")

    if snapshot['warnings']:
        print("\n⚠️ WARNINGS LOGGED:")
        for w in snapshot['warnings']:
            print(f"  - {w}")

    print("\n📁 EVIDENCE DIRECTORY VERIFICATION:")
    for level in ["low", "medium", "high", "red", "uncertain", "justification"]:
        path = os.path.join("evidence", level)
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"  - evidence/{level:14s}: {count} item(s)")

    print("================================================================================\n")

if __name__ == "__main__":
    main()