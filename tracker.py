"""
Render inference detections onto the original video.

Modes:
  realtime (default) — chạy song song với inference, hiển thị ngay khi có detection
  post               — đọc detections.json sau khi inference xong, render ra output.mp4

Usage:
    python tracker.py                            # realtime, video từ config.yaml
    python tracker.py --mode post                # render ra output.mp4 sau khi xong
    python tracker.py --video path/video.mp4
    python tracker.py --stream detections_stream.jsonl
    python tracker.py --detections detections.json --mode post
"""
import os
import cv2
import json
import time
import argparse
import yaml

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

COLOR = (0, 255, 0)


def draw_dets(frame, dets):
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cls = det["class"]
        score = det["score"]
        name = COCO_NAMES[cls] if cls < len(COCO_NAMES) else str(cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
        cv2.putText(frame, f"{name} {score:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1)
    return frame


def get_video_path():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["data"]


parser = argparse.ArgumentParser(description="Tracker: render detections lên video")
parser.add_argument("--mode", choices=["realtime", "post"], default="realtime",
                    help="realtime=chạy song song inference | post=render file sau khi xong")
parser.add_argument("--video", default=None)
parser.add_argument("--stream", default="detections_stream.jsonl", help="Stream file (realtime mode)")
parser.add_argument("--detections", default="detections.json", help="Detections file (post mode)")
parser.add_argument("--output", default="output.mp4", help="Output video (post mode)")
args = parser.parse_args()

if args.video is None:
    args.video = get_video_path()

# ─── REALTIME MODE ────────────────────────────────────────────────────────────
if args.mode == "realtime":
    print(f"[Tracker] Realtime mode — polling {args.stream}")
    print(f"[Tracker] Video: {args.video}  |  Press Q to quit")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        exit(1)

    pending = {}       # {frame_num: dets}
    frame_num = 0
    stream_pos = 0
    timeout = 30       # giây không có data mới (sau khi inference đã bắt đầu) thì dừng
    received_any = False
    last_data_time = None

    if os.path.exists(args.stream):
        os.remove(args.stream)

    print("[Tracker] Waiting for inference to start ...")

    while True:
        # Đọc các dòng mới từ stream file
        try:
            with open(args.stream, "r") as f:
                f.seek(stream_pos)
                new_lines = f.readlines()
                stream_pos = f.tell()
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                pending[entry["frame"]] = entry["dets"]
                if not received_any:
                    received_any = True
                    print("[Tracker] Inference started, displaying ...")
                last_data_time = time.time()
        except FileNotFoundError:
            pass  # chưa có file, chờ inference bắt đầu

        # Hiển thị frame tiếp theo nếu đã có detection
        next_frame = frame_num + 1
        if next_frame in pending:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            frame = cv2.resize(frame, (640, 640))
            frame = draw_dets(frame, pending.pop(next_frame))
            cv2.imshow("Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Timeout chỉ tính sau khi đã nhận data đầu tiên
            if received_any and time.time() - last_data_time > timeout:
                print(f"[Tracker] No new data for {timeout}s, inference done.")
                break
            time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()
    print(f"[Tracker] Done ({frame_num} frames displayed)")

# ─── POST MODE ────────────────────────────────────────────────────────────────
else:
    print(f"[Tracker] Post mode — {args.detections} → {args.output}")

    with open(args.detections, "r") as f:
        detections = json.load(f)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 640))

    print(f"Rendering {total} frames at {fps:.1f} fps ...")
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        frame = cv2.resize(frame, (640, 640))
        frame = draw_dets(frame, detections.get(str(frame_num), []))
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[OK] Saved {args.output} ({frame_num} frames)")
