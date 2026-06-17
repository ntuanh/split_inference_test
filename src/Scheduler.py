import threading
import torch
import cv2
import pickle
from tqdm import tqdm
import copy
import time
import csv
import os
import psutil
import numpy as np

_detections_stream_lock = threading.Lock()

from src.Compress import Encoder,Decoder
import src.Log as Log
from src.Model import inference, postprocess_yolo
from src.Utils import RAM_GUARD_PAUSE_ACTION, RAM_GUARD_RESUME_ACTION

# Hard ceiling enforced by the broker itself: once an intermediate queue holds
# this many bytes of unconsumed messages, RabbitMQ drops the oldest message to
# make room (x-overflow=drop-head). This guarantees the broker's RAM usage for
# this queue can never exceed the cap, regardless of producer/consumer pace —
# a last-resort safety net behind the proactive throttle below.
INTERMEDIATE_QUEUE_MAX_BYTES = 256 * 1024 * 1024
INTERMEDIATE_QUEUE_ARGS = {
    'x-max-length-bytes': INTERMEDIATE_QUEUE_MAX_BYTES,
    'x-overflow': 'drop-head',
}
# Producer self-throttles once the queue is estimated to hold this many bytes,
# well below the hard ceiling — so drop-head should rarely actually trigger.
INTERMEDIATE_QUEUE_THROTTLE_BYTES = INTERMEDIATE_QUEUE_MAX_BYTES // 2

# RabbitMQ refuses any message larger than its configured max_message_size by
# CLOSING the publishing channel — which silently kills the rest of the run
# (DONE sentinel, metrics, STOP all fail with "Channel is closed"). The default
# is 16 MiB on RabbitMQ 4.x (128 MiB on 3.x). So batches whose serialized form
# exceeds this are split into CHUNK messages of this size on the wire and
# reassembled by the consumer. 8 MiB (+ ~100 B envelope) stays safely under the
# strictest default.
WIRE_CHUNK_BYTES = 8 * 1024 * 1024


class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device

        # Only remove this instance's own metrics file — global cleanup is done once before threads start
        own_metrics = f"metrics_raw_{str(client_id).replace('-', '')}.csv"
        if os.path.exists(own_metrics):
            try:
                os.remove(own_metrics)
            except PermissionError:
                Log.print_with_color(f"[!] Cannot delete {own_metrics} (file is open). Close it and retry.", "red")

        # Raw per-batch timestamp logs (nanosecond "start"/"get input"/"output"/"end"
        # markers) written by first_layer/last_layer. Reset at the start of each run.
        cid_short = str(client_id).replace('-', '')[:12]
        self._timing_log_edge  = f"timing_edge_{cid_short}.log"
        self._timing_log_cloud = f"timing_cloud_{cid_short}.log"
        for tlog in [self._timing_log_edge, self._timing_log_cloud]:
            if os.path.exists(tlog):
                try:
                    os.remove(tlog)
                except Exception:
                    pass

        self.size_message = None
        self.intermediate_queue = f"intermediate_queue"
        self.channel.queue_declare(self.intermediate_queue, durable=False, arguments=INTERMEDIATE_QUEUE_ARGS)
        self._my_metrics_queue = None  # set by _setup_metrics_fanout_queue
        self._reply_queue = f"reply_{self.client_id}"
        self._send_paused = False  # set by the server's RAM guard via PAUSE_SEND/RESUME_SEND

        self.map_metric = None
        self.gt_dict = {}
        self._det_results = {}
        self._load_gt_dict()

        self.prev_time = None 

    def get_ram_mb(self):
        try:
            import subprocess, re
            result = subprocess.run(
                ['tegrastats', '--once'],
                capture_output=True, text=True, timeout=2
            )
            m = re.search(r'RAM (\d+)/\d+MB', result.stdout)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def write_metrics(self, mode, role, best_cut, batch_id, batch_size, latency_ms, fps, ram_mb, message_size_bytes=0, e2e_latency_ms=0, edge_start_time=None):
        file_path = f"metrics_raw_{str(self.client_id).replace('-', '')}.csv"
        file_exists = os.path.exists(file_path)

        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "mode",
                    "role",
                    "best_cut",
                    "batch_id",
                    "batch_size",
                    "latency_ms",
                    "fps",
                    "ram_mb",
                    "message_size_bytes",
                    "e2e_latency_ms",
                    "edge_start_time",
                ])

            writer.writerow([
                mode,
                role,
                best_cut,
                batch_id,
                batch_size,
                round(latency_ms, 3),
                round(fps, 3) if fps > 0 else "",  # fps=0 (first batch) → empty
                round(ram_mb, 3),
                message_size_bytes,
                round(e2e_latency_ms, 3),
                edge_start_time if edge_start_time is not None else "",
            ])

    def _setup_metrics_fanout_queue(self):
        """Cloud client gọi trước khi inference: tạo queue riêng bind vào fanout exchange.
        Mỗi cloud nhận một bản copy metrics từ tất cả edge trong cluster."""
        exchange = f"metrics_fanout_{self.intermediate_queue}"
        my_queue = f"mfq_{str(self.client_id).replace('-', '')}"
        try:
            self.channel.exchange_declare(exchange=exchange, exchange_type='fanout', durable=False)
            self.channel.queue_declare(my_queue, durable=False)
            self.channel.queue_bind(queue=my_queue, exchange=exchange)
            self._my_metrics_queue = my_queue
        except Exception as e:
            Log.print_with_color(f"[Metrics] Fanout setup failed: {e}", "yellow")
            self._my_metrics_queue = None

    def _poll_send_control(self):
        """Drain PAUSE_SEND/RESUME_SEND messages the server's RAM guard has
        dropped on our reply queue and update self._send_paused accordingly.
        Non-blocking. Returns the first non-control message encountered (e.g.
        the server's final STOP) so the caller can handle it, or None."""
        while True:
            _, _, body = self.channel.basic_get(queue=self._reply_queue, auto_ack=True)
            if not body:
                return None
            try:
                control = pickle.loads(body)
            except Exception:
                continue
            action = control.get("action")
            if action == RAM_GUARD_PAUSE_ACTION:
                if not self._send_paused:
                    self._send_paused = True
                    Log.print_with_color("[RAM-Guard] PAUSE received from server — holding off.", "yellow")
            elif action == RAM_GUARD_RESUME_ACTION:
                if self._send_paused:
                    self._send_paused = False
                    Log.print_with_color("[RAM-Guard] RESUME received from server — continuing.", "green")
            else:
                return control

    def _wait_for_queue_capacity(self):
        """Block the producer while it isn't safe/permitted to publish the next
        batch: either the server's RAM guard has told us to pause (broker memory
        running high), or the intermediate queue is estimated to already be
        holding too many bytes. The latter uses message_count * size of the last
        published message as a byte estimate (messages are uniform size within a
        run — same batch_size/compression). Together these keep the producer at
        the consumer's pace instead of flooding the broker's RAM."""
        while True:
            leftover = self._poll_send_control()
            if leftover is not None:
                # Nothing but RAM-guard control should arrive here (the server's
                # STOP only comes after we report our own) — log it just in case.
                Log.print_with_color(f"[!] Unexpected message while sending: {leftover}", "yellow")
            if self._send_paused:
                time.sleep(0.5)
                continue

            q = self.channel.queue_declare(self.intermediate_queue, passive=True)
            count = q.method.message_count
            if self.size_message:
                if count * self.size_message < INTERMEDIATE_QUEUE_THROTTLE_BYTES:
                    break
            elif count < 2:
                break
            time.sleep(0.5)

    def send_next_layer(self, intermediate_queue, data, compress):
        timings = {}

        t0 = time.perf_counter()  # compress (or just move to cpu if compression disabled)
        if compress["enable"]:
            data["data"] = [t.cpu().numpy() if isinstance(t, torch.Tensor) else None for t in
                                     data["data"]]
            data["data"], data["shape"] = Encoder(data_output=data["data"], num_bits=compress["num_bit"])

        else:
            data["data"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in
                                     data["data"]]
        timings["compress_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()  # serialize the message for the wire
        message = pickle.dumps({
            "action": "OUTPUT",
            "data": data
        })
        self.size_message = len(message)
        timings["serialize_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()  # publish to the intermediate queue
        if len(message) <= WIRE_CHUNK_BYTES:
            self.channel.basic_publish(
                exchange='',
                routing_key=intermediate_queue,
                body=message,
            )
        else:
            self._publish_chunked(intermediate_queue, message)
        timings["publish_ms"] = (time.perf_counter() - t0) * 1000

        return timings

    def _publish_chunked(self, intermediate_queue, message):
        """Send a serialized batch that exceeds WIRE_CHUNK_BYTES as a sequence of
        CHUNK messages the consumer reassembles. Keeps every wire message under
        the broker's max_message_size (which would otherwise close the channel)
        and paces the chunks so their combined size never trips the queue's
        x-max-length-bytes drop-head cap."""
        total_chunks = -(-len(message) // WIRE_CHUNK_BYTES)
        Log.print_with_color(
            f"[Chunked] Batch is {len(message) / 1e6:.0f} MB — sending as "
            f"{total_chunks} chunks of <= {WIRE_CHUNK_BYTES / 1e6:.0f} MB.", "cyan")
        # Cap in-queue chunk bytes at the throttle threshold (half the queue's
        # hard byte cap), so a slow consumer can never push us into drop-head.
        max_waiting = max(1, INTERMEDIATE_QUEUE_THROTTLE_BYTES // WIRE_CHUNK_BYTES)
        for seq in range(total_chunks):
            while True:
                q = self.channel.queue_declare(intermediate_queue, passive=True)
                if q.method.message_count < max_waiting:
                    break
                time.sleep(0.1)
            self.channel.basic_publish(
                exchange='',
                routing_key=intermediate_queue,
                body=pickle.dumps({
                    "action": "CHUNK",
                    "seq": seq,
                    "total": total_chunks,
                    "payload": message[seq * WIRE_CHUNK_BYTES:(seq + 1) * WIRE_CHUNK_BYTES],
                }),
            )

    def _load_gt_dict(self, gt_dir="datasets/groundtruth"):
        if not os.path.isdir(gt_dir):
            return
        try:
            from torchmetrics.detection import MeanAveragePrecision
            self.map_metric = MeanAveragePrecision(iou_type="bbox")
            self.map_metric.warn_on_many_detections = False
        except ImportError:
            Log.print_with_color("[!] torchmetrics not installed, mAP disabled", "red")
            return
        for fname in sorted(os.listdir(gt_dir)):
            if not fname.endswith(".txt"):
                continue
            try:
                num = int(os.path.splitext(fname)[0].split("_")[-1])
            except ValueError:
                continue
            boxes, labels = [], []
            with open(os.path.join(gt_dir, fname)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls, cx, cy, bw, bh = map(float, parts[:5])
                    boxes.append([(cx - bw/2)*640, (cy - bh/2)*640,
                                  (cx + bw/2)*640, (cy + bh/2)*640])
                    labels.append(int(cls))
            self.gt_dict[num] = {
                "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4)),
                "labels": torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros(0, dtype=torch.int64),
            }
        Log.print_with_color(f"[mAP] Loaded GT for {len(self.gt_dict)} frames from '{gt_dir}'", "green")

    def _update_map(self, batch_results, batch_id, batch_size, map_results=None):
        import json
        # map_results uses conf≈0.001 so torchmetrics gets the full PR curve;
        # batch_results (conf=0.25) is only for the detection stream / display.
        _map = map_results if map_results is not None else batch_results
        for img_idx, (r, rm) in enumerate(zip(batch_results, _map)):
            frame_num = batch_id * batch_size + img_idx + 1
            dets = [
                {
                    "box":   r["boxes"][i].cpu().tolist(),
                    "score": round(float(r["scores"][i]), 4),
                    "class": int(r["classes"][i]),
                }
                for i in range(len(r["boxes"]))
            ]
            self._det_results[frame_num] = dets
            with _detections_stream_lock:
                with open("detections_stream.jsonl", "a") as f:
                    f.write(json.dumps({"frame": frame_num, "dets": dets}) + "\n")
            if self.map_metric is None or frame_num not in self.gt_dict:
                continue
            self.map_metric.update(
                [{"boxes":  rm["boxes"].cpu().float(),
                  "scores": rm["scores"].cpu().float(),
                  "labels": rm["classes"].cpu().long()}],
                [self.gt_dict[frame_num]]
            )

    def _print_map(self):
        if self.map_metric is None:
            Log.print_with_color("[mAP] Skipped: groundtruth not found on this device (datasets/groundtruth/ missing)", "yellow")
            return
        if not self.gt_dict:
            Log.print_with_color("[mAP] Skipped: groundtruth folder exists but no valid .txt files loaded", "yellow")
            return
        try:
            result = self.map_metric.compute()
            print("=" * 50)
            print(f"  [mAP]   mAP@50={result['map_50']:.4f}  mAP@50:95={result['map']:.4f}")
            print("=" * 50)
        except Exception as e:
            Log.print_with_color(f"[mAP] compute failed: {e}", "red")

    def _write_detections_json(self):
        import json
        out = "detections.json"
        with open(out, "w") as f:
            json.dump({str(k): v for k, v in sorted(self._det_results.items())}, f)
        Log.print_with_color(f"[Tracker] Saved {out} ({len(self._det_results)} frames)", "green")

    def send_to_server(self, message):
        print("[DEBUG] sent to server ")
        print(message)
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

    def first_layer(self, model, data, batch_size, splits, logger, compress, mode="split", save_set=None, total_layers=None):
        orig_images = []
        input_image = []
        model.eval()
        model.to(self.device)

        video_path = data
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            Log.print_with_color(f"Not open video", "red")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        batch_id = 0
        prev_batch_end = None  # perf_counter() of previous batch's end, used to compute fps
        with open(self._timing_log_edge, "w") as _tf:
            print(str(time.time_ns()) + " start", file=_tf)  # raw timestamp: edge loop started
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 640))
                orig_images.append(copy.deepcopy(frame))
                frame = frame.astype('float32') / 255.0
                tensor = torch.from_numpy(frame).permute(2, 0, 1)  # shape: (3, 640, 640)
                input_image.append(tensor)

                if len(input_image) == batch_size:
                    with open(self._timing_log_edge, "a") as _tf:
                        print(str(time.time_ns()) + " get input", file=_tf)  # raw timestamp: batch of frames ready
                    batch_start = time.perf_counter()  # start of this batch's "total" timing window (-> latency_ms)
                    edge_start_wall = time.time()  # wall-clock time embedded in the message for e2e latency on the cloud side

                    t0 = time.perf_counter()  # stack frames into a batch + move to device
                    input_image = torch.stack(input_image)
                    input_image = input_image.to(self.device)
                    to_device_ms = (time.perf_counter() - t0) * 1000

                    # Defaults for stages that don't apply to every mode
                    inference_ms = 0.0
                    postprocess_ms = 0.0
                    queue_wait_ms = 0.0
                    compress_ms = 0.0
                    serialize_ms = 0.0
                    publish_ms = 0.0

                    # ===== ONLY CLOUD =====
                    if mode == "only_cloud":
                        frames_cpu = input_image.cpu()
                        y = {
                            "data": [frames_cpu[i].clone() for i in range(len(frames_cpu))],
                            "width": width,
                            "height": height,
                            "edge_start_time": edge_start_wall
                        }

                        t0 = time.perf_counter()  # throttle until intermediate queue has room
                        self._wait_for_queue_capacity()
                        queue_wait_ms = (time.perf_counter() - t0) * 1000

                        send_timings = self.send_next_layer(
                            self.intermediate_queue,
                            y,
                            {"enable": False}
                        )
                        compress_ms  = send_timings["compress_ms"]
                        serialize_ms = send_timings["serialize_ms"]
                        publish_ms   = send_timings["publish_ms"]

                    # ===== ONLY EDGE =====
                    elif mode == "only_edge":

                        t0 = time.perf_counter()  # full local inference
                        y = []
                        x, y = inference(model, input_image, y, 0, save_set)
                        inference_ms = (time.perf_counter() - t0) * 1000

                        t0 = time.perf_counter()  # NMS + mAP bookkeeping
                        results     = postprocess_yolo(x, conf_thres=0.25,  iou_thres=0.5)
                        map_results = postprocess_yolo(x, conf_thres=0.001, iou_thres=0.5)
                        self._update_map(results, batch_id, batch_size, map_results=map_results)
                        postprocess_ms = (time.perf_counter() - t0) * 1000

                    # ===== SPLIT INFERENCE =====
                    else:
                        # splits<=0: edge has no layers to run at all (layers[:0] == []).
                        edge_has_no_layers = (len(model) == 0)
                        # splits>=total_layers: cloud's slice (layers[splits:]) is empty —
                        # nothing downstream needs the skip-connection tensors anymore.
                        cloud_has_no_layers = (total_layers is not None and splits >= total_layers)

                        t0 = time.perf_counter()  # edge-side partial inference up to the split point
                        if edge_has_no_layers:
                            # Nothing to run locally — forward the raw batch untouched.
                            x = input_image
                            y = [x]
                        else:
                            y = []
                            x, y = inference(model, input_image, y, 0, save_set)
                            y[-1] = x
                            if cloud_has_no_layers:
                                # Cloud has nothing left to compute — the skip-connection
                                # tensors saved above would just be dead weight on the wire.
                                y = [x]
                        inference_ms = (time.perf_counter() - t0) * 1000

                        y = {
                            "data": y,
                            "width": width,
                            "height": height,
                            "edge_start_time": edge_start_wall
                        }

                        t0 = time.perf_counter()  # throttle until intermediate queue has room
                        self._wait_for_queue_capacity()
                        queue_wait_ms = (time.perf_counter() - t0) * 1000

                        send_timings = self.send_next_layer(
                            self.intermediate_queue, y, compress
                        )
                        compress_ms  = send_timings["compress_ms"]
                        serialize_ms = send_timings["serialize_ms"]
                        publish_ms   = send_timings["publish_ms"]
                    batch_end = time.perf_counter()  # end of this batch's "total" timing window
                    with open(self._timing_log_edge, "a") as _tf:
                        print(str(time.time_ns()) + " output", file=_tf)  # raw timestamp: batch finished processing
                    latency_ms = (batch_end - batch_start) * 1000  # total time to process this batch (= sum of stages below + overhead)
                    fps = batch_size / (batch_end - prev_batch_end) if prev_batch_end is not None else 0.0  # throughput since previous batch's end
                    e2e_latency_ms = latency_ms if mode == "only_edge" else 0.0  # only_edge: no cloud hop, so e2e == local latency

                    t0 = time.perf_counter()  # query current RAM usage
                    ram_mb = self.get_ram_mb()
                    ram_ms = (time.perf_counter() - t0) * 1000

                    msg_size = self.size_message if self.size_message is not None else 0

                    t0 = time.perf_counter()  # append row to metrics CSV
                    self.write_metrics(
                        mode=mode,
                        role="edge_sender" if mode == "only_cloud" else "edge",
                        best_cut="N/A" if splits is None else splits,
                        batch_id=batch_id,
                        batch_size=batch_size,
                        latency_ms=latency_ms,
                        fps=fps,
                        ram_mb=ram_mb,
                        message_size_bytes=msg_size,
                        e2e_latency_ms=e2e_latency_ms,
                        edge_start_time=edge_start_wall,
                    )
                    write_csv_ms = (time.perf_counter() - t0) * 1000

                    # Anything left over: dict/list bookkeeping, timing-log writes,
                    # latency/fps math, etc. (queue_wait excluded — it's a deliberate
                    # throttle, not work).
                    measured_ms = (to_device_ms + inference_ms + postprocess_ms +
                                    compress_ms + serialize_ms + publish_ms +
                                    ram_ms + write_csv_ms)
                    overhead_ms = latency_ms - queue_wait_ms - measured_ms

                    # Log.print_with_color(
                    #     f"[Timing] batch={batch_id} | to_device={to_device_ms:.2f}ms "
                    #     f"inference={inference_ms:.2f}ms postprocess={postprocess_ms:.2f}ms "
                    #     f"queue_wait={queue_wait_ms:.2f}ms compress={compress_ms:.2f}ms "
                    #     f"serialize={serialize_ms:.2f}ms publish={publish_ms:.2f}ms "
                    #     f"ram={ram_ms:.2f}ms write_csv={write_csv_ms:.2f}ms "
                    #     f"overhead={overhead_ms:.2f}ms | total={latency_ms:.2f}ms",
                    #     "header"
                    # )

                    batch_id += 1
                    prev_batch_end = batch_end  # remember end time so the next batch's fps can be computed

                    input_image = []
                    orig_images = []
                    pbar.update(batch_size)
                else:
                    continue
        except Exception as loop_err:
            Log.print_with_color(f"[!] Video loop error at batch {batch_id}: {loop_err}", "red")
        finally:
            with open(self._timing_log_edge, "a") as _tf:
                print(str(time.time_ns()) + " end", file=_tf)  # raw timestamp: edge loop exited (video done or error)
            print(f'size message: {self.size_message} bytes.')
            cap.release()
            pbar.close()

        # Always send DONE before notifying server — cloud cannot proceed without it
        if mode != "only_edge":
            try:
                self.channel.basic_publish(
                    exchange='',
                    routing_key=self.intermediate_queue,
                    body=pickle.dumps({"action": "DONE", "client_id": self.client_id})
                )
                Log.print_with_color("[>>>] Sent DONE sentinel to cloud.", "cyan")
            except Exception as done_err:
                Log.print_with_color(f"[!] Failed to send DONE sentinel: {done_err}", "red")

        # Broadcast metrics CSV lên tất cả cloud trong cluster qua fanout exchange
        metrics_file = f"metrics_raw_{str(self.client_id).replace('-', '')}.csv"
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'rb') as f:
                    metrics_data = f.read()
                exchange = f"metrics_fanout_{self.intermediate_queue}"
                self.channel.exchange_declare(exchange=exchange, exchange_type='fanout', durable=False)
                self.channel.basic_publish(
                    exchange=exchange,
                    routing_key='',
                    body=pickle.dumps({"action": "METRICS", "filename": os.path.basename(metrics_file), "data": metrics_data})
                )
                Log.print_with_color(f"[Metrics] Broadcast metrics via fanout ({len(metrics_data)} bytes)", "cyan")
            except Exception as e:
                Log.print_with_color(f"[Metrics] Failed to send metrics: {e}", "yellow")

        notify_data = {"action": "STOP", "client_id": self.client_id, "layer_id": self.layer_id,
                       "message": "Finish training!"}

        self.send_to_server(notify_data)
        # print(f"[ DEBUG ] send notify stop to server ")

        broadcast_queue_name = f'reply_{self.client_id}'
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:

                received_data = pickle.loads(body)
                Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "STOP":
                    Log.print_with_color("[>>>] Finish!", "red")
                    break
            time.sleep(0.5)


    def last_layer(self, model, batch_size, splits, logger, compress, mode="split", save_set=None):
        model.eval()
        model.to(self.device)

        self.channel.basic_qos(prefetch_count=10)
        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        batch_id = 0
        send_notify = False
        _empty_poll_count = 0
        _chunk_buf = {}  # seq -> payload, for reassembling oversized CHUNK'ed batches
        prev_batch_end = None  # perf_counter() of previous batch's end, used to compute fps
        with open(self._timing_log_cloud, "w") as _tf:
            print(str(time.time_ns()) + " start", file=_tf)  # raw timestamp: cloud loop started

        wait_start = time.perf_counter()  # start of "wait_network" timer for the first message
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=self.intermediate_queue, auto_ack=True)
            if method_frame and body:
                wait_ms = (time.perf_counter() - wait_start) * 1000  # time spent waiting for this message to arrive
                _empty_poll_count = 0
                received_message_size = len(body)

                t0 = time.perf_counter()  # start of "total" timing window + deserialize
                batch_start = t0
                received_data = pickle.loads(body)

                # Oversized batch arriving as CHUNK pieces — collect until the
                # set is complete, then reassemble into the original message.
                # Safe because each intermediate queue has exactly one producer,
                # so chunks of one batch arrive contiguously and in order.
                if received_data.get("action") == "CHUNK":
                    _chunk_buf[received_data["seq"]] = received_data["payload"]
                    if len(_chunk_buf) < received_data["total"]:
                        wait_start = time.perf_counter()
                        continue  # more chunks of this batch still in transit
                    body = b"".join(_chunk_buf[i] for i in range(received_data["total"]))
                    _chunk_buf = {}
                    received_message_size = len(body)
                    received_data = pickle.loads(body)

                deserialize_ms = (time.perf_counter() - t0) * 1000

                # Edge sent DONE sentinel — all frames have been transmitted
                if received_data.get("action") == "DONE":
                    if not send_notify:
                        Log.print_with_color("[ DEBUG ] inference completely !", "green")
                        notify_data = {"action": "STOP", "client_id": self.client_id,
                                       "layer_id": self.layer_id, "message": "Finish training!"}
                        self.send_to_server(notify_data)
                        send_notify = True
                    wait_start = time.perf_counter()
                    continue  # fall through to else branch next iteration to await server STOP

                with open(self._timing_log_cloud, "a") as _tf:
                    print(str(time.time_ns()) + " get input", file=_tf)  # raw timestamp: batch message received & deserialized
                y = received_data["data"]
                edge_start_time = y.get("edge_start_time", time.time())

                decompress_ms = 0.0  # only set in split-inference mode when compression is enabled

                # ===== ONLY CLOUD =====
                if mode == "only_cloud":
                    input_tensor = y["data"]

                    t1 = time.perf_counter()  # stack frames + move batch to device
                    if isinstance(input_tensor, list):
                        input_tensor = torch.stack(input_tensor)

                    input_tensor = input_tensor.to(self.device)
                    to_device_ms = (time.perf_counter() - t1) * 1000

                    t1 = time.perf_counter()  # full-model inference
                    x, _ = inference(model, input_tensor, [], 0, save_set)
                    inference_ms = (time.perf_counter() - t1) * 1000
                # ===== SPLIT INFERENCE =====
                else:

                    if compress["enable"]:
                        t1 = time.perf_counter()  # decompress the received tensors
                        y["data"] = Decoder(y["data"], y["shape"])

                        y["data"] = [
                            torch.from_numpy(t) if t is not None else None
                            for t in y["data"]
                        ]
                        decompress_ms = (time.perf_counter() - t1) * 1000

                    t1 = time.perf_counter()  # move received tensors to device
                    y["data"] = [
                        t.to(self.device) if t is not None else None
                        for t in y["data"]
                    ]
                    to_device_ms = (time.perf_counter() - t1) * 1000

                    list_output = y["data"]

                    t1 = time.perf_counter()  # remaining-layers inference
                    if splits == 0:
                        # Edge ran zero layers — list_output is just [raw_batch], not a
                        # layer-indexed accumulator. Run the whole model from scratch,
                        # same as only_cloud, but through the split-mode wire format.
                        x, _ = inference(model, list_output[-1], [], 0, save_set)
                    else:
                        x = list_output[-1]
                        x, _ = inference(model, x, list_output, splits, save_set)
                    inference_ms = (time.perf_counter() - t1) * 1000

                t1 = time.perf_counter()  # NMS post-processing
                results     = postprocess_yolo(x, conf_thres=0.25,  iou_thres=0.5)
                map_results = postprocess_yolo(x, conf_thres=0.001, iou_thres=0.5)
                nms_ms = (time.perf_counter() - t1) * 1000

                t1 = time.perf_counter()  # mAP bookkeeping + detection stream write
                self._update_map(results, batch_id, batch_size, map_results=map_results)
                map_update_ms = (time.perf_counter() - t1) * 1000

                batch_end = time.perf_counter()  # end of this batch's "total" timing window
                with open(self._timing_log_cloud, "a") as _tf:
                    print(str(time.time_ns()) + " output", file=_tf)  # raw timestamp: batch finished processing
                cloud_end_wall = time.time()  # wall-clock time, paired with edge_start_time for e2e latency
                latency_ms = (batch_end - batch_start) * 1000  # total time to process this batch (= sum of stages below + overhead)
                fps = batch_size / (batch_end - prev_batch_end) if prev_batch_end is not None else 0.0  # throughput since previous batch's end
                e2e_latency_ms = (cloud_end_wall - edge_start_time) * 1000  # edge-send -> cloud-done round trip

                t1 = time.perf_counter()  # query current RAM usage
                ram_mb = self.get_ram_mb()
                ram_ms = (time.perf_counter() - t1) * 1000

                t1 = time.perf_counter()  # append row to metrics CSV
                self.write_metrics(
                    mode=mode,
                    role="cloud",
                    best_cut="N/A" if splits is None else splits,
                    batch_id=batch_id,
                    batch_size=batch_size,
                    latency_ms=latency_ms,
                    fps=fps,
                    ram_mb=ram_mb,
                    message_size_bytes=received_message_size,
                    e2e_latency_ms=e2e_latency_ms,
                    edge_start_time=edge_start_time,
                )
                write_csv_ms = (time.perf_counter() - t1) * 1000

                # Anything left over: dict/list bookkeeping, timing-log writes,
                # latency/fps math, etc.
                measured_ms = (deserialize_ms + decompress_ms + to_device_ms + inference_ms +
                                nms_ms + map_update_ms + ram_ms + write_csv_ms)
                overhead_ms = latency_ms - measured_ms

                # Print the full per-batch timing breakdown: network wait, deserialize,
                # decompress, device transfer, inference, NMS, mAP update, RAM query,
                # CSV write, leftover overhead, and overall total.
                # Log.print_with_color(
                #     f"[Timing] batch={batch_id} | wait_network={wait_ms:.2f}ms "
                #     f"deserialize={deserialize_ms:.2f}ms decompress={decompress_ms:.2f}ms "
                #     f"to_device={to_device_ms:.2f}ms inference={inference_ms:.2f}ms "
                #     f"nms={nms_ms:.2f}ms map_update={map_update_ms:.2f}ms "
                #     f"ram={ram_ms:.2f}ms write_csv={write_csv_ms:.2f}ms "
                #     f"overhead={overhead_ms:.2f}ms | total={latency_ms:.2f}ms",
                #     "header"
                # )

                batch_id += 1
                prev_batch_end = batch_end  # remember end time so the next batch's fps can be computed
                wait_start = time.perf_counter()  # restart "wait_network" timer for the next message

                pbar.update(batch_size)

            elif send_notify is False and batch_id > 0:
                # Queue ran dry mid-run: handle RAM-guard PAUSE/RESUME so a guard
                # pause isn't mistaken for a lost DONE sentinel.
                leftover = self._poll_send_control()
                if leftover is not None and leftover.get("action") == "STOP":
                    Log.print_with_color("[>>>] Finish!", "red")
                    break
                if self._send_paused:
                    # Edge senders are paused by the server's RAM guard — an
                    # empty queue is expected, don't count it toward the fallback.
                    _empty_poll_count = 0
                else:
                    _empty_poll_count += 1
                    # Fallback: DONE sentinel lost — fire after 60 empty polls (~30 s)
                    if _empty_poll_count >= 60:
                        Log.print_with_color("[!] DONE sentinel not received — sending STOP via fallback.", "yellow")
                        notify_data = {"action": "STOP", "client_id": self.client_id, "layer_id": self.layer_id,
                               "message": "Finish training!"}
                        self.send_to_server(notify_data)
                        send_notify = True
                        _empty_poll_count = 0
                time.sleep(0.5)

            else:
                received_data = self._poll_send_control()
                if received_data is not None:
                    Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                    if received_data.get("action") == "STOP":
                        Log.print_with_color("[>>>] Finish!", "red")
                        break
                else:
                    time.sleep(0.5)

        with open(self._timing_log_cloud, "a") as _tf:
            print(str(time.time_ns()) + " end", file=_tf)  # raw timestamp: cloud loop exited (STOP received)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        pbar.close()

    def middle_layer(self, model):
        pass

    def _pivot_and_save(self):
        import glob as _glob

        lock_path = "metrics_pivot.lock"
        out_path = "metrics_pivoted.csv"

        # Chỉ 1 client thắng lock mới làm pivot (atomic exclusive create)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return  # Client khác đang làm pivot

        # Đợi các client còn lại ghi xong hàng cuối
        time.sleep(2.0)

        # Thu thập metrics CSV từ personal fanout queue (mỗi cloud có bản copy riêng)
        my_q = self._my_metrics_queue
        if my_q:
            try:
                while True:
                    method_frame, _, body = self.channel.basic_get(queue=my_q, auto_ack=True)
                    if not method_frame:
                        break
                    msg = pickle.loads(body)
                    if msg.get("action") == "METRICS":
                        fname = msg["filename"]
                        with open(fname, 'wb') as f:
                            f.write(msg["data"])
                        Log.print_with_color(f"[Metrics] Received remote metrics: {fname}", "cyan")
            except Exception as e:
                Log.print_with_color(f"[Metrics] Warning collecting remote metrics: {e}", "yellow")

        edge_rows = []
        cloud_rows = []

        edge_seq_counter = 0
        cloud_seq_counter = 0

        for fpath in sorted(_glob.glob("metrics_raw_*.csv")):
            with open(fpath, newline="") as f:
                rows_in_file = list(csv.DictReader(f))
            if not rows_in_file:
                continue
            role = rows_in_file[0]["role"]
            if role in ("edge", "edge_sender"):
                edge_seq_counter += 1
                for row in rows_in_file:
                    row["device_seq"] = edge_seq_counter
                    edge_rows.append(row)
            elif role == "cloud":
                cloud_seq_counter += 1
                for row in rows_in_file:
                    row["device_seq"] = cloud_seq_counter
                    cloud_rows.append(row)

        # Join edge ↔ cloud bằng edge_start_time (timestamp edge nhúng vào mỗi message)
        edge_by_time = {
            row["edge_start_time"]: row
            for row in edge_rows
            if row.get("edge_start_time")
        }
        matched_pairs = []
        matched_edge_times = set()
        for c in cloud_rows:
            t = c.get("edge_start_time", "")
            e = edge_by_time.get(t, {})
            matched_pairs.append((e, c))
            if t:
                matched_edge_times.add(t)
        # Edge rows không có cloud tương ứng (only_edge mode)
        for e in edge_rows:
            if e.get("edge_start_time", "") not in matched_edge_times:
                matched_pairs.append((e, {}))
        # Sắp xếp theo edge_start_time tăng dần
        matched_pairs.sort(key=lambda p: float(p[0].get("edge_start_time") or p[1].get("edge_start_time") or 0))

        n_rows = len(matched_pairs)
        fieldnames = [
            "batch_id", "batch_size", "best_cut",
            "edge_device", "edge_latency_ms", "edge_fps", "edge_ram_mb", "edge_message_size_bytes",
            "cloud_device", "cloud_arrival_order", "cloud_latency_ms", "cloud_fps", "cloud_ram_mb", "cloud_message_size_bytes",
            "e2e_latency_ms",
        ]

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, (e, c) in enumerate(matched_pairs):
                writer.writerow({
                    "batch_id":                i,
                    "batch_size":              e.get("batch_size") or c.get("batch_size", ""),
                    "best_cut":                e.get("best_cut")   or c.get("best_cut", ""),
                    "edge_device":             e.get("device_seq", ""),
                    "edge_latency_ms":         e.get("latency_ms", ""),
                    "edge_fps":                e.get("fps", ""),
                    "edge_ram_mb":             e.get("ram_mb", ""),
                    "edge_message_size_bytes": e.get("message_size_bytes", ""),
                    "cloud_device":            c.get("device_seq", ""),
                    "cloud_arrival_order":     c.get("batch_id", ""),
                    "cloud_latency_ms":        c.get("latency_ms", ""),
                    "cloud_fps":               c.get("fps", ""),
                    "cloud_ram_mb":            c.get("ram_mb", ""),
                    "cloud_message_size_bytes":c.get("message_size_bytes", ""),
                    "e2e_latency_ms":          c.get("e2e_latency_ms") or e.get("e2e_latency_ms", ""),
                })

        for fpath in _glob.glob("metrics_raw_*.csv"):
            try:
                os.remove(fpath)
            except FileNotFoundError:
                pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass

        def avg(rows, key, skip_zero_fps=False):
            filtered = rows
            if skip_zero_fps:
                filtered = [r for r in rows if float(r.get("fps") or 0) > 0]
            vals = [float(r[key]) for r in filtered if r.get(key)]
            return round(sum(vals) / len(vals), 3) if vals else None

        def total_fps(rows):
            # Với mỗi device: tính trung bình FPS qua các batch (bỏ batch fps=0)
            # Tổng hệ thống = cộng trung bình FPS của từng device
            by_device = {}
            for r in rows:
                seq = r.get("device_seq")
                val = float(r.get("fps") or 0)
                if val > 0 and seq is not None:
                    by_device.setdefault(seq, []).append(val)
            device_avgs = [sum(v) / len(v) for v in by_device.values() if v]
            return round(sum(device_avgs), 3) if device_avgs else None

        def mb(val):
            return round(val / 1024 / 1024, 3) if val is not None else "N/A"

        cuts = set(r.get("best_cut", "N/A") for r in (edge_rows or cloud_rows))
        cut_str = "/".join(sorted(str(c) for c in cuts))
        all_rows = cloud_rows if cloud_rows else edge_rows
        final_rows = cloud_rows if cloud_rows else edge_rows
        system_fps = total_fps(final_rows)
        valid_batches = len([r for r in final_rows if float(r.get("fps") or 0) > 0])
        # Edge metrics chỉ tính trên batch có cloud match (batch kia tính ở cloud kia)
        # Fallback về tất cả edge_rows nếu không có cloud (only_edge mode)
        matched_edge_rows = [e for e, c in matched_pairs if c and e]
        summary_edge_rows = matched_edge_rows if cloud_rows else edge_rows
        print("=" * 50)
        print(f"  SUMMARY  |  batches={n_rows} (valid={valid_batches})  cut={cut_str}")
        print("=" * 50)
        print(f"  [EDGE]  latency={avg(summary_edge_rows,'latency_ms',True)} ms  fps={avg(summary_edge_rows,'fps',True)}  ram={avg(summary_edge_rows,'ram_mb',True)} MB  msg={mb(avg(summary_edge_rows,'message_size_bytes'))} MB")
        print(f"  [CLOUD] latency={avg(cloud_rows,'latency_ms',True)} ms  fps={avg(cloud_rows,'fps',True)}  ram={avg(cloud_rows,'ram_mb',True)} MB  msg={mb(avg(cloud_rows,'message_size_bytes'))} MB")
        print(f"  [E2E]   latency={avg(all_rows,'e2e_latency_ms',True)} ms")
        print(f"  [SYSTEM TOTAL FPS] {system_fps} fps  (sum of avg fps across {len(set(r.get('device_seq') for r in final_rows))} final device(s))")
        print("=" * 50)
        Log.print_with_color(f"Saved metrics_pivoted.csv ({n_rows} batches)", "green")
        n_edge_devices = len(set(r.get("device_seq") for r in edge_rows))
        if n_edge_devices > 1:
            Log.print_with_color(
                f"[mAP] Skipped: {n_edge_devices} edge devices in this cluster — "
                f"frame alignment undefined for multi-edge mAP.", "yellow")
        else:
            self._print_map()

        if self._det_results:
            self._write_detections_json()

    def inference_func(self, model, data, num_layers, splits, batch_size, logger, compress, mode="split", queue_name="intermediate_queue", save_set=None, total_layers=None):
        try:
            os.remove("detections_stream.jsonl")
        except FileNotFoundError:
            pass
        if queue_name != self.intermediate_queue:
            self.intermediate_queue = queue_name
            self.channel.queue_declare(self.intermediate_queue, durable=False, arguments=INTERMEDIATE_QUEUE_ARGS)

        if self.layer_id == 1:
            try:
                self.first_layer(model, data, batch_size, splits, logger, compress, mode, save_set, total_layers)
            except Exception as e:
                Log.print_with_color(f"[!] Error during inference: {e} — saving metrics anyway.", "yellow")
            if mode == "only_edge":
                self._pivot_and_save()
        elif self.layer_id == num_layers:
            self._setup_metrics_fanout_queue()
            try:
                self.last_layer(model, batch_size, splits, logger, compress, mode, save_set)
            except Exception as e:
                Log.print_with_color(f"[!] Error during inference: {e} — saving metrics anyway.", "yellow")
            self._pivot_and_save()
        else:
            self.middle_layer(model)
