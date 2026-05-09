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

from src.Compress import Encoder,Decoder
import src.Log as Log
from src.Model import inference, postprocess_yolo

class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device

        if self.layer_id == 1:
            import glob as _glob
            for f in _glob.glob("metrics_raw_*.csv") + ["metrics_pivoted.csv", "metrics_pivot.lock"]:
                if os.path.exists(f):
                    os.remove(f)

        self.size_message = None
        self.intermediate_queue = f"intermediate_queue"
        self.channel.queue_declare(self.intermediate_queue, durable=False)

        self.map_metric = None
        self.gt_dict = {}
        self._load_gt_dict()

    def get_ram_mb(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def write_metrics(self, mode, role, best_cut, batch_id, batch_size, latency_ms, fps, ram_mb, message_size_bytes=0, e2e_latency_ms=0):
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
                    "e2e_latency_ms"
                ])

            writer.writerow([
                mode,
                role,
                best_cut,
                batch_id,
                batch_size,
                round(latency_ms, 3),
                round(fps, 3),
                round(ram_mb, 3),
                message_size_bytes,
                round(e2e_latency_ms, 3)
            ])

    def send_next_layer(self, intermediate_queue, data, compress):

        if compress["enable"]:
            data["data"] = [t.cpu().numpy() if isinstance(t, torch.Tensor) else None for t in
                                     data["data"]]
            data["data"], data["shape"] = Encoder(data_output=data["data"], num_bits=compress["num_bit"])

        else:
            data["data"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in
                                     data["data"]]
        message = pickle.dumps({
            "action": "OUTPUT",
            "data": data
        })
        if self.size_message is None:
            self.size_message = len(message)


        self.channel.basic_publish(
            exchange='',
            routing_key=intermediate_queue,
            body=message,
            #body= "."
        )

    def _load_gt_dict(self, gt_dir="datasets/groundtruth"):
        if not os.path.isdir(gt_dir):
            return
        try:
            from torchmetrics.detection import MeanAveragePrecision
            self.map_metric = MeanAveragePrecision(iou_type="bbox")
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

    def _update_map(self, batch_results, batch_id, batch_size):
        if self.map_metric is None:
            return
        for img_idx, r in enumerate(batch_results):
            frame_num = batch_id * batch_size + img_idx + 1
            if frame_num not in self.gt_dict:
                continue
            self.map_metric.update(
                [{"boxes":  r["boxes"].cpu().float(),
                  "scores": r["scores"].cpu().float(),
                  "labels": r["classes"].cpu().long()}],
                [self.gt_dict[frame_num]]
            )

    def _print_map(self):
        if self.map_metric is None:
            return
        try:
            result = self.map_metric.compute()
            print("=" * 50)
            print(f"  [mAP]   mAP@50={result['map_50']:.4f}  mAP@50:95={result['map']:.4f}")
            print("=" * 50)
        except Exception as e:
            Log.print_with_color(f"[mAP] compute failed: {e}", "red")

    def send_to_server(self, message):
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

    def first_layer(self, model, data, batch_size, splits, logger, compress, mode="split"):
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
        prev_batch_end = None
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
                batch_start = time.perf_counter()

                input_image = torch.stack(input_image)
                input_image = input_image.to(self.device)

                # ===== ONLY CLOUD =====
                if mode == "only_cloud":

                    y = {
                        "data": input_image.cpu(),
                        "width": width,
                        "height": height,
                        "edge_start_time": batch_start
                    }

                    self.send_next_layer(
                        self.intermediate_queue,
                        y,
                        {"enable": False}
                    )

                # ===== ONLY EDGE =====
                elif mode == "only_edge":

                    y = []
                    x, y = inference(model, input_image, y, 0)

                    results = postprocess_yolo(x, conf_thres=0.01, iou_thres=0.5)
                    self._update_map(results, batch_id, batch_size)

                # ===== SPLIT INFERENCE =====
                else:

                    y = []
                    x, y = inference(model, input_image, y, 0)
                    y[-1] = x

                    y = {
                        "data": y,
                        "width": width,
                        "height": height,
                        "edge_start_time": batch_start
                    }

                    self.send_next_layer(
                        self.intermediate_queue,y,compress
                    )
                batch_end = time.perf_counter()
                latency_ms = (batch_end - batch_start) * 1000
                fps = batch_size / (batch_end - prev_batch_end) if prev_batch_end is not None else 0.0
                e2e_latency_ms = latency_ms if mode == "only_edge" else 0.0
                ram_mb = self.get_ram_mb()
                msg_size = self.size_message if self.size_message is not None else 0

                self.write_metrics(
                    mode=mode,
                    role="edge_sender" if mode == "only_cloud" else "edge",
                    best_cut="N/A" if splits is None else splits - 1,
                    batch_id=batch_id,
                    batch_size=batch_size,
                    latency_ms=latency_ms,
                    fps=fps,
                    ram_mb=ram_mb,
                    message_size_bytes=msg_size,
                    e2e_latency_ms=e2e_latency_ms
                )

                batch_id += 1
                prev_batch_end = batch_end

                input_image = []
                orig_images = []
                pbar.update(batch_size)
            else:
                continue
        print(f'size message: {self.size_message} bytes.')
        cap.release()
        pbar.close()

        notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                       "message": "Finish training!"}

        self.send_to_server(notify_data)

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


    def last_layer(self, model, batch_size, splits, logger, compress, mode="split"):
        model.eval()
        model.to(self.device)

        self.channel.basic_qos(prefetch_count=10)
        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        batch_id = 0
        prev_batch_end = None
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=self.intermediate_queue, auto_ack=True)
            if method_frame and body:
                batch_start = time.perf_counter()
                received_message_size = len(body)
                received_data = pickle.loads(body)
                y = received_data["data"]
                edge_start_time = y.get("edge_start_time", batch_start)

                # ===== ONLY CLOUD =====
                if mode == "only_cloud":
                    input_tensor = y["data"]

                    if isinstance(input_tensor, list):
                        input_tensor = torch.stack(input_tensor)

                    input_tensor = input_tensor.to(self.device)

                    x, _ = inference(model, input_tensor, [], 0)
                # ===== SPLIT INFERENCE =====
                else:

                    if compress["enable"]:
                        y["data"] = Decoder(y["data"], y["shape"])

                        y["data"] = [
                            torch.from_numpy(t) if t is not None else None
                            for t in y["data"]
                        ]

                    y["data"] = [
                        t.to(self.device) if t is not None else None
                        for t in y["data"]
                    ]

                    list_output = y["data"]

                    x = list_output[-1]
                    x, _ = inference(model,x,list_output,splits)
                results = postprocess_yolo(x, conf_thres=0.01, iou_thres=0.5)
                self._update_map(results, batch_id, batch_size)

                batch_end = time.perf_counter()
                latency_ms = (batch_end - batch_start) * 1000
                fps = batch_size / (batch_end - prev_batch_end) if prev_batch_end is not None else 0.0
                e2e_latency_ms = (batch_end - edge_start_time) * 1000
                ram_mb = self.get_ram_mb()

                self.write_metrics(
                    mode=mode,
                    role="cloud",
                    best_cut="N/A" if splits is None else splits - 1,
                    batch_id=batch_id,
                    batch_size=batch_size,
                    latency_ms=latency_ms,
                    fps=fps,
                    ram_mb=ram_mb,
                    message_size_bytes=received_message_size,
                    e2e_latency_ms=e2e_latency_ms
                )

                batch_id += 1
                prev_batch_end = batch_end

                pbar.update(batch_size)

            else:
                broadcast_queue_name = f'reply_{self.client_id}'
                method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                if body:
                    received_data = pickle.loads(body)
                    Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                    if received_data["action"] == "STOP":
                        Log.print_with_color("[>>>] Finish!", "red")
                        break
                else:
                    time.sleep(0.5)

        cv2.destroyAllWindows()
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

        n_rows = max(len(edge_rows), len(cloud_rows))
        fieldnames = [
            "batch_id", "batch_size", "best_cut",
            "edge_device", "edge_latency_ms", "edge_fps", "edge_ram_mb", "edge_message_size_bytes",
            "cloud_device", "cloud_latency_ms", "cloud_fps", "cloud_ram_mb", "cloud_message_size_bytes",
            "e2e_latency_ms",
        ]

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n_rows):
                e = edge_rows[i] if i < len(edge_rows) else {}
                c = cloud_rows[i] if i < len(cloud_rows) else {}
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
                    "cloud_latency_ms":        c.get("latency_ms", ""),
                    "cloud_fps":               c.get("fps", ""),
                    "cloud_ram_mb":            c.get("ram_mb", ""),
                    "cloud_message_size_bytes":c.get("message_size_bytes", ""),
                    "e2e_latency_ms":          c.get("e2e_latency_ms") or e.get("e2e_latency_ms", ""),
                })

        for fpath in _glob.glob("metrics_raw_*.csv"):
            os.remove(fpath)
        os.remove(lock_path)

        def avg(rows, key):
            vals = [float(r[key]) for r in rows if r.get(key)]
            return round(sum(vals) / len(vals), 3) if vals else None

        def mb(val):
            return round(val / 1024 / 1024, 3) if val is not None else "N/A"

        first_row = (edge_rows or cloud_rows)[0] if (edge_rows or cloud_rows) else {}
        cut = first_row.get("best_cut", "N/A")
        print("=" * 50)
        print(f"  SUMMARY  |  batches={n_rows}  cut={cut}")
        print("=" * 50)
        all_rows = cloud_rows if cloud_rows else edge_rows
        print(f"  [EDGE]  latency={avg(edge_rows,'latency_ms')} ms  fps={avg(edge_rows,'fps')}  ram={avg(edge_rows,'ram_mb')} MB  msg={mb(avg(edge_rows,'message_size_bytes'))} MB")
        print(f"  [CLOUD] latency={avg(cloud_rows,'latency_ms')} ms  fps={avg(cloud_rows,'fps')}  ram={avg(cloud_rows,'ram_mb')} MB  msg={mb(avg(cloud_rows,'message_size_bytes'))} MB")
        print(f"  [E2E]   latency={avg(all_rows,'e2e_latency_ms')} ms")
        print("=" * 50)
        Log.print_with_color(f"Saved metrics_pivoted.csv ({n_rows} batches)", "green")
        self._print_map()

    def inference_func(self, model, data, num_layers, splits, batch_size, logger, compress, mode="split", queue_name="intermediate_queue"):
        if queue_name != self.intermediate_queue:
            self.intermediate_queue = queue_name
            self.channel.queue_declare(self.intermediate_queue, durable=False)

        if self.layer_id == 1:
            self.first_layer(model, data, batch_size, splits, logger, compress, mode)
            if mode == "only_edge":
                self._pivot_and_save()
        elif self.layer_id == num_layers:
            self.last_layer(model, batch_size, splits, logger, compress, mode)
            self._pivot_and_save()
        else:
            self.middle_layer(model)
