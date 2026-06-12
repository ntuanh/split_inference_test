import numpy as np
import os
import sys
import base64
import pika
import pickle
import threading
import time
import requests
from requests.auth import HTTPBasicAuth
import src.Model
import src.Log
from src.Utils import RAM_GUARD_PAUSE_ACTION, RAM_GUARD_RESUME_ACTION
from ultralytics import YOLO

from src.Clustering import (
    ManualExperimentConfig,
    DeterministicSimilarityAssignmentSolver,
    run_manual_hungarian_case,
    print_result,
    get_cut_data_sizes,
    get_raw_input_mb,
)


class Server:
    def __init__(self, config):
        self.config = config
        self.address = config["rabbit"]["address"]
        self.username = config["rabbit"]["username"]
        self.password = config["rabbit"]["password"]
        self.virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]
        self.batch_size = config["server"]["batch-size"]

        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.address,
                port=5672,
                virtual_host=f"{self.virtual_host}",
                credentials=credentials,
                heartbeat=3600,
                blocked_connection_timeout=600
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue', durable=False)
        self.channel.queue_purge(queue='rpc_queue')

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []
        self.registered_ids = set()
        self.notified = False
        self.count_clients = 0
        self.unique_client_ids = {}     # {setup_uuid/client_id: partition_point} for tail clients
        self.client_setup_ids = {}      # {actual client_id: setup_uuid/client_id}
        self.client_assignments = {}    # {client_id: {"splits": int, "queue_name": str}}
        self.client_profile_data = {}   # {client_id_str: np.array of per-layer times}
        self.client_bandwidth_data = {} # {client_id_str: float MB/s}
        self.client_name_data = {}      # {client_id_str: str name}
        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        # RAM guard: watches the broker's own memory usage (via the management
        # HTTP API) and tells layer-1 (edge) clients to pause/resume publishing
        # to the intermediate queues, so the broker never gets close to OOM.
        ram_guard_cfg = config.get("ram-guard", {})
        self.ram_guard_enable = ram_guard_cfg.get("enable", True)
        self.ram_pause_ratio = float(ram_guard_cfg.get("pause_ratio", 0.70))
        self.ram_resume_ratio = float(ram_guard_cfg.get("resume_ratio", 0.50))
        self.ram_poll_interval_s = float(ram_guard_cfg.get("poll_interval_s", 2.0))
        self._ram_guard_stop = threading.Event()
        self._ram_guard_paused = False

        self.data = config["data"]
        self.compress = config["compress"]

        log_path = config["log-path"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", config["debug-mode"])
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")
        src.Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")

    def _get_mode(self):
        exp = self.config.get("experiment", {})
        if exp.get("enable", False):
            return exp.get("mode", "split")
        return "split"

    def _queue_name_for_setup_uuid(self, setup_uuid):
        safe_id = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in str(setup_uuid)
        )
        return f"intermediate_queue_{safe_id}"

    def _assign_one_to_one_queues(self, default_splits):
        layers_by_setup_uuid = {}
        for client_id, layer_id in self.list_clients:
            setup_uuid = self.client_setup_ids.get(client_id, client_id)
            layers_by_setup_uuid.setdefault(setup_uuid, set()).add(layer_id)

        logged = set()
        for client_id, layer_id in self.list_clients:
            setup_uuid = self.client_setup_ids.get(client_id, client_id)
            queue_name = self._queue_name_for_setup_uuid(setup_uuid)
            assigned_splits = self.unique_client_ids.get(setup_uuid, default_splits)
            assignment_key = (client_id, layer_id)
            assignment = dict(
                self.client_assignments.get(
                    assignment_key,
                    self.client_assignments.get(client_id, {})
                )
            )
            assignment["queue_name"] = queue_name
            if assigned_splits is not None:
                assignment["splits"] = assigned_splits
            self.client_assignments[assignment_key] = assignment

            if setup_uuid not in logged:
                layers = sorted(layers_by_setup_uuid.get(setup_uuid, []))
                src.Log.print_with_color(
                    f"[Pairing] uuid={setup_uuid} layers={layers} "
                    f"queue={queue_name} splits={assignment.get('splits')}",
                    "green"
                )
                logged.add(setup_uuid)

    def on_request(self, ch, method, _, body):
        message = pickle.loads(body)
        action = message["action"]

        if action == "REGISTER":
            client_id = str(message["client_id"])
            layer_id = int(message["layer_id"])
            setup_uuid = str(message.get("setup_uuid", client_id))

            src.Log.print_with_color(f"[<<<] Received REGISTER from client {client_id} layer={layer_id}", "blue")

            if layer_id < 1 or layer_id > len(self.register_clients):
                src.Log.print_with_color(
                    f"[!] Ignored client with unexpected layer_id={layer_id} (expected 1..{len(self.register_clients)})", "red")
                return

            registration_key = (client_id, layer_id)
            if registration_key in self.registered_ids:
                src.Log.print_with_color(
                    f"[!] Duplicate REGISTER from {client_id} layer={layer_id}, ignored.", "yellow")
                return

            self.registered_ids.add(registration_key)
            self.list_clients.append(registration_key)
            self.client_setup_ids[client_id] = setup_uuid

            if layer_id == len(self.total_clients) and "partition_point" in message:
                partition_point = message["partition_point"]
                if setup_uuid in self.unique_client_ids:
                    src.Log.print_with_color(
                        f"[!] Overlap UUID at layer {layer_id}: {setup_uuid}", "yellow")
                else:
                    self.unique_client_ids[setup_uuid] = partition_point

            layer_times = message.get("layer_times", None)
            if layer_times is not None:
                self.client_profile_data[client_id] = np.array(layer_times, dtype=float)
                src.Log.print_with_color(
                    f"[Profile] Stored profiling data from client {client_id} "
                    f"({len(layer_times)} layers, total={sum(layer_times)*1000:.1f} ms)", "cyan")

            bandwidth_mb_s = message.get("bandwidth_mb_s", None)
            if bandwidth_mb_s is not None:
                self.client_bandwidth_data[client_id] = float(bandwidth_mb_s)
                src.Log.print_with_color(
                    f"[Bandwidth] Stored bandwidth from client {client_id}: {bandwidth_mb_s:.1f} MB/s", "cyan")

            client_name = message.get("client_name", None)
            if client_name:
                self.client_name_data[client_id] = client_name

            self.register_clients[layer_id - 1] += 1

            if self.register_clients == self.total_clients and not self.notified:
                self.notified = True
                if self.unique_client_ids:
                    src.Log.print_with_color(
                        f"[Log] unique_client_ids: {self.unique_client_ids}", "green")
                src.Log.print_with_color("All clients connected. Sending notifications.", "green")
                self.notify_clients()

        elif action == "BW_TEST":
            client_id = message["client_id"]
            self.send_to_response(str(client_id), pickle.dumps({"action": "BW_ACK"}))

        elif action == "STOP":
            self.count_clients += 1
            print(f"[ DEBUG ] total clients {self.total_clients[0]}")
            print(f"[ DEBUG ] counted clients {self.count_clients}")
            # only_edge: cloud clients skip inference, so only edge clients send STOP
            mode = self._get_mode()
            expected = self.total_clients[0] if mode == "only_edge" else self.total_clients[0] * 2
            if self.count_clients == expected:
                # print(f"[ DEBUG ] total clients {self.total_clients[0]}")
                # print(f"[ DEBUG ] counted clients {self.count_clients}")
                self.logger.log_info("Stop Inference !!!")
                self.notify_clients(start=False)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                self.channel.stop_consuming()
                return

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_response(self, client_id, message):
        reply_queue_name = f"reply_{client_id}"
        self.reply_channel.queue_declare(queue=reply_queue_name, durable=False)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(exchange='', routing_key=reply_queue_name, body=message)

    def _wait_reply_queues_drained(self, client_ids, timeout_s=10.0):
        pending = {str(client_id) for client_id in client_ids}
        deadline = time.perf_counter() + timeout_s

        while pending and time.perf_counter() < deadline:
            for client_id in list(pending):
                queue_name = f"reply_{client_id}"
                try:
                    result = self.reply_channel.queue_declare(queue=queue_name, passive=True)
                except pika.exceptions.ChannelClosedByBroker:
                    self.reply_channel = self.connection.channel()
                    pending.remove(client_id)
                    continue

                if result.method.message_count == 0:
                    pending.remove(client_id)

            if pending:
                time.sleep(0.2)

        if pending:
            src.Log.print_with_color(
                f"[Cleanup] Reply queues still have STOP messages or no reader: {sorted(pending)}",
                "yellow"
            )

    def _broker_memory_ratio(self):
        """Query the RabbitMQ management API for this node's mem_used/mem_limit
        ratio. Returns None if the API is unreachable or fields are missing."""
        try:
            url = f"http://{self.address}:15672/api/nodes"
            response = requests.get(
                url, auth=HTTPBasicAuth(self.username, self.password), timeout=5
            )
            response.raise_for_status()
            nodes = response.json()
            if not nodes:
                return None
            mem_used = nodes[0].get("mem_used")
            mem_limit = nodes[0].get("mem_limit")
            if not mem_used or not mem_limit:
                return None
            return mem_used / mem_limit
        except Exception:
            return None

    def _broadcast_send_control(self, channel, action):
        # Broadcast to every client: edges (layer 1) pause/resume publishing,
        # clouds (last layer) use it to know an empty queue is expected and
        # must not trigger their DONE-lost fallback STOP.
        for client_id, layer_id in list(self.list_clients):
            reply_queue_name = f"reply_{client_id}"
            channel.queue_declare(queue=reply_queue_name, durable=False)
            channel.basic_publish(
                exchange='', routing_key=reply_queue_name,
                body=pickle.dumps({"action": action})
            )
            src.Log.print_with_color(
                f"[RAM-Guard] >>> {action} -> client {client_id} (layer {layer_id})", "yellow")

    def _ram_guard_loop(self):
        """Background thread: polls broker memory usage and tells edge (layer-1)
        clients to pause/resume publishing to the intermediate queues, with
        hysteresis between pause_ratio and resume_ratio so it doesn't flap.
        Uses its own connection — pika's BlockingConnection is not thread-safe,
        so it can't share self.connection/self.channel with the consumer loop."""
        credentials = pika.PlainCredentials(self.username, self.password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.address,
                port=5672,
                virtual_host=f"{self.virtual_host}",
                credentials=credentials,
                heartbeat=3600,
                blocked_connection_timeout=600
            )
        )
        channel = connection.channel()
        try:
            while not self._ram_guard_stop.is_set():
                ratio = self._broker_memory_ratio()
                if ratio is not None:
                    if not self._ram_guard_paused and ratio >= self.ram_pause_ratio:
                        self._ram_guard_paused = True
                        self._broadcast_send_control(channel, RAM_GUARD_PAUSE_ACTION)
                        src.Log.print_with_color(
                            f"[RAM-Guard] Broker memory at {ratio:.0%} (>= {self.ram_pause_ratio:.0%}) "
                            f"— told edge clients to pause sending.", "yellow")
                    elif self._ram_guard_paused and ratio <= self.ram_resume_ratio:
                        self._ram_guard_paused = False
                        self._broadcast_send_control(channel, RAM_GUARD_RESUME_ACTION)
                        src.Log.print_with_color(
                            f"[RAM-Guard] Broker memory at {ratio:.0%} (<= {self.ram_resume_ratio:.0%}) "
                            f"— told edge clients to resume sending.", "green")
                self._ram_guard_stop.wait(self.ram_poll_interval_s)
        finally:
            connection.close()

    def start(self):
        guard_thread = None
        if self.ram_guard_enable:
            guard_thread = threading.Thread(target=self._ram_guard_loop, name="ram-guard", daemon=True)
            guard_thread.start()
        try:
            self.channel.start_consuming()
        finally:
            self._ram_guard_stop.set()
        self.connection.close()
        sys.exit(0)

    def _run_hungarian(self):
        cfg = self.config.get("clustering", {})
        network_rate = float(cfg.get("network_rate_mb_s", 1000.0))
        max_clusters = cfg.get("max_clusters", 1)

        # Dùng real profiling data nếu tất cả client đã gửi
        edge_times_list = [
            self.client_profile_data[str(cid)]
            for cid, lid in self.list_clients
            if lid == 1 and str(cid) in self.client_profile_data
        ]
        cloud_times_list = [
            self.client_profile_data[str(cid)]
            for cid, lid in self.list_clients
            if lid == len(self.total_clients) and str(cid) in self.client_profile_data
        ]
        n_edge = sum(1 for _, lid in self.list_clients if lid == 1)
        n_cloud = sum(1 for _, lid in self.list_clients if lid == len(self.total_clients))
        profile_source = cfg.get("profile_source", "auto")
        has_real = (len(edge_times_list) == n_edge and len(cloud_times_list) == n_cloud
                    and n_edge > 0 and n_cloud > 0)

        if profile_source == "real" and not has_real:
            raise RuntimeError("[Clustering] profile_source=real nhưng chưa có đủ profiling từ clients")

        use_real = has_real if profile_source == "auto" else (profile_source == "real")

        if use_real:
            src.Log.print_with_color(
                f"[Clustering] Using REAL profiles ({n_edge} edge, {n_cloud} cloud) [profile_source={profile_source}]", "cyan")
            N = len(edge_times_list)
            M = len(cloud_times_list)
            edge_clients = [cid for cid, lid in self.list_clients
                            if lid == 1 and str(cid) in self.client_profile_data]
            rates_matrix = np.array([
                [self.client_bandwidth_data.get(str(cid), network_rate)] * M
                for cid in edge_clients
            ]) if edge_clients else np.full((N, M), network_rate)
            cloud_clients = [cid for cid, lid in self.list_clients
                             if lid == len(self.total_clients) and str(cid) in self.client_profile_data]
            solver = DeterministicSimilarityAssignmentSolver(
                client_layer_times=np.vstack(edge_times_list),
                server_layer_times=np.vstack(cloud_times_list),
                cut_data_sizes=get_cut_data_sizes(self.model_name, self.batch_size),
                input_data_size=get_raw_input_mb(self.batch_size),
                network_rates=rates_matrix,
            )
            solver.client_type_names = [
                self.client_name_data.get(str(cid), f"edge_{str(cid)[:8]}")
                for cid in edge_clients
            ]
            solver.cloud_type_names = [
                self.client_name_data.get(str(cid), f"cloud_{str(cid)[:8]}")
                for cid in cloud_clients
            ]
            result = solver.solve_best_over_k("hungarian", max_clusters=max_clusters)["best_result"]
            print_result(result, solver, title="HUNGARIAN MATCHING RESULT (real profiles)")
        else:
            src.Log.print_with_color(
                f"[Clustering] Using SIMULATED profiles (DEVICE_A/B/C hardcoded) [profile_source={profile_source}]", "yellow")
            manual_cfg = ManualExperimentConfig(
                num_A=cfg.get("num_A", 1),
                num_B=cfg.get("num_B", 0),
                num_C=cfg.get("num_C", 0),
                num_cloud=cfg.get("num_cloud", 1),
                network_rate_mb_s=network_rate,
                max_clusters=max_clusters,
                exact_max_k=max_clusters,
                model_name=self.model_name,
                batch_size=self.batch_size,
                input_data_mb=get_raw_input_mb(self.batch_size),
            )
            results = run_manual_hungarian_case(manual_cfg)
            solver = results["solver"]
            result = results["hungarian"]

        return solver, result

    def notify_clients(self, start=True):
        if start:
            default_splits = {"a": 16, "b": 11, "c": 17, "d": 23}

            if os.path.exists(f"{self.model_name}.pt"):
                src.Log.print_with_color(f"Exist {self.model_name}.pt", "green")
            else:
                src.Log.print_with_color(f"Download {self.model_name}", "yellow")
                _ = YOLO(f"{self.model_name}.pt")

            mode = self._get_mode()
            splits = None

            if mode in ["only_edge", "only_cloud"]:
                src.Log.print_with_color(f"[Benchmark] mode={mode}, skip split selection", "yellow")

            else:
                clustering_cfg = self.config.get("clustering", {})
                use_hungarian = clustering_cfg.get("enable", False)

                if use_hungarian:
                    try:
                        _, h = self._run_hungarian()
                        edge_labels  = h.edge_labels
                        cloud_labels = h.cloud_labels
                        matching     = h.matching
                        best_cuts    = h.best_cuts
                        K            = h.num_clusters
                        inv_matching = {int(matching[k]): k for k in range(K)}

                        edge_ord  = [(cid, lid) for cid, lid in self.list_clients if lid == 1]
                        cloud_ord = [(cid, lid) for cid, lid in self.list_clients if lid == len(self.total_clients)]

                        self.client_assignments = {}
                        for i, (cid, _) in enumerate(edge_ord):
                            k = int(edge_labels[i]) if i < len(edge_labels) else 0
                            self.client_assignments[cid] = {
                                "splits":     int(best_cuts[k]) + 1,
                                "queue_name": f"intermediate_queue_{k}",
                            }
                        for j, (cid, _) in enumerate(cloud_ord):
                            l = int(cloud_labels[j]) if j < len(cloud_labels) else 0
                            k = inv_matching.get(l, 0)
                            self.client_assignments[cid] = {
                                "splits":     int(best_cuts[k]) + 1,
                                "queue_name": f"intermediate_queue_{k}",
                            }

                        splits = int(best_cuts[0]) + 1 if len(best_cuts) > 0 else None
                        src.Log.print_with_color(
                            f"[Clustering] K={K}  best_cuts={best_cuts.tolist()}", "green")

                    except Exception as e:
                        raise RuntimeError(f"Hungarian clustering failed: {e}")

                elif self.cut_layer in default_splits:
                    splits = default_splits[self.cut_layer]
                    src.Log.print_with_color(
                        f"[Benchmark] Fixed split '{self.cut_layer}' -> splits={splits}", "yellow")
                else:
                    raise ValueError(f"Invalid cut-layer: '{self.cut_layer}'. Use a/b/c/d or set clustering.enable: True")

            self._assign_one_to_one_queues(splits)

            file_path = f"{self.model_name}.pt"
            if not os.path.exists(file_path):
                src.Log.print_with_color(f"{self.model_name}.pt does not exist.", "yellow")
                self.connection.close()
                sys.exit(1)

            with open(file_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')

            # Deduplicate list_clients trong trường hợp pika callback reentrant
            seen_notify = set()
            clients_to_notify = []
            for entry in self.list_clients:
                if entry not in seen_notify:
                    seen_notify.add(entry)
                    clients_to_notify.append(entry)

            src.Log.print_with_color(
                f"Sending model {self.model_name} to {len(clients_to_notify)} clients "
                f"(list_clients={len(self.list_clients)}).", "green")

            for (client_id, layer_id) in clients_to_notify:
                assignment = self.client_assignments.get(
                    (client_id, layer_id),
                    self.client_assignments.get(client_id, {})
                )
                setup_uuid = self.client_setup_ids.get(client_id, client_id)
                assigned_splits = assignment.get(
                    "splits",
                    self.unique_client_ids.get(setup_uuid, splits)
                )
                response = {
                    "action":     "START",
                    "message":    "Server accept the connection",
                    "model":      encoded,
                    "splits":     assigned_splits,
                    "queue_name": assignment.get("queue_name", "intermediate_queue"),
                    "batch_size": self.batch_size,
                    "num_layers": len(self.total_clients),
                    "model_name": self.model_name,
                    "data":       self.data,
                    "compress":   self.compress,
                    "mode":       self._get_mode(),
                }
                self.send_to_response(client_id, pickle.dumps(response))
        else:
            response = {"action": "STOP", "message": "Stop inference !!!"}
            for (client_id, layer_id) in self.list_clients:
                self.send_to_response(client_id, pickle.dumps(response))
            self._wait_reply_queues_drained(client_id for client_id, _ in self.list_clients)
