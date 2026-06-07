import argparse
import glob as _glob
import json
import os
import threading
import traceback

import pika
import torch
import yaml

import src.Log
from src.RpcClient import RpcClient
from src.Scheduler import Scheduler


SETUP_PTH = "setup.json"

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--device', type=str, required=False, help='Device of client')
parser.add_argument('--name', type=str, required=False, default=None, help='Name of this machine (e.g. machine-2, device-1)')

args = parser.parse_args()

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

with open(SETUP_PTH, 'r', encoding='utf-8') as json_file:
    setup = json.load(json_file)

address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]

device = "cpu"
print("Using device: CPU")

configured_clients = config["server"]["clients"]
if args.layer_id < 1 or args.layer_id > len(configured_clients):
    raise ValueError(
        f"Invalid layer_id={args.layer_id}. Expected 1..{len(configured_clients)}"
    )

NUM_THREADS = 3

streams = [None] * NUM_THREADS

logger = src.Log.Logger("./app.log", config['debug-mode'])
logger.log_info("Application start.")

thread_errors = []
thread_errors_lock = threading.Lock()


def create_rabbit_channel():
    credentials = pika.PlainCredentials(username, password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=address,
            port=5672,
            virtual_host=f"{virtual_host}",
            credentials=credentials,
            heartbeat=3600,
            blocked_connection_timeout=600
        )
    )
    return connection, connection.channel()


def get_thread_setup(thread_id):
    setup_key = f"md{thread_id}"
    if setup_key not in setup:
        raise KeyError(f"Missing '{setup_key}' in {SETUP_PTH}")
    if "uuid" not in setup[setup_key]:
        raise KeyError(f"Missing 'uuid' for '{setup_key}' in {SETUP_PTH}")
    return setup_key, setup[setup_key]


def execute_client(thread_id, _):
    connection = None
    try:
        setup_key, thread_setup = get_thread_setup(thread_id)
        setup_uuid = str(thread_setup["uuid"])
        client_id = setup_uuid
        reply_queue_name = f"reply_{client_id}"

        connection, channel = create_rabbit_channel()
        channel.queue_declare(queue=reply_queue_name, durable=False)

        data = {
            "action": "REGISTER",
            "client_id": client_id,
            "layer_id": args.layer_id,
            "message": f"Hello from Thread {thread_id}",
            "layer_times": None,
            "bandwidth_mb_s": None,
            "client_name": args.name,
            "setup_uuid": setup_uuid
        }

        if "partition_point" in thread_setup:
            data["partition_point"] = thread_setup["partition_point"]

        scheduler = Scheduler(
            client_id,
            args.layer_id,
            channel,
            device
        )
        logger.log_debug(
            f"thread={thread_id}, setup_key={setup_key}, setup_uuid={setup_uuid}, "
            f"client_id={client_id}, "
            f"reply_queue={reply_queue_name}, stage={args.layer_id}, "
            f"channel={channel}, device={device}"
        )

        client = RpcClient(
            client_id,
            args.layer_id,
            channel,
            logger,
            scheduler.inference_func,
            device
        )

        def run_client():
            channel.queue_declare(queue=reply_queue_name, durable=False)
            client.send_to_server(data)
            client.wait_response()


        run_client()

        print(f"[THREAD {thread_id}] Finished")
    except Exception as e:
        with thread_errors_lock:
            thread_errors.append((thread_id, e))
        logger.log_error(
            f"[THREAD {thread_id}] Failed:\n{traceback.format_exc()}"
        )
    finally:
        if connection is not None and connection.is_open:
            connection.close()


if __name__ == "__main__":
    for _f in _glob.glob("metrics_raw_*.csv") + ["metrics_pivoted.csv", "metrics_pivot.lock"]:
        try:
            os.remove(_f)
        except (FileNotFoundError, PermissionError):
            pass

    src.Log.print_with_color(
        f"[>>>] Starting {NUM_THREADS} client thread(s) for layer_id={args.layer_id}...",
        "red"
    )

    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(
            target=execute_client,
            args=(i, None),
            name=f"client-thread-{i}"
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if thread_errors:
        for thread_id, error in thread_errors:
            print(f"[THREAD {thread_id}] Failed: {error}")
        raise SystemExit(1)

    print("All threads completed.")
