import os
import pika
import uuid
import argparse
import yaml

import torch

import src.Log
from src.RpcClient import RpcClient
from src.Scheduler import Scheduler

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--device', type=str, required=False, help='Device of client')
parser.add_argument('--name', type=str, required=False, default=None, help='Name of this machine (e.g. machine-2, device-1)')

args = parser.parse_args()

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]

device = None

if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print(f"Using device: CPU")
else:
    device = args.device
    print(f"Using device: {device}")

logger = src.Log.Logger(f"./app.log" , config['debug-mode'])
logger.log_info(f"Application start.")

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
channel = connection.channel()

if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")

    layer_times = None
    model_name = config["server"]["model"]
    clustering_cfg = config.get("clustering", {})
    use_real_profile = clustering_cfg.get("profile_source", "auto") != "simulated"
    if clustering_cfg.get("enable", False) and use_real_profile and os.path.exists(f"{model_name}.pt"):
        try:
            from src.Profiler import profile_or_load
            ckpt = torch.load(f"{model_name}.pt", map_location=device, weights_only=False)
            model_obj = ckpt["model"].float().eval().to(device)
            layer_times = profile_or_load(
                model_name, model_obj, device,
                batch_size=config["server"]["batch-size"]
            ).tolist()
            del model_obj, ckpt
        except Exception as e:
            src.Log.print_with_color(f"[Profile] Warning: {e}", "yellow")
    elif not use_real_profile:
        src.Log.print_with_color("[Profile] Skipped (profile_source=simulated)", "yellow")

    bandwidth_mb_s = None
    if clustering_cfg.get("enable", False):
        if clustering_cfg.get("measure_bandwidth", True):
            try:
                from src.Profiler import measure_bandwidth
                bandwidth_mb_s = measure_bandwidth(channel, str(client_id))
            except Exception as e:
                src.Log.print_with_color(f"[Bandwidth] Warning: {e}", "yellow")
                if not channel.is_open:
                    channel = connection.channel()
        else:
            bandwidth_mb_s = float(clustering_cfg.get("network_rate_mb_s", 100.0))
            src.Log.print_with_color(f"[Bandwidth] Using fixed rate from config: {bandwidth_mb_s} MB/s", "cyan")

    data = {"action": "REGISTER", "client_id": client_id, "layer_id": args.layer_id,
            "message": "Hello from Client!", "layer_times": layer_times,
            "bandwidth_mb_s": bandwidth_mb_s, "client_name": args.name}
    scheduler = Scheduler(client_id, args.layer_id, channel, device)
    logger.log_debug(f"client_id : {client_id} , stage {args.layer_id} , "
                     f"channel {channel} , device {device}")
    client = RpcClient(client_id, args.layer_id, channel ,logger ,scheduler.inference_func, device)
    client.send_to_server(data)
    client.wait_response()
