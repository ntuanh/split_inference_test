import os
import time
import numpy as np
import torch
import src.Log as Log


def profile_or_load(model_name: str, model, device: str,
                    batch_size: int = 4, warmup: int = 5, runs: int = 30):
    """
    Profile per-layer inference time of model on device.
    Returns np.array of shape (n_layers,) — mean seconds per layer per batch.
    Cache saved as profile_{model_name}_{device}.npy next to client.py.
    """
    cache_path = f"profile_{model_name}_{device}.npy"

    if os.path.exists(cache_path):
        times = np.load(cache_path)
        Log.print_with_color(
            f"[Profile] Loaded cache '{cache_path}'  "
            f"({len(times)} layers, total={times.sum()*1000:.1f} ms/batch)",
            "green"
        )
        return times

    Log.print_with_color(
        f"[Profile] Profiling {model_name} on {device} "
        f"({warmup} warmup + {runs} runs) ...",
        "yellow"
    )

    layers = model.model
    n = len(layers)
    t0 = {}
    layer_times = [[] for _ in range(n)]
    hooks = []

    for i in range(n):
        def _pre(idx):
            def fn(m, inp):
                t0[idx] = time.perf_counter()
            return fn

        def _post(idx):
            def fn(m, inp, out):
                layer_times[idx].append(time.perf_counter() - t0[idx])
            return fn

        hooks.append(layers[i].register_forward_pre_hook(_pre(i)))
        hooks.append(layers[i].register_forward_hook(_post(i)))

    dummy = torch.randn(batch_size, 3, 640, 640).to(device)

    with torch.no_grad():
        for r in range(warmup + runs):
            model(dummy)

    for h in hooks:
        h.remove()

    avg = np.array([
        np.mean(layer_times[i][warmup:]) if len(layer_times[i]) > warmup else 0.0
        for i in range(n)
    ])

    np.save(cache_path, avg)
    Log.print_with_color(
        f"[Profile] Saved '{cache_path}'  "
        f"(total={avg.sum()*1000:.1f} ms/batch)",
        "green"
    )
    return avg
