import argparse
import ctypes
import gc
import json
import os
import threading
import time

import psutil
import torch

from circuit_tracer.replacement_model.replacement_model_transformerlens import TransformerLensReplacementModel
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

DESCRIPTION = ('Commit charge (private bytes) of the process at each stage of loading a circuit-tracer replacement model '
               'as filter_prediction.py does (ReplacementModel.from_pretrained with the transformerlens backend, split '
               'into its two calls), with the peak between stages from a sampling thread, and the GPU memory torch holds; '
               'then after a garbage collection and after returning freed heap memory to the system, and the private '
               'bytes every 2 s over the whole load.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device-context', action='store_true',
                        help='load the model inside torch.device(cuda), so that new tensors are created on the GPU')
    parser.add_argument('--no-fold-value-biases', action='store_true',
                        help='skip TransformerLens value-bias folding, which clones the float32 state dict')
    args = parser.parse_args()
    proc = psutil.Process(os.getpid())
    peak = [0]
    series = []
    stop = threading.Event()
    t0 = time.time()

    def sample():
        k = 0
        while not stop.is_set():
            p = proc.memory_info().private
            peak[0] = max(peak[0], p)
            if k % 4 == 0:
                series.append((round(time.time() - t0, 1), round(p / 2**30, 2)))
            k += 1
            time.sleep(.5)

    threading.Thread(target=sample, daemon=True).start()
    rows = []

    def mark(stage):
        m = proc.memory_info()
        rows.append(dict(stage=stage, elapsed=round(time.time() - t0, 1), private_gib=round(m.private / 2**30, 2),
                         peak_private_gib=round(max(peak[0], m.private) / 2**30, 2), resident_gib=round(m.rss / 2**30, 2),
                         gpu_gib=round(torch.cuda.memory_reserved() / 2**30, 2)))
        print(json.dumps(rows[-1]), flush=True)
        peak[0] = 0

    dev = torch.device('cuda')
    mark('start')
    torch.zeros(1, device=dev)
    mark('cuda')
    transcoders, _ = load_transcoder_from_hub(args.dictionary, device=dev, dtype=torch.bfloat16, lazy_encoder=False,
                                              lazy_decoder=True)
    mark('transcoders')
    extra = dict(fold_value_biases=False) if args.no_fold_value_biases else {}
    with torch.device(dev if args.device_context else 'cpu'):
        model = TransformerLensReplacementModel.from_pretrained_and_transcoders(args.model, transcoders, device=dev,
                                                                               dtype=torch.bfloat16, **extra)
    mark('model')
    gc.collect()
    mark('gc')
    # Return freed heap memory of the C runtime and of the process heap to the system.
    ctypes.cdll.ucrtbase._heapmin()
    mark('heapmin')
    kernel32 = ctypes.windll.kernel32
    kernel32.GetProcessHeap.restype = ctypes.c_void_p
    kernel32.HeapCompact.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.HeapCompact(kernel32.GetProcessHeap(), 0)
    mark('heapcompact')
    stop.set()
    with open(args.out, 'w') as f:
        json.dump(dict(description=DESCRIPTION, model=args.model, dictionary=args.dictionary,
                       device_context=args.device_context, no_fold_value_biases=args.no_fold_value_biases, rows=rows,
                       series=series), f, indent=1)
    del model


if __name__ == '__main__':
    main()
