import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random
import os
import logging
from Models import IF
from collections import deque

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def train(model, device, train_loader, criterion, optimizer, T):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for i, (images, labels) in enumerate((train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        if T > 0:
            outputs = model(images).mean(0)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


def val(model, test_loader, device, T):
    correct = 0
    total = 0
    counter = SynOpsCounter(model)
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((test_loader)):
            inputs = inputs.to(device)
            if T > 0:
                outputs = model(inputs).mean(0)
                counter.add_batch(inputs.shape[0])
            else:
                outputs = model(inputs)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        final_acc = 100 * correct / total
        if T > 0:
            total_synops = counter.compute_synops()
            print(f"Total spikes (all IF layers, all samples): {counter.total_spikes:.0f}")
            energy = total_synops * 0.9e-12
            print("Energy per sample (J):", energy)
            counter.remove()
    return final_acc


class SynOpsCounter:
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []

        self.layer_spikes = {}   # spikes summed over dataset
        self.layer_fanout = {}   # c_k per IF (outgoing synapses per neuron)
        self.total_samples = 0 
        self.total_spikes = 0.0 

        # stable names
        self._module_to_name = {m: n for n, m in model.named_modules()}

        # IFs that have fired, waiting to be assigned a consumer layer
        self._pending_ifs = deque()

        # hook IFs
        for m in model.modules():
            if isinstance(m, IF):
                if_name = self._module_to_name.get(m, f"IF@{id(m)}")
                self.layer_spikes[if_name] = 0.0
                self.layer_fanout[if_name] = None
                self.handles.append(m.register_forward_hook(self._if_hook(if_name)))

        # hook weighted layers (consumers)
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.handles.append(m.register_forward_hook(self._consumer_hook))

    def _if_hook(self, if_name):
        def hook(module, inp, out):
            # spike count
            spikes = (out > 0).float()
            spike_count = spikes.sum().item()
            self.layer_spikes[if_name] += spike_count
            self.total_spikes += spike_count

            # enqueue: the *next* Conv/Linear executed will be treated as its consumer
            self._pending_ifs.append(if_name)
        return hook

    def _fanout_from_consumer(self, consumer_module: nn.Module):
        if isinstance(consumer_module, nn.Conv2d):
            kx, ky = consumer_module.kernel_size

            # If the consumer is depthwise: each input-channel neuron connects to kx*ky outputs
            if consumer_module.groups == consumer_module.in_channels and consumer_module.out_channels == consumer_module.in_channels:
                return kx * ky  # typically 9

            # Normal / pointwise: each input neuron connects to out_channels per spatial location; with kx*ky neighborhoods
            # For pointwise k=1 => out_channels
            return consumer_module.out_channels * kx * ky

        if isinstance(consumer_module, nn.Linear):
            return consumer_module.out_features

        return None

    def _consumer_hook(self, consumer_module, inp, out):
        # assign this consumer to all pending IFs that don't have a fanout yet
        if not self._pending_ifs:
            return

        fanout = self._fanout_from_consumer(consumer_module)
        consumer_name = self._module_to_name.get(consumer_module, consumer_module.__class__.__name__)

        while self._pending_ifs:
            if_name = self._pending_ifs.popleft()
            if self.layer_fanout[if_name] is None:
                self.layer_fanout[if_name] = fanout
                # debug print
                if isinstance(consumer_module, nn.Conv2d):
                    print(
                        f"[bind-next] {if_name} => {consumer_name} "
                        f"(k={consumer_module.kernel_size}, groups={consumer_module.groups}, "
                        f"in={consumer_module.in_channels}, out={consumer_module.out_channels}) "
                        f"=> fanout={fanout}"
                    )
                elif isinstance(consumer_module, nn.Linear):
                    print(f"[bind-next] {if_name} => {consumer_name} (out={consumer_module.out_features}) => fanout={fanout}")

    def add_batch(self, batch_size: int):
        self.total_samples += int(batch_size)

    def compute_synops(self):
        total_synops = 0.0
        print("\nLayer statistics:")

        for name, spike_sum in self.layer_spikes.items():
            fanout = self.layer_fanout.get(name, None)

            avg_spikes = spike_sum / self.total_samples if self.total_samples > 0 else 0.0
            synops = (avg_spikes * fanout) if fanout is not None else 0.0 #SynOps per sample for this layer

            print(f"{name}: avg spikes/sample={avg_spikes:.2f}, fanout={fanout}, synops/sample={synops:.2f}")
            total_synops += synops

        print("\nTotals:")
        print(f"  total spikes (all IF layers, all samples): {self.total_spikes:.0f}")  # <--- added
        print(f"  total samples: {self.total_samples}")
        print(f"  avg spikes per sample (all IF layers): {(self.total_spikes / self.total_samples) if self.total_samples else 0.0:.2f}")
        print("\nTotal SynOps per sample:", total_synops)
        return total_synops

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []