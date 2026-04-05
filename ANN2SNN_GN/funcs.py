import numpy as np
import torch
from tqdm import tqdm
from utils import *
import torch.distributed as dist
import random
import os
from spikingjelly.activation_based import functional
from torch.utils.tensorboard import SummaryWriter   
from modules import CombinedNode, GN, GN_TTFS
from encoding_utils import decode_ttfs_output
from collections import deque

def seed_all(seed=42):
    #print(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def amp_train_ann(train_dataloader, test_dataloader, model, 
              epochs, device, loss_fn,lr=0.1,lr_min=1e-5,wd=5e-4 , save=None, parallel=False,
                rank=0):
    use_amp=True

    if rank==0:
        with open('./runs/'+save+'_log.txt','a') as log:
            log.write('lr={},epochs={},wd={}\n'.format(lr,epochs,wd))

    model.cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=lr, weight_decay=wd, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,eta_min=lr_min, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc=0.
    for epoch in range(epochs):
        model.train()
        if parallel:
            train_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0
        length = 0
        model.train()
        for img, label in tqdm(train_dataloader):
            img = img.to(device)
            label = label.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                out = model(img)
                loss = loss_fn(out, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            epoch_loss += loss.item()
            length += len(label)
        tmp_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device, rank)
        if parallel:
            dist.all_reduce(tmp_acc)
            tmp_acc/=dist.get_world_size()
        if rank == 0 and save != None and tmp_acc >= best_acc:
            checkpoint = {"model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
            torch.save(checkpoint, './saved_models/' + save + '.pth')
        if rank == 0:
            info='Epoch:{},Train_loss:{},Val_loss:{},Acc:{}'.format(epoch, epoch_loss/length,val_loss, tmp_acc.item())
            with open('./runs/'+save+'_log.txt','a') as log:
                log.write(info+'\n')
            if epoch % 10 == 0:
                print(model)
        best_acc = max(tmp_acc, best_acc)
        scheduler.step()

    return best_acc, model


def train_ann(train_dataloader, test_dataloader, model, 
              epochs, device, loss_fn,lr=0.1,lr_min=1e-6,wd=5e-4 , save=None, parallel=False,
                rank=0):
    # model.cuda(device)
    # writer = SummaryWriter('./runs/'+save)
    # mt=monitor.InputMonitor(model,SteppedReLU)
    # qcfs_vth={}
    # cnt=1
    # for name in mt.monitored_layers:
    #     qcfs=get_module_by_name(model,name)[1]
    #     #assert isinstance(qcfs,QCFS)
    #     qcfs_vth[str(cnt)+'+'+name]=qcfs.v_threshold
    #     #qcfs_p0[str(cnt)+'+'+name]=qcfs.p0
    #     cnt=cnt+1

    # mt.clear_recorded_data()
    # mt.remove_hooks()
    if parallel:
        wd=1e-4

    if rank==0:
        with open('./runs/'+save+'_log.txt','a') as log:
            log.write('lr={},epochs={},wd={}\n'.format(lr,epochs,wd))

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=lr, weight_decay=wd, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,eta_min=lr_min, T_max=epochs)


    best_acc=eval_ann(test_dataloader, model, loss_fn, device, rank)[0]
    if parallel:
        dist.all_reduce(best_acc)
        best_acc/=dist.get_world_size()
    if rank==0:
        print(best_acc)
    for epoch in tqdm(range(epochs)):
        model.train()
        if parallel:
            train_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0
        length = 0
        model.train()
        for img, label in tqdm(train_dataloader):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            length += len(label)

        tmp_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device, rank)
        if parallel:
            dist.all_reduce(tmp_acc)
            tmp_acc/=dist.get_world_size()
        if rank == 0 and save != None and tmp_acc >= best_acc:
            torch.save(model.state_dict(), './saved_models/' + save + '.pth')
        if rank == 0:
            info='Epoch:{},Train_loss:{},Val_loss:{},Acc:{},lr:{}'.format(epoch, epoch_loss/length,val_loss, tmp_acc.item(),scheduler.get_last_lr()[0])
            with open('./runs/'+save+'_log.txt','a') as log:
                log.write(info+'\n')
        best_acc = max(tmp_acc, best_acc)
        # print('Epoch:{},Train_loss:{},Val_loss:{},Acc:{}'.format(epoch, epoch_loss/length,val_loss, tmp_acc), flush=True)
        # print(f'lr={scheduler.get_last_lr()[0]}')
        # print('best_acc: ', best_acc)

        # writer.add_scalars('Acc',{'val_acc':tmp_acc,'best_acc':best_acc},epoch)
        # writer.add_scalars('Loss',{'train_loss':epoch_loss/length,'val_loss':val_loss},epoch)
        # writer.add_scalar('lr',scheduler.get_last_lr()[0],epoch)
        # writer.add_scalars('vth',qcfs_vth,epoch)
        scheduler.step()
        #print(module)
    # writer.close()
    return best_acc, model

def eval_snn(test_dataloader, model,loss_fn, device, sim_len=8, rank=0):
    tot = torch.zeros(sim_len).cuda()
    length = 0
    model = model.cuda()
    model.eval()
    counter = SynOpsCounter(model)

    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm((test_dataloader))):
            spikes = 0
            length += len(label)
            img = img.cuda()
            label = label.cuda()
            counter.add_batch(len(label))
            for t in range(sim_len):
                out = model(img)
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum()
            spikes/=sim_len
            loss = loss_fn(spikes, label)
            functional.reset_net(model)
        counter.compute_synops()
        total_synops = counter.compute_synops()
        print(f"Total spikes (all IF layers, all samples): {counter.total_spikes:.0f}")
        energy = total_synops * 0.9e-12
        print("Energy per sample (J):", energy)
        counter.remove()
    return (tot/length),loss.item()/length

def eval_snn_ttfs(test_dataloader, model, loss_fn, device, sim_len=8, rank=0):
    """Evaluate an SNN that uses TTFS (Time-to-First-Spike) temporal coding.

    For each output neuron the timestep of its *first* spike is recorded.
    Spike times are decoded back to analog confidence values using:
        decoded = 1 - spike_time / sim_len
    so that an early spike corresponds to high confidence.  Neurons that
    never fire receive a decoded value of 0.

    The cumulative accuracy is measured at every timestep so that the
    accuracy-vs-latency trade-off can be inspected (same format as
    eval_snn).

    Args:
        test_dataloader: DataLoader for the evaluation dataset.
        model: SNN model whose ReLU/QCFS layers have been replaced by
               GN_TTFS neurons.
        loss_fn: Loss function (applied to the decoded output at the end
                 of the simulation window).
        device: Device string or index (e.g. 'cuda').
        sim_len: Number of simulation timesteps T.
        rank: Process rank (for distributed evaluation; only rank 0 logs).

    Returns:
        Tuple (acc_per_step, final_loss) where acc_per_step is a tensor of
        length sim_len containing the fraction of correctly classified
        samples at each timestep.
    """
    tot = torch.zeros(sim_len, device=device)
    length = 0
    epoch_loss = 0.0
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for _, (img, label) in enumerate(tqdm(test_dataloader)):
            length += len(label)
            img = img.to(device)
            label = label.to(device)

            # first_spike_time[b, c] = timestep of first spike for sample b,
            # class c.  Initialised to sim_len (sentinel for "never spiked").
            first_spike_time = None

            for t in range(sim_len):
                out = model(img)  # [batch, num_classes]

                if first_spike_time is None:
                    first_spike_time = torch.full(
                        out.shape, sim_len, dtype=torch.float, device=out.device
                    )

                # Record the first timestep at which each output neuron fires
                spiked_now = out > 0
                not_yet_spiked = first_spike_time >= sim_len
                first_spike_time[spiked_now & not_yet_spiked] = float(t)

                # Decode accumulated first-spike times and measure accuracy
                decoded = decode_ttfs_output(first_spike_time, sim_len)
                tot[t] += (label == decoded.max(1)[1]).sum()

            # Accumulate loss on the final decoded output for this batch
            decoded = decode_ttfs_output(first_spike_time, sim_len)
            epoch_loss += loss_fn(decoded, label).item()
            functional.reset_net(model)

    if length == 0:
        return tot, 0.0
    return (tot / length), epoch_loss / length


def eval_ann(test_dataloader, model, loss_fn, device, rank=0):
    epoch_loss = 0
    tot = torch.tensor(0.).cuda(device)
    model.eval()
    model.cuda(device)
    length = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            img = img.cuda(device)
            label = label.cuda(device)
            out = model(img)
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            length += len(label)    
            tot += (label==out.max(1)[1]).sum().data
    return (tot/length), epoch_loss/length


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
            if isinstance(m, (CombinedNode, GN, GN_TTFS)):
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