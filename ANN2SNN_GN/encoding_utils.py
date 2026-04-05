"""
TTFS (Time-to-First-Spike) input/output encoding utilities.

Reference:
    B. Rueckauer and S.-C. Liu, "Conversion of analog to spiking neural
    networks using sparse temporal coding," 2017.

In TTFS coding, information is encoded in spike latency rather than spike
rate:
    - Earlier spike  <-> higher input value
    - Later spike    <-> lower input value
    - No spike       <-> zero / very low value
"""

import torch


def encode_ttfs_input(ann_output: torch.Tensor, v_threshold: float) -> torch.Tensor:
    """Pre-charge the membrane potential based on the ANN layer output.

    Scales the ANN output (assumed to be in [0, v_threshold]) to an initial
    membrane potential in the same range.  Neurons with a high pre-charge
    value are closer to their firing threshold and therefore fire earlier,
    encoding the magnitude in spike latency.

    Args:
        ann_output: ANN layer output tensor in [0, v_threshold].
        v_threshold: The neuron firing threshold.

    Returns:
        Initial membrane potential tensor with the same shape as ann_output,
        scaled to [0, v_threshold].
    """
    # Normalise to [0, 1] then scale back to [0, v_threshold]
    return torch.clamp(ann_output, 0.0, v_threshold)


def decode_ttfs_output(spike_times: torch.Tensor, T: int) -> torch.Tensor:
    """Decode TTFS first-spike timesteps back to analog confidence values.

    Implements the inverse mapping described in Rueckauer & Liu (2017):
        decoded = 1 - spike_time / T

    An earlier first spike (small spike_time) yields a value close to 1.0
    (high confidence), while a later spike yields a value close to 0.0.
    Neurons that never fired during the simulation window are assigned 0.0.

    Args:
        spike_times: Integer or float tensor of first-spike timesteps.
                     Entries equal to T (or > T) indicate "never spiked".
        T: Total number of simulation timesteps.

    Returns:
        Decoded analog values in [0, 1] with the same shape as spike_times.
        Never-spiked entries are set to 0.0.
    """
    never_spiked = spike_times >= T
    decoded = 1.0 - spike_times.float() / T
    decoded = torch.clamp(decoded, 0.0, 1.0)
    decoded[never_spiked] = 0.0
    return decoded
