"""Pack request-owned hybrid caches for one variable-length decode step.

No vLLM/SGLang runtime is used. Padding is masked explicitly; rotary positions
remain the original per-request positions. Only a successful step is committed.
"""
from copy import copy

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer


class DecodeCacheBatch:
    def __init__(self, caches):
        if not caches or len({id(cache) for cache in caches}) != len(caches):
            raise ValueError('Decode requires distinct nonempty request caches')
        self.owners = caches
        self.lengths = [cache.get_seq_length() for cache in caches]
        if min(self.lengths) < 1:
            raise ValueError('Decode caches must already contain a prompt')
        self.padded_length = max(self.lengths)
        self.cache = copy(caches[0])
        self.cache.layers = []
        self.committed = False
        if any(getattr(cache, 'offloading', False) for cache in caches):
            raise ValueError('Batched decode does not support offloaded caches')
        if len({len(cache.layers) for cache in caches}) != 1:
            raise ValueError('Mismatched cache layer counts')
        for index, template in enumerate(caches[0].layers):
            sources = [cache.layers[index] for cache in caches]
            if any(type(layer) is not type(template) for layer in sources):
                raise ValueError('Mismatched cache layer types')
            layer = copy(template)
            if type(template) is DynamicLayer:
                for name in ('keys', 'values'):
                    tensors = [getattr(source, name) for source in sources]
                    if any(t.shape[0] != 1 or t.shape[-2] != n for t, n in zip(tensors, self.lengths)):
                        raise ValueError('Expected one unpadded sequence per request')
                    setattr(layer, name, torch.cat([
                        F.pad(t, (0, 0, 0, self.padded_length - n))
                        for t, n in zip(tensors, self.lengths)
                    ], dim=0))
            elif type(template) is LinearAttentionLayer:
                if any(source.record_past or source.number_of_states != 1
                       or not source.has_previous_state[0] for source in sources):
                    raise ValueError('Expected initialized single-state GDN caches')
                for name in ('conv_states', 'recurrent_states'):
                    tensors = [getattr(source, name)[0] for source in sources]
                    if any(t is None or t.shape[0] != 1 for t in tensors):
                        raise ValueError('Expected one GDN state per request')
                    setattr(layer, name, {0: torch.cat(tensors, dim=0)})
                for name in ('is_conv_states_initialized', 'is_recurrent_states_initialized',
                             'has_previous_state', 'conv_kernel_size'):
                    setattr(layer, name, dict(getattr(template, name)))
            else:
                raise TypeError(f'Unsupported hybrid cache layer: {type(template).__name__}')
            self.cache.layers.append(layer)

    def model_inputs(self, device):
        lengths = torch.tensor(self.lengths, device=device, dtype=torch.long)
        columns = torch.arange(self.padded_length + 1, device=device)
        allowed = (columns[None] < lengths[:, None]) | (columns[None] == self.padded_length)
        # Supplying the prebuilt mask avoids treating logical rotary positions
        # as physical positions in the right-padded KV cache.
        return {'attention_mask': {'full_attention': allowed[:, None, None, :],
                                   'linear_attention': None},
                'position_ids': lengths[:, None]}

    def commit(self):
        if self.committed:
            raise RuntimeError('A decode batch can be committed only once')
        for layer in self.cache.layers:
            if type(layer) is DynamicLayer and layer.keys.shape[-2] != self.padded_length + 1:
                raise ValueError('Expected exactly one decoded token before commit')
        for row, (owner, length) in enumerate(zip(self.owners, self.lengths)):
            for source, target in zip(self.cache.layers, owner.layers):
                if type(source) is DynamicLayer:
                    for name in ('keys', 'values'):
                        tensor = getattr(source, name)[row:row + 1]
                        setattr(target, name, torch.cat((tensor[..., :length, :], tensor[..., -1:, :]), dim=-2))
                else:
                    for name in ('conv_states', 'recurrent_states'):
                        setattr(target, name, {0: getattr(source, name)[0][row:row + 1]})
        self.committed = True
