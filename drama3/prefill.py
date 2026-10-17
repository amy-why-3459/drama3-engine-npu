"""Experimental left-padded prefill with request-owned unpadded caches."""
from copy import copy
import torch
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer
from drama3.batching import DecodeState
from drama3.model.tower import TowerBatch


@torch.inference_mode()
def prefill_many(generator, prompts, samplings):
    if not prompts or len(prompts) != len(samplings):
        raise ValueError('Expected matching nonempty prompts and sampling parameters')
    model = generator.model
    lengths = [len(p) for p in prompts]
    for p, n, s in zip(prompts, lengths, samplings):
        if p.ndim != 2 or p.shape[1] != 1 + model.tower.num_codebooks or n < 1:
            raise ValueError('Invalid prompt')
        if n + s.max_frames > model.tower.cache.max_len:
            raise ValueError('Prompt and generation exceed cache capacity')
    acquired = []
    try:
        for _ in prompts:
            owner = object()
            slot = model.tower.cache.acquire(owner, generator.active_slots)
            generator.active_slots.add(slot)
            acquired.append((owner, slot))
        maximum = max(lengths)
        device = model.devices[0]
        frames = torch.zeros((len(prompts), maximum, prompts[0].shape[1]), dtype=torch.long, device=device)
        columns = torch.arange(maximum, device=device)[None]
        offsets = torch.tensor([maximum-n for n in lengths], device=device)[:, None]
        mask = columns >= offsets
        positions = (columns-offsets).clamp_min(0)
        for row, (p, n) in enumerate(zip(prompts, lengths)):
            frames[row, -n:] = p.to(device)
        result = model.hidden(frames, attention_mask=mask, position_ids=positions)
        hidden = result.last_hidden_state.to(model.devices[-1])
        inputs = [p.to(model.devices[-1]) for p in prompts]
        selected = [i for i,n in enumerate(lengths) if n > 1]
        if selected:
            teacher_hidden = torch.cat([hidden[i, -lengths[i]:-1] for i in selected])
            before = torch.cat([inputs[i][:-1] for i in selected])
            after = torch.cat([inputs[i][1:] for i in selected])
            model.tower.run(teacher_hidden, before[:, 1:], model.tower.is_semantic(before[:, 0]),
                            after[:, 0], after[:, 1:],
                            TowerBatch.groups([acquired[i][1] for i in selected],
                                              [0]*len(selected), [lengths[i]-1 for i in selected], hidden.device),
                            (1.,30,.9))
        states = []
        for row, (n, sample, (owner, slot)) in enumerate(zip(lengths, samplings, acquired)):
            cache = copy(result.past_key_values)
            cache.layers = []
            for source in result.past_key_values.layers:
                target = copy(source)
                if type(source) is DynamicLayer:
                    target.keys = source.keys[row:row+1, :, -n:].clone()
                    target.values = source.values[row:row+1, :, -n:].clone()
                elif type(source) is LinearAttentionLayer:
                    for name in ('conv_states','recurrent_states'):
                        setattr(target,name,{0:getattr(source,name)[0][row:row+1].clone()})
                    for name in ('is_conv_states_initialized','is_recurrent_states_initialized','has_previous_state','conv_kernel_size'):
                        setattr(target,name,dict(getattr(source,name)))
                else:
                    raise TypeError(f'Unsupported cache layer: {type(source).__name__}')
                cache.layers.append(target)
            states.append(DecodeState(owner,slot,n-1,inputs[row][-1:],hidden[row,-1:].clone(),cache,
                                      torch.full((1,16),-1,device=hidden.device,dtype=torch.long),sample))
        return states
    except BaseException:
        for owner, slot in acquired:
            generator.active_slots.remove(slot)
            model.tower.cache.release(slot,owner)
        raise
