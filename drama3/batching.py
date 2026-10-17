"""Continuous batched decode over request-owned DRAMA-3 hybrid state."""
from dataclasses import dataclass

import torch

from drama3.generation import Sampling
from drama3.model.batch_cache import DecodeCacheBatch
from drama3.model.tower import TowerBatch, sample_rows


@dataclass(eq=False)
class DecodeState:
    owner: object
    slot: int
    position: int
    inputs: torch.Tensor
    hidden: torch.Tensor
    cache: object
    history: torch.Tensor
    sampling: Sampling
    generated: int = 0
    closed: bool = False
    finished: bool = False


class BatchedGenerator:
    def __init__(self, model):
        self.model = model
        self.active_slots = set()

    @torch.inference_mode()
    def prefill(self, prompt, sampling):
        model = self.model
        if prompt.ndim != 2 or prompt.shape[1] != 1 + model.tower.num_codebooks:
            raise ValueError('Prompt must contain token and codebook columns')
        length = len(prompt)
        if length < 1 or length + sampling.max_frames > model.tower.cache.max_len:
            raise ValueError('Prompt and generation exceed cache capacity')
        owner = object()
        slot = model.tower.cache.acquire(owner, self.active_slots)
        self.active_slots.add(slot)
        try:
            result = model.hidden(prompt[None])
            hidden = result.last_hidden_state[0].to(model.devices[-1])
            inputs = prompt.to(model.devices[-1])
            if length > 1:
                model.tower.run(hidden[:-1], inputs[:-1, 1:],
                                model.tower.is_semantic(inputs[:-1, 0]),
                                inputs[1:, 0], inputs[1:, 1:],
                                TowerBatch.prefill(slot, length - 1, hidden.device),
                                (sampling.temperature, sampling.top_k, sampling.top_p))
            return DecodeState(owner, slot, length - 1, inputs[-1:], hidden[-1:],
                               result.past_key_values,
                               torch.full((1, 16), -1, device=hidden.device, dtype=torch.long), sampling)
        except BaseException:
            self.active_slots.remove(slot)
            model.tower.cache.release(slot, owner)
            raise

    @torch.inference_mode()
    def step(self, states):
        """Return one CPU acoustic frame (or EOS/limit None) per input state.

        Completed/cancelled requests remain owned until the caller closes them.
        Fresh prefills and continued decodes can participate in the same step.
        """
        if len({id(state) for state in states}) != len(states) or any(s.closed for s in states):
            raise ValueError('Expected distinct open request states')
        outputs = [None] * len(states)
        active = [(i, s) for i, s in enumerate(states) if not s.finished and s.generated < s.sampling.max_frames]
        if not active:
            return outputs
        model = self.model
        continued = [s for _, s in active if s.generated]
        if continued:
            batch = DecodeCacheBatch([s.cache for s in continued])
            frames = torch.cat([s.inputs for s in continued])[:, None]
            result = model.hidden(frames, cache=batch.cache, **batch.model_inputs(model.devices[0]))
            hidden = result.last_hidden_state[:, -1].to(model.devices[-1])
            batch.commit()
            for row, state in enumerate(continued):
                state.hidden = hidden[row:row + 1]
        hidden = torch.cat([s.hidden for _, s in active])
        logits = model.semantic_logits(hidden)
        width = logits.shape[-1]
        temperatures = torch.tensor([s.sampling.temperature for _, s in active], device=hidden.device)
        top_k = torch.tensor([s.sampling.top_k if 0 < s.sampling.top_k <= width else width
                              for _, s in active], device=hidden.device)
        top_p = torch.tensor([s.sampling.top_p for _, s in active], device=hidden.device)
        first = model.candidate_ids[sample_rows(logits, temperatures, top_k, top_p)]
        redraw = model.candidate_ids[sample_rows(logits, torch.ones_like(temperatures),
                                                torch.full_like(top_k, min(30, width)),
                                                torch.full_like(top_p, .95))]
        history = torch.cat([s.history for _, s in active])
        repeated = (history == first[:, None]).any(-1)
        token = torch.where(repeated & model.tower.is_semantic(first), redraw, first)
        live_rows = (token != model.config['eos_token_id']).nonzero().flatten()
        live_indices = live_rows.cpu().tolist()
        live_set = set(live_indices)
        for row, (_, state) in enumerate(active):
            if row not in live_set:
                state.finished = True
        if not live_indices:
            return outputs
        live_states = [active[row][1] for row in live_indices]
        inputs = torch.cat([s.inputs for s in live_states])
        slots = torch.tensor([s.slot for s in live_states], device=hidden.device)
        positions = torch.tensor([s.position for s in live_states], device=hidden.device)
        tokens = token[live_rows]
        codes = model.tower.run(hidden[live_rows], inputs[:, 1:], model.tower.is_semantic(inputs[:, 0]),
                                tokens, None, TowerBatch.decode(
                                    slots, positions, positions_cpu=[s.position for s in live_states],
                                    cache_length=model.tower.cache.max_len),
                                (temperatures[live_rows], top_k[live_rows], top_p[live_rows]))
        cpu_codes = codes.cpu()
        next_inputs = torch.cat((tokens[:, None], codes), dim=-1)
        next_history = torch.cat((history[live_rows, 1:], tokens[:, None]), dim=-1)
        for row, active_row in enumerate(live_indices):
            output_index, state = active[active_row]
            outputs[output_index] = cpu_codes[row]
            state.inputs = next_inputs[row:row + 1]
            state.history = next_history[row:row + 1]
            state.position += 1
            state.generated += 1
        return outputs

    def close(self, state):
        if not state.closed:
            self.model.tower.cache.release(state.slot, state.owner)
            self.active_slots.remove(state.slot)
            state.closed = True
            state.cache = None
            state.inputs = state.hidden = state.history = None
