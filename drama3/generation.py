"""Sequential reference generation with explicit request-owned tower state."""

from collections.abc import Iterator

import torch

from drama3.model.backbone import Drama3Model
from drama3.model.tower import TowerBatch, sample_top_k_top_p


from drama3.sampling import Sampling


class ReferenceGenerator:
    def __init__(self, model: Drama3Model):
        self.model = model
        self.active_slots: set[int] = set()

    def frames(
        self, prompt: torch.Tensor, sampling: Sampling
    ) -> Iterator[torch.Tensor]:
        """Yield CPU [10] code frames, excluding EOS. Closing releases the slot.

        This eager path establishes model semantics before batched scheduling.
        It is deliberately not presented as a performance result.
        """
        model = self.model
        if prompt.ndim != 2 or prompt.shape[1] != 1 + model.tower.num_codebooks:
            raise ValueError("Prompt must contain token and codebook columns")
        length = len(prompt)
        if length < 1 or length + sampling.max_frames > model.tower.cache.max_len:
            raise ValueError("Prompt and generation exceed the tower cache capacity")
        owner = object()
        slot = model.tower.cache.acquire(owner, self.active_slots)
        self.active_slots.add(slot)
        try:
            with torch.inference_mode():
                result = model.hidden(prompt.unsqueeze(0))
                hidden = result.last_hidden_state[0].to(model.devices[-1])
                cache = result.past_key_values
                inputs = prompt.to(model.devices[-1])
                parameters = (sampling.temperature, sampling.top_k, sampling.top_p)
                if length > 1:
                    model.tower.run(
                        hidden[:-1],
                        inputs[:-1, 1:],
                        model.tower.is_semantic(inputs[:-1, 0]),
                        inputs[1:, 0],
                        inputs[1:, 1:],
                        TowerBatch.prefill(slot, length - 1, hidden.device),
                        parameters,
                    )
                hidden = hidden[-1:]
                inputs = inputs[-1:]
                history = torch.full(
                    (1, 16), -1, device=hidden.device, dtype=torch.long
                )
            for step in range(sampling.max_frames):
                # Do not hold inference_mode across yield: consumers may interleave work.
                with torch.inference_mode():
                    logits = model.semantic_logits(hidden)
                    first = model.candidate_ids[sample_top_k_top_p(logits, *parameters)]
                    redraw = model.candidate_ids[
                        sample_top_k_top_p(logits, 1.0, 30, 0.95)
                    ]
                    repeated = (history == first[:, None]).any(-1)
                    token = torch.where(
                        repeated & model.tower.is_semantic(first), redraw, first
                    )
                    if token.item() == model.config["eos_token_id"]:
                        return
                    position = length - 1 + step
                    codes = model.tower.run(
                        hidden,
                        inputs[:, 1:],
                        model.tower.is_semantic(inputs[:, 0]),
                        token,
                        None,
                        TowerBatch.extend(slot, position, position + 1, hidden.device),
                        parameters,
                    )
                    history = torch.cat((history[:, 1:], token[:, None]), dim=-1)
                    inputs = torch.cat((token[:, None], codes), dim=-1)
                    output = codes[0].cpu()
                yield output
                if step + 1 < sampling.max_frames:
                    with torch.inference_mode():
                        result = model.hidden(inputs.unsqueeze(0), cache=cache)
                        cache = result.past_key_values
                        hidden = result.last_hidden_state[:, -1].to(model.devices[-1])
        finally:
            self.active_slots.remove(slot)
            model.tower.cache.release(slot, owner)
