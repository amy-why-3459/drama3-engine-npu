"""Batch equal-length DAC chunks without padding or changing streaming boundaries."""
from collections import defaultdict
import torch


def consume_many(backend, sessions, frames, max_batch):
    if max_batch < 1 or len(sessions) != len(frames):
        raise ValueError('Invalid codec batch dimensions')
    results = [None] * len(sessions)
    groups = defaultdict(list)
    for index, (session, code) in enumerate(zip(sessions, frames)):
        try:
            finished = code is None
            if code is not None:
                # All tower heads have 4096 logits. The upstream DAC normalizes
                # residual indices to its 1024-entry codebooks before lookup.
                if (code.shape != (10,) or code.dtype not in (torch.int32, torch.int64)
                        or not ((code >= 0) & (code < 4096)).all()):
                    raise RuntimeError('Invalid generated acoustic frame')
                session.codes.append(code)
                session.generated_frames += 1
            if finished and session.generated_frames == 0:
                raise RuntimeError('EOS without acoustic frames')
            pending = session.generated_frames - session.emitted_frames
            target = backend.first_frames if session.emitted_frames == 0 else backend.chunk_frames
            if pending and (finished or pending >= target):
                groups[len(session.codes)].append((index, session, len(session.codes)-pending, finished))
            else:
                results[index] = (None, finished)
        except Exception as error:
            results[index] = error
    for members in groups.values():
        for first in range(0, len(members), max_batch):
            group = members[first:first+max_batch]
            try:
                codes = torch.stack([torch.stack(session.codes).T for _, session, _, _ in group])
                audio = backend.codec.decode(codes.to(backend.model.devices[-1]))
                valid = torch.isfinite(audio).all(dim=-1).cpu().tolist()
                pcm = (audio.clamp(-1, 1)*32767).short().cpu().numpy().astype('<i2')
                for row, (index, session, context, finished) in enumerate(group):
                    if not valid[row]:
                        results[index] = RuntimeError('Nonfinite decoded audio')
                        continue
                    payload = pcm[row, context*backend.codec.hop_length:].tobytes()
                    session.emitted_frames = session.generated_frames
                    session.codes = session.codes[-backend.left_frames:] if backend.left_frames else []
                    results[index] = (payload, finished)
            except Exception as error:
                for index, _, _, _ in group:
                    results[index] = error
    return results
