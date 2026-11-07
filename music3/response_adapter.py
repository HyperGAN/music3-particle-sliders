"""Music3 final-token coordinates over the shared model-independent AR binding."""
import torch

from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter


class Music3ARResponseAdapter(ARLastHiddenResponseAdapter):
    family = 'music3_native_qwen3_ar'

    def _hidden(self, ids):
        device = next(self.model.parameters()).device
        tokens = torch.tensor([ids], device=device, dtype=torch.long)
        return self.model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                use_cache=False).last_hidden_state[:, -1]
