"""Music3 autoregressive response binding from the MIT-licensed core."""
import torch
from particle_sliders.audio import ARLastHiddenResponseAdapter


class Music3ARResponseAdapter(ARLastHiddenResponseAdapter):
    family = 'music3_native_qwen3_ar'

    def _hidden(self, ids):
        device = next(self.model.parameters()).device
        tokens = torch.tensor([ids], device=device, dtype=torch.long)
        # Unpadded token rows: the final index is the native last-real-token read.
        return self.model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                use_cache=False).last_hidden_state[:, -1]
