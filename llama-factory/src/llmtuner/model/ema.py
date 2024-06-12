import tree
from torch import nn

from .utils import get_safe_state_dict


class ModelEMA(nn.Module):
    def __init__(
        self, init_state_dict, decays: list, on_cpu=True, local_init_model_name=None
    ) -> None:
        super().__init__()
        assert init_state_dict is not None
        assert isinstance(decays, list)
        if local_init_model_name is not None:
            cur_state_dict = init_state_dict
            init_state_dict = get_safe_state_dict(cur_state_dict, local_init_model_name)
        self.on_cpu = on_cpu
        self.original_device = tree.map_structure(lambda x: x.device, init_state_dict)
        print(
            "Original model device",
            self.original_device,
        )

        self.ema_state_dict = {}
        self.decays = decays
        for decay in decays:
            if on_cpu:
                self.ema_state_dict[decay] = tree.map_structure(
                    lambda x: x.cpu().clone(), init_state_dict
                )
            else:
                # TODO on device computation
                self.ema_state_dict[decay] = tree.map_structure(
                    lambda x: x.clone(), init_state_dict
                )
            print(f"loaded state copy for decay: {decay} ...")

    def _update(self, online_state_dict, decay: float):
        weight = 1 - decay
        for ema_v, model_v in zip(
            self.ema_state_dict[decay].values(), online_state_dict.values()
        ):
            if self.on_cpu:
                ema_v.lerp_(model_v.cpu(), weight)
            else:
                ema_v.lerp_(model_v, weight)

    def update(self, online_state_dict):
        for decay in self.decays:
            self._update(online_state_dict, decay)
