import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0")) and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


# class OpenAIWrapper(IdentityWrapper):
#     def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
#         for key in c:
#             c[key] = c[key].to(self.dtype)

#         if x.dim() == 4:
#             x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
#         elif x.dim() == 5:
#             x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=2)
#         else:
#             raise ValueError("Input tensor must be 4D or 5D")

#         return self.diffusion_model(
#             x,
#             timesteps=t,
#             context=c.get("crossattn", None),
#             y=c.get("vector", None),
#             **kwargs,
#         )

#     def switch_adaln_layer(self, mixin_class_name):
#         if hasattr(self.diffusion_model, 'switch_adaln_layer'):
#             self.diffusion_model.switch_adaln_layer(mixin_class_name)
#         else:
#             raise AttributeError(f"The diffusion model does not have a method named 'switch_adaln_layer'")

class OpenAIWrapper(IdentityWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        # cast cond to correct dtype
        for key in c:
            v = c[key]
            if isinstance(v, torch.Tensor):
                c[key] = v.to(self.dtype)

        # --- I2V conditioning ---
        concat_images = kwargs.pop("concat_images", None)
        if concat_images is not None:
            z = concat_images.to(x.device, dtype=x.dtype)

            # make 5D
            if z.dim() == 4:
                z = z.unsqueeze(1)

            if z.dim() != 5:
                raise ValueError(f"Expected concat_images to be [B,T,C,H,W], got {z.shape}")

            # broadcast temporal dim
            B, Tx, Cx, H, W = x.shape
            Bz, Tz, Cz, Hz, Wz = z.shape

            if Tz != Tx:
                if Tz == 1:
                    z = z.expand(B, Tx, Cz, Hz, Wz)
                else:
                    raise RuntimeError(f"Cannot broadcast z T={Tz} to x T={Tx}")

            # concat along channels
            x = torch.cat([x, z], dim=2)

        # call transformer
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
        )


    def switch_adaln_layer(self, mixin_class_name):
        if hasattr(self.diffusion_model, 'switch_adaln_layer'):
            self.diffusion_model.switch_adaln_layer(mixin_class_name)
        else:
            raise AttributeError(
                f"The diffusion model does not have a method named 'switch_adaln_layer'"
            )

