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
        # cast condition tensors
        for key in c:
            if isinstance(c[key], torch.Tensor):
                c[key] = c[key].to(self.dtype)

        # --- I2V conditioning image latent ---
        concat_images = kwargs.pop("concat_images", None)

        if concat_images is not None:
            z = concat_images.to(x.device, dtype=x.dtype)

            # Ensure 5D: [B, C, T, H, W]
            if x.dim() != 5 or z.dim() != 5:
                raise RuntimeError(f"Expected 5D video tensors but got x:{x.shape}, z:{z.shape}")

            # --- THE FIX YOU ASKED ABOUT IS HERE ---
            # Expand image latent along the time dimension to match x
            if z.shape[2] != x.shape[2]:
                if z.shape[2] == 1:
                    # broadcast image-latent across all T frames
                    z = z.expand(-1, -1, x.shape[2], -1, -1)
                else:
                    raise RuntimeError(
                        f"Cannot broadcast image-latent T={z.shape[2]} to match latent T={x.shape[2]}"
                    )
            # ---------------------------------------

            # concat on CHANNEL dimension → [B, 32, T, H, W]
            x = torch.cat([x, z], dim=1)

        # call inner DiT
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )

    def switch_adaln_layer(self, mixin_class_name):
        if hasattr(self.diffusion_model, 'switch_adaln_layer'):
            self.diffusion_model.switch_adaln_layer(mixin_class_name)
        else:
            raise AttributeError(
                f"The diffusion model does not have a method named 'switch_adaln_layer'"
            )

