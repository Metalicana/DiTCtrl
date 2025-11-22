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
        # Cast all cond tensors to the wrapper dtype
        for key in c:
            if isinstance(c[key], torch.Tensor):
                c[key] = c[key].to(self.dtype)

        # --- original "concat" conditioning (if used) ---
        concat = c.get("concat", None)
        if concat is not None and concat.numel() > 0:
            if x.dim() == 4:
                # [B, C, H, W]
                x = torch.cat([x, concat.to(x.device, dtype=x.dtype)], dim=1)
            elif x.dim() == 5:
                # x: [B,T,C,H,W]
                z = c.get("concat", None)
                if z is not None:
                    z = z.to(x.device, dtype=x.dtype)

                    # ensure z is 5D [B,T,C,H,W]
                    if z.dim() == 4:
                        z = z.unsqueeze(1)   # [B,1,C,H,W]

                    # broadcast z across time if needed
                    if z.shape[1] != x.shape[1]:
                        if z.shape[1] == 1:
                            z = z.expand(-1, x.shape[1], -1, -1, -1)
                        else:
                            raise RuntimeError(
                                f"Cannot broadcast z T={z.shape[1]} to x T={x.shape[1]}"
                            )

                    # concat ALONG CHANNELS → dim=2
                    x = torch.cat([x, z], dim=2)

            else:
                raise ValueError("Input tensor must be 4D or 5D")

        # --- new: image-to-video conditioning via concat_images ---
        concat_images = kwargs.pop("concat_images", None)
        if concat_images is not None:
            z = concat_images.to(x.device, dtype=x.dtype)

            if z.dim() == 4:
                # [B, C, H, W] -> [B, 1, C, H, W]
                z = z.unsqueeze(1)

            if z.dim() != 5:
                raise ValueError(f"Expected concat_images to be 5D, got {z.shape}")

            # Broadcast over time: z: [B, 1, C, H, W] -> [B, T, C, H, W]
            if z.shape[1] == 1 and x.shape[1] > 1:
                z = z.expand(-1, x.shape[1], -1, -1, -1)

            # Now shapes should align on B,T,H,W; concat along **channels**
            x = torch.cat([x, z], dim=2)  # [B, T, C+16, H, W]

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

