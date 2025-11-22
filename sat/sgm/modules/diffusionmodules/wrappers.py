import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"
OPENAII2VUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAII2VWrapper"

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


class OpenAIWrapper(IdentityWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        for key in c:
            c[key] = c[key].to(self.dtype)

        if x.dim() == 4:
            x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        elif x.dim() == 5:
            x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=2)
        else:
            raise ValueError("Input tensor must be 4D or 5D")

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
            raise AttributeError(f"The diffusion model does not have a method named 'switch_adaln_layer'")


class IdentityI2VWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        from packaging import version

        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0")) and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAII2VWrapper(IdentityWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        # Cast all cond tensors to wrapper dtype
        for key, v in c.items():
            if isinstance(v, torch.Tensor):
                c[key] = v.to(self.dtype)

        # -------------------------------------------------
        # I2V: extra conditioning frames via `concat_images`
        # -------------------------------------------------
        concat_images = kwargs.pop("concat_images", None)
        if concat_images is not None:
            z = concat_images.to(x.device, dtype=x.dtype)

            # Make z 5D: [B, T, C, H, W]
            if z.dim() == 4:
                # [B, C, H, W] -> [B, 1, C, H, W]
                z = z.unsqueeze(1)

            if z.dim() != 5:
                raise ValueError(f"Expected concat_images to be [B,T,C,H,W] (or [B,C,H,W]), got {z.shape}")

            B, Tx, Cx, Hx, Wx = x.shape
            Bz, Tz, Cz, Hz, Wz = z.shape
            # --- match temporal dim Tz -> Tx ---

            if Tz != Tx:
                # temporal center-crop or interpolate
                if Tz > Tx:
                    # center crop
                    start = (Tz - Tx) // 2
                    z = z[:, start:start+Tx]
                else:
                    # upsample temporally (rare)
                    z = torch.nn.functional.interpolate(
                        z.permute(0, 2, 3, 4, 1),  # BCHWT -> BCHWT'
                        size=Tx,
                        mode="nearest"
                    ).permute(0, 4, 1, 2, 3)

            # --- batch broadcast if needed ---
            if Bz != B:
                if Bz == 1:
                    z = z.expand(B, Tz, Cz, Hz, Wz)
                    Bz = B
                else:
                    raise RuntimeError(f"Cannot broadcast concat_images batch Bz={Bz} to B={B}")

            # --- spatial sanity check (you can relax this if you want) ---
            if (Hz != Hx) or (Wz != Wx):
                # You *could* interpolate here; for now, hard error to catch misconfig.
                raise RuntimeError(
                    f"Spatial mismatch between x ({Hx}x{Wx}) and concat_images ({Hz}x{Wz})"
                )

            # --- temporal alignment: make z.T == x.T ---
            if Tz != Tx:
                if Tz == 1:
                    # single frame → broadcast
                    z = z.expand(B, Tx, Cz, Hz, Wz)

                elif Tz > Tx:
                    # More frames than latent time → downsample / subsample to Tx
                    # Simple even sampling: map Tx positions into [0, Tz-1]
                    idx = torch.linspace(0, Tz - 1, Tx, device=x.device)
                    idx = idx.round().long().clamp(0, Tz - 1)
                    z = z[:, idx]  # [B, Tx, Cz, Hz, Wz]

                else:  # 1 < Tz < Tx
                    # Fewer frames than latent time → repeat to fill
                    repeat = math.ceil(Tx / Tz)
                    z = z.repeat_interleave(repeat, dim=1)[:, :Tx]

            # Now z.shape[1] == Tx and z.shape[0] == B
            # Concat along channel dimension: x: [B, T, Cx, H, W], z: [B, T, Cz, H, W]
            x = torch.cat([x, z], dim=2)

        # -----------------------------
        # Call the underlying transformer
        # -----------------------------
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,  # keep kwargs flowing if something upstream relies on them
        )

    def switch_adaln_layer(self, mixin_class_name):
        if hasattr(self.diffusion_model, "switch_adaln_layer"):
            self.diffusion_model.switch_adaln_layer(mixin_class_name)
        else:
            raise AttributeError("The diffusion model does not have a method named 'switch_adaln_layer'")
