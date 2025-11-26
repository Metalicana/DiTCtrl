An NN Module. 
## Properties
1. `Model`, specifically the [[DiffusionTransformer]]
2. Lots of configs, all came from model config. Specifically:
	1. `log_keys, input_key`
	2. `network_config, network_wrapper`
	3. `denoiser_config`
	4. `sampler_single_config, sampler_multi_prompt_config`
	5. `sampler_ddim_config`
	6. `conditioner_config`
	7. `first_stage_config`
	8. `loss_fn_config`
	9. `scale_factor`
	10. `latent_input`
	11. `no_cond_log`
	12. `compile_model`
	13. `en_and_decode_n_samples_a_time`
	14. `lr_scale`
	15. `lora_train`
	16. `use_pd`
3. Items instantiated from the configs
	1. `model`
	2. `denoiser`
	3. `sampler_single`
	4. `sampler_multi_prompt`
	5. `sampler_ddim`
	6. `conditioner`
	7. `loss_fn`
	
#### Methods
1. `disable_untrainable_params`
2. `_init_first_stage`
3. `forward`
4. `shared_step`
5. `get_input`
6. `decode_first_stage`
7. `encode_first_stage`
8. [[sample_single_or_multiple]]
9. `log_conditionings`
10. `log_video`
11. `switch_adaln_layer`
#### `disable_untrainable_params`
Go through all the named parameters, if the requires_grad is false, then continue. Otherwise, checks some prefix, or if starts with some prefix within the not_trainable_prefixes, then flag is raised, and their requires_grad is turned False.
#### `_init_first_stage`
it loads model from config. Then disables training. makes all parameter requires_grad false, and then sets first_stage_model of this class, as it.

#### `forward`
It calculates loss, by invoking self.loss_fn with modell, denoiser, conditioner, x, and batch. Here what this x is, is important. Then it calculates the mean, updates the loss dictionary and returns them both.

#### `get_input`
From this dictionary called batch, returns the input.
#### `shared_step`
In our batch dictionary, we are doing a few calculations.
Firstly, we wanna encode the low_resolution version of the input.
So we down sample it with 1/lr_scale, upscale back to original dimension.
use the `encode_first_stage` method, to encode the low_res image and save it.

then input X has dimensions B T C H W
Gotta change it to B C T H W for VAE
then change it back.
Forward call, to collect loss, then update the encoded input.

#### `decode_first_stage`
Latents to RGB.
Firstly, gotta scale it. Because transformers and vae's have different scaling expectations.
They then get how many samples, and rounds they will need to do the decoding.
Gotta do it in chunks, otherwise OOM.
We then simply use `first_stage_model`'s decode function for this each chunk in a loop.
Output is concatenate them and return.

#### `encode_first_stage`
Firstly, get temporal shape.
B C T H W is the expected dims for VAE.
If it's a video and latents are already calculated, just reorder for transformer, rescale it, send it back.

Otherwise, do that same chunking. n_samples, n_rounds etc
Then for each of those chunks, `first_stage_model`'s encode function is called. Concatenate output, scale it.


Then we have, of course the [[sample_single_or_multiple]]

#### `log_conditioning, log_video`
Just used for debugging. Not interested.

#### `switch_adaln_layer`
Given the mixin class name, it calls the model's `switch_adaln_layer`
method.

Understood