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
	
## Methods
1. [[sample_single_or_multiple]]

processing