Wrapper function for denoising.

#### `Parameters`
1. `cond` list of dictionary
2. `uc` unconditional conditions
3. `tile_size`
4. `overlap_size`
declares a [[Denoiser]] with lambda keyword
Calls the [[sampler_single_or_multiple]] function

#### Methods
1. `disable_untrainable_params`
2. `_init_first_stage`
3. `forward`
4. `shared_step`
5. `get_input`
6. `decode_first_stage`
7. `encode_first_stage`
8. `sample_single`
9. `sample_multi_prompt`
10. `log_conditionings`
11. `log_video`

#### `disable_untrainable_params`

processing