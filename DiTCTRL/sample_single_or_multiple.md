Wrapper function for denoising.

#### `Parameters`
1. `cond` list of dictionary
2. `uc` unconditional conditions
3. `tile_size`
4. `overlap_size`
declares a [[Denoiser]] with lambda keyword
Calls the [[sampler_single_or_multiple]] function


Also, it does model parallelism.

Understood