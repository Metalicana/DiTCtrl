This actually uses [[ADALN_mixin]]
This also takes the conditions from `generate_conditioning_parts`
Sets the seed.
My assumption is this is where the main magic happens.

It loops for each adaln names from the config
First of all, it switches the ADALN layer.

It then uses the sample function to get an output.
Specifically, the [[sampler_single_or_multiple]]
since this is for multi prompt, it uses the multiprompt sampling.


Processing