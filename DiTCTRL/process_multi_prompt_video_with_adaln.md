This actually uses [[BaseAdaLNMixin]]
This also takes the conditions from `generate_conditioning_parts`
Sets the seed.
My assumption is this is where the main magic happens.

It loops for each adaln names from the config (inside the callee, not inside this function)
First of all, it switches the ADALN layer given in the input.

It then uses the sample function to get an output.
Specifically, the [[sampler_single_or_multiple]]
since this is for multi prompt, it uses the multiprompt sampling.

The code loads the model first stage, which is the vae decoder I assume.
then it uses a for loop to decode chunks, instead of full 81 frames or however many frames.
saves video.
Deletes the temporary stuff.

Understood