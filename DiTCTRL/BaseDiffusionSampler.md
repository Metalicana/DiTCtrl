### Methods
`__init__()`:
	It contains the following variables that are relevant:
		1. `discretization_config`
		2. `num_steps`
		3. `guider_config`
	The discretization and guider are loaded from config using the `instantiate_from_config` method.
	Have to understand [[Discretization]] and [[Guider]]
`prepare_sampling_loop`
	The mechanism is easy, but requires I understand the concept of [[Discretization]]
`denoise`
	Not only do I need to understand the noise schedule concept, but need to understand how denoising happens with the [[Denoiser]] class.
`get_sigma_gen`
	Printing infos only.
The more specific implementation of this is [[VideoDDIMSampler]]

Processing