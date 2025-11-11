Input:
`prompts, model, num_samples, num_transition_blocks, longer_mid_segment`
Returns two arrays, conditional, and unconditional

First it calls [[calculate_segments_per_prompt]]
Then it iterates through each prompt.
Declares a dictionary, conditional = {'txt': prompt}, unconditional = {'txt' ""}
Uses this function [[get_unconditional_conditioning]] for each prompt, and then appends this to variables called
`base_conditions` and `base_uc_conditions`
For each prompt, there are set number of segments, from the first function call.
Then, that itself is in the loop. so for each segment within the prompt, append the conditions to `c_total` and `uc_total`.
My assumption is, these are the final conditions.

Also, if the prompt is not the first prompt, then there are middle parts,  which interpolates and adds the conditions and appends it.l

So end result is `c_total` and `uc_total`

Understood