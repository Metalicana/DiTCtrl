### Attributes
1. `height, width` are the height, width of the frames, when you patch it. (128x128 image, with 4x4 patches mean grid of 32x32, width and height means that 32)
2. `hidden_size` is size of the transformer embedding (d_model)
3. `num_layers` how many layers of the transformer
4. `time_embed_dim` this has something to do with time conditioning for video diffusion
5. `compressed_num_frames` is total frames after some temporal downsampling
6. `text_length` number of tokens in text input part
7. `qk_ln` query-key layer normalization. Stabilizing factor in training
8. `elementwise_affine` learns scale and shift parameters in LayerNorm
9. `start_layer, start_step` the layer and step where this is used
10. `layer_idx, step_idx` the indices of the layer and step where this KV sharing will be used
11. `end_step, end_layer` where it ends
12. `overlap_size` number of frames overlapping between two videos
13. `sampling_num_frames` number of frames per sampling iteration
14. `mask_save_dir` directory where mask will be saved
15. `ref_token_idx, cur_token_idx` not sure, not used in this
16. `attn_map_step_idx, attn_map_layer_idx` not used
17. `thres=0.1` assuming this is used in the mask guided version
18. `num_prompts, num_transition_blocks, longer_mid_segment` are parameters for sampling long videos
19. `is_edit` true for editing, not for this layer
20. `reweight_token_idx, reweight_scale` for editing, not relevant here
21. `adaLN_modulations` is an nn moduleList, which is an nn.Sequential of SiLU, Linear(time_embed_dim, 12*hidden_size) for all the layers.
22. `query_layernorm_list` is another nn moduleList, which is list of whatever the LayerNorm function returns. But it takes into it, elementwise_affine so scale, shift, gate parameters, and also the hidden_size_head
23. `key_layernorm_list` same as query_layer_norm


### Methods
The methods are:
1. `init`
2. `layer_forward`
3. `reinit`
4. `attention_fn`
5. `attention_batch`
6. `after_total_layers`

#### `__init__()`
Starts off with setting some parameters.
Then, adds the adaLN_modulations, query_layernorm_list, and key_layernorm_list.

#### `layer_forward()`
Through a single transformer layer, what happens.
For some reason, they take the text_length from the kwargs, and not self.text_length.

Then, they take the hidden states for text, and image.
`hidden_state[:, :text_length]` B X N X D
`hidden_state[:, text_length:]` B x (T, N) x D

Then, we have the layer loaded from `self.transformer.layers['layer_id]`
and also, we have the specific `adaLN_modulation` which contains a bunch of stuff, like `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, text_shift_msa, text_scale_msa, text_gate_msa, text_shift_mlp, text_scale_mlp, text_gate_mlp`
We also got `gate_msa, gate_mlp, text_gate_msa, text_gate_mlp` unsqueezed at dimension 1. dk why.

Apply layer normalization to image hidden states, text hidden state, and also apply modulate function.

So after this layer normalization, modulation, we finally concatenate the attention back in `attention_input`
and `attention_output` is calculated using `layer.attention`. This takes in a `mask`, which might be relevant later.
Split outputs into text, and image part again.

This part is interesting:
`text/image_hidden_states = text/image_hidden_states + text/image_gate_msa * text/image_attention_ouput` I am assuming, we use the gate as the controlling parameter on how much the residual connection should go.

Also, these hidden states go through post attention layernorm, for the mlp stuff. Almost same logic as attention input, but for mlp. This time it uses the `gate_mlp`

Finally, concatenate the hidden states again.

add 1 to the cur_layer variable.
if cur_layer equal to num_layers, meaning, we are crossed all layers, we reset it to 0. If this is edit, then we add 1 to cur_step.
Otherwise, we add 1 to the segment.

#### `reinit`

Havent been called in the previous two functions, but it resets the cur_step, cur_layer, and for each layer in adaLN_modulations, `nn.init.constant_` is applied to weights and biases. Assuming it restarts.

#### `attention_fn`
Actually very easy.
Firstly, if the query key layer norm is true, apply that.
Then if we are in the right layer and right step, we basically first get the query, key, and value from their corresponding layers.
Then, we use old implementation to calculate the source value.
For target however, we use target query, but key and value is old.
This is essentially the KV sharing mechanism.

What I have to do, is cache this value.

#### `attn_batch`
Just for batch operation i guess.
#### `after_total_layers`
Does nothing
Understood