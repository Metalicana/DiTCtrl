Very similar.
#### `init()`
Here, the `mask_save_dir, ref_token_idx, cur_token_idx, attn_map_step_idx, attn_map_layer_idx, thres` are all used!

If attn_map_step_idx and layer is not given, it's just every 5 steps.
At the end, it makes sure the save directory is active.

#### `layer_forward(), reinit()`
same as the KVharingAdaLNMixin

#### `attention_fn`
This actually uses this method called [[AttentionMapController]]

First, same as before, but then the `attn_controller` is saved with save_cur_attn_map for query layer, key layer, with current step and layer.

If in correct step and correct layer,
Same as before. the source is calculated using old implementation.

Then, 
if there is no cross attention yet, or if it is 0, 
then we set the self attention mask as None.
Target output now becomes `attn_batch` method with
query target, and KV coming from the source with some weights.
Basically calculate with no mask available, so this automatically becomes trivial.

If there is cross attention score,
We first get the map, by doing that aggregate cross attention map function, which basically gives you the map attending to the tokens.
Globally set self attention mask as this current self attention mask, which actually comes from the aggreate cross attention map. WEIRD.

TBC

#### `attn_batch`
For some reason, very elaborate. OKay,
It takes in q, k, v. attention_mask, ref_token_index, attention_dropout, log_attention_weights, scaling_attention_score.

First take the self_attn_mask from attention controller,
then we binarize this based on the threshold.

Great.
Now, we have text_length, total length, and video length separated.

We make a full mask, which is 
total length x total length.

the binarized mask is unsqueezed to become video length x video length

`full_mask[text_length:, text_length:] = self_attn_mask`

Then, we made foreground mask, and background mask.

Set text parts of foreground and background mask to 0.



set foreground masks's text parts to full
Understood