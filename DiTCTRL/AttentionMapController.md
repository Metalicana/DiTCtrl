### Attributes
1. `save_dir` where we save the attention visualizations
2. `text_length` token amount 
3. `height, width` latent height, width
4. `compressed_num_frames` frames downsampled temporally
5. `thres` threshold of some sort
6. `cross_attn_sum, video_self_attn_sum, and text_self_attn_sum` they literally add up the values over multiple passes!
7. `cross_attn_count, video_self_attn_count, text_self_attn_count` are used to count so we can average
8. `self_attn_mask` probably the mask that guides. IMPORTANT
### Setter-Getter-Resetter
These are for attributes 6-8

#### `save_cur_attn_map`
So this basically helps save the attention map for future.
First, it extracts batch size, number of heads, seq length, head dimension etc from q matrix.
We have q and k.


Creates a tensor called `attn_map_mean`
Nested loop:
for each batch size, step count using batch chunk.
for each num head, step count is head chunk.
chunk always set to 1.

q_chunk, and k_chunk is just the sliced version of q, k.
`[batch_idx: batch_idx+batch_chunk, head_idx: head_idx+head_chunk]`

attention score is calculated for each of these chunks, and also added `softmax`.
then in the `attn_map_mean`'s chunks, we add summation of the `attn_probs` gotten from `softmaxxing`.

the mean is then mean/num_heads
because, the summation is done for each head. for all batches.

Then we get something like `video_to_text_attn` which is a slice from the mean, `attn_map_mean[:, self.text_length:, : self.text_length]`

We also keep adding the video_to_text_attn value into cross_attn_sum

Then we do the same thing for video_to_video_attn,
and add that to video_self_attn_sum

and finally, text_to_text_attn is also added to text_self_attn_sum

all counts are increased.

#### `aggregate_cross_attn_map`

Very important.
firstly, attn_map is mean of the cross_attn_sum.
Remember, this is basically video to text attention.

Then it reshapes it. it used to be BxHWFxT
Now, it's B, F, H, W, T

Here T is seq_len, tokens.
This part is interesting, lets say you have token_idx.
It can be a single token, or multiple tokens, and just take that particular attention map.

Then get the min and max values from the attention map.

finally do normalization and send it back.

#### Visualization functions
1. `visualize_text_attn_map`
2. `visualize_video_video_temporal_attn_map`
3. `visualize_token_attention_masks`

Understood