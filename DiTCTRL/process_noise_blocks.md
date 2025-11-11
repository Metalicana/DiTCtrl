Takes in a tensor of shape `1xTxCxHxW` in latent space.
C is channel in latent space, `16`
H, W is downscaled by a factor F
T is long video size

`total_segments` is number of video segment + number of overlapping segments
It does some kind of tiling here, so that we have frame indices for each segments.
For example:
`{[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]}`

Input is `3` prompts, tile size is `13`
Based on prompts, total segments is `5` (3 main parts 2 transition parts)
Overlap size = `9`
Total frames = `37`
Based on the overlap, number of tiles becomes (37-13)/(13-9) + 1 

Understood