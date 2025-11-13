### Helper Methods
1. `read_from_cli`, `read_from_text_file` reads prompts from cli or txt files.
2. `read_from_yaml_file` reads from yaml file.
3. `get_unique_embedder_keys_from_conditioner` takes in conditions, returns the set of it ( delete duplicates)
4. `get_batch` takes key and value dictionary returns it in an orderly fashion
5. `save_video_as_grid_and_mp4` helps with visualization I guess
6. `resize_for_rectangle_crop` speaks for itself
7. `print_current_gpu_memory` this too
8. [[process_multi_prompt_video_with_adaln]]
9. `interpolate_conditions` Linear interpolation of two conditions
10. [[calculate_segments_per_prompt]] first and last prompt use 1 segment, rest use more.
11. `calculate_total_segments` a very weird way to sum up total segments
12. [[generate_conditioning_parts]]
13. `calculate_video_length` simple enough
14. [[process_noise_blocks]]
15. `get_base_prompt_indices_with_longer_mid` returns an array of indices. Something to do with tiling.
### Input
Input args, model class ([[SATVideoDiffusionEngine]])
### Flow
1. Model loading and variable defining
2. Calculate `long_video_size` using the `calculate_video_length` function
3. Noise and tiles returned from [[process_noise_blocks]]
4. Get conditioning parts from [[generate_conditioning_parts]]
5. Gets this layer called  which is in [[process_multi_prompt_video_with_adaln]]

Processing
