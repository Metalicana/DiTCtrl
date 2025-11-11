Returns an array calculating how many segments each prompt should get.
If number of prompts is 1 or 2, then answer is
`[1]` or `[1, 1]` so 1 segment for each prompt.
However,
if there are more prompts, then first and last prompt will get 1 segment, rest of the middle ones will get 1 + `longer_mid_segment`

This longer_mid_segment is a parameter which is currently 0 here
understood