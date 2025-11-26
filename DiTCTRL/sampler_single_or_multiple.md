The base class for a sampler is: [[BaseDiffusionSampler]]
Let's just say, this is an entry point. 
A `__call__` method is basically used. So, through this wrapper we will go to [[BaseDiffusionSampler]] and go in depth to specific implementations from there.
But we can also do a skip connection. 
Basically, for this particular operation, we use
[[VPSDEDPMPP2MSampler_Single]] and [[VPSDEDPMPP2MSampler_MultiPrompt]]
understood