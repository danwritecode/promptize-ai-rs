# Promptize

Promptize attempts to solve the issues with context limits when working with AI systems. It allows a user to add an attribute to their struct called "Promptize" with a "chunkable" attribute on a specific field. 

If the prompt exceeds the allowable token size for your model, then the "chunkable" field will be chunked up into equal sized parts that fit within the context limit. 

The fields from the original struct are repeated in a vec of a new version of that struct but with the chunked fields being broken up accordingly. 

It's up to the caller to call openai multiple times for each prompt that is returned as a result of chunking.