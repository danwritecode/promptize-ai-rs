# Promptize

Promptize attempts to solve the issues with context limits when working with AI systems. It allows a user to add an attribute to their struct called "Promptize" with a "chunkable" attribute on a specific field. 

If the prompt exceeds the allowable token size for your model, then the "chunkable" field will be chunked up into equal sized parts that fit within the context limit. 

The fields from the original struct are repeated in a vec of a new version of that struct but with the chunked fields being broken up accordingly. 

It's up to the caller to call openai multiple times for each prompt that is returned as a result of chunking.

## Example Usage
```rust
use promptize::Promptize;

#[derive(Promptize, Debug, serde::Serialize)]
pub struct SomePrompt {
    system_prompt: String,
    user_prompt: String,
    filename: String,
    #[chunkable]
    large_content: String
}

fn main() {
    const TOKEN_CONTEXT_LIMIT:i32 = 8192;
    const MAX_CHUNKS:i32 = 5;

    let foo = SomePrompt::builder()
        .system_prompt("System prompt here".to_string())
        .user_prompt("User prompt here".to_string())
        .filename("huge_file.rs".to_string())
        .large_content("some huge amount of content here".to_string())
        .build_prompt("gpt-4", TOKEN_CONTEXT_LIMIT, MAX_CHUNKS)?;
}
```

## Limitations
1. Only 1 field can be "chunkable"
2. A chunkable field has a limit to how many times it can be chunked to fit within the context limit
3. There are some dependencies like tiktoken-rs, serde, anyhow
4. I chose to use tiktoken's prompt struct. It's up to you to map this back to whatever struct you're using

## Other considerations
1. It's up to the caller to handle the response and call open-ai multiple times for each prompt
