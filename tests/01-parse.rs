use promptize::Promptize;
use promptize_internals::Promptize;

#[derive(Promptize, Debug, serde::Serialize)]
pub struct FileContent {
    system_prompt: String,
    user_prompt: String,
    filename: String,
    #[chunkable]
    file_content: String,
    #[model]
    model: String,
    #[max_chunks]
    max_chunks: i32
}

pub const MODEL:&str = "gpt-4";
pub const TOKEN_CONTEXT_LIMIT:i32 = 8192;
pub const MAX_CHUNKS:i32 = 5;

fn main() {
    let res = FileContent::builder()
        .system_prompt(format!("You are a computer system that responds only in JSON format with no other words except for the JSON."))
        .user_prompt(format!("You are a computer system that responds only in JSON format with no other words except for the JSON."))
        .filename("foo".to_string())
        .file_content("foo".to_string())
        .model("gpt-4".to_string())
        .max_chunks(5 as i32)
        .build()
        .unwrap();
    
    let (res, file_content) = res.build_prompt().unwrap();
    let foo = file_content.reassemble("AHHHH".to_string()).unwrap();
}
