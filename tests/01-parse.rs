use promptize::Promptize;

#[derive(Promptize, Debug, serde::Serialize)]
pub struct FileContent {
    system_prompt: String,
    user_prompt: String,
    pub filename: String,
    #[chunkable]
    pub file_content: String
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
        .build()
        .unwrap();
    
    let res = res.build_prompt(MODEL, TOKEN_CONTEXT_LIMIT, MAX_CHUNKS).unwrap();
}
