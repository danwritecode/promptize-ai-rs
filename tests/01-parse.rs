use promptize::Promptize;

#[derive(Promptize, Debug, serde::Serialize)]
pub struct FileContent {
    system_prompt: String,
    user_prompt: String,
    pub bar: String,
    pub baz: String,
    pub foo: String,
    pub filename: String,
    #[chunkable]
    pub file_content: String
}

fn main() {

}
