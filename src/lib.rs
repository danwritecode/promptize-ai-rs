use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse_macro_input, DeriveInput
};

const USER_PROMPT_FIELD_NAME: &str = "user_prompt";
const SYSTEM_PROMPT_FIELD_NAME: &str = "system_prompt";
const CHUNKABLE_ATTRIBUTE_NAME: &str = "chunkable";
const MODEL_ATTRIBUTE_NAME: &str = "model";
const MAX_CHUNKS_ATTRIBUTE_NAME: &str = "max_chunks";


/// Allows fields of a struct to be marked as Chunkable as to denote that they are able to be
/// broken up into chunks when being fed to an LLM to fit inside of a context window.
#[proc_macro_derive(Promptize, attributes(chunkable, model, max_chunks))]
pub fn promptize(input: TokenStream) -> TokenStream {
    let input_ast = parse_macro_input!(input as DeriveInput);
    let struct_name = &input_ast.ident;

    let fields = match get_fields(&input_ast, struct_name) {
        Ok(fields) => fields,
        Err(err) => return err,
    };

    // field validation
    match validate_fields(struct_name, fields.clone()) {
        Ok(valid) => valid,
        Err(err) => return err,
    };
    
    // assemble fields
    let chunk_field = match get_and_val_field_by_attribute(struct_name, CHUNKABLE_ATTRIBUTE_NAME, "String", fields.clone()) {
        Ok(chunk_field) => chunk_field,
        Err(err) => return err,
    };
    let cf_name = &chunk_field.ident;
    let cf_type = &chunk_field.ty;

    let model_field = match get_and_val_field_by_attribute(struct_name, MODEL_ATTRIBUTE_NAME, "String", fields.clone()) {
        Ok(mf) => mf,
        Err(err) => return err,
    };
    let mf_name = &model_field.ident;

    let max_chunks_fields = match get_and_val_field_by_attribute(struct_name, MAX_CHUNKS_ATTRIBUTE_NAME, "i32", fields.clone()) {
        Ok(mcf) => mcf,
        Err(err) => return err,
    };
    let mcf_name = &max_chunks_fields.ident;

    let standard_fields = get_standard_fields(fields.clone());
    let standard_fields_template = standard_fields.iter().map(|f| {
        let name = &f.ident;
        quote! {
            #name
        }
    });

    let fmt_string_braces = standard_fields
        .iter()
        .map(|field| {
            let name = &field.ident;
            return format!("{}:{{:?}}", name.clone().unwrap());
        })
        .collect::<String>();

    let std_fields_fmt_template = quote! {
        format!(#fmt_string_braces, #(self.#standard_fields_template.clone()),*)
    };

    let user_prompt_field = match get_field_ident_by_name(struct_name, fields.clone(), USER_PROMPT_FIELD_NAME) {
        Ok(field) => field,
        Err(err) => return err,
    };

    let system_prompt_field = match get_field_ident_by_name(struct_name, fields.clone(), SYSTEM_PROMPT_FIELD_NAME) {
        Ok(field) => field,
        Err(err) => return err,
    };

    // build fields
    let builder_name = format!("{}Builder", struct_name); 
    let builder_ident = syn::Ident::new(&builder_name, struct_name.span());

    let template_fields = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;

        if is_optional(&f) {
            return quote! {
                #name: #ty
            };
        }

        return quote! {
            #name: std::option::Option<#ty>
        };
    });

    let fields_empty = fields.iter().map(|f| {
        let name = &f.ident;
        quote! {
            #name: None
        }
    });

    let build_fields = fields.iter().map(|f| {
        let name = &f.ident;
        if is_optional(&f) {
            return quote! {
                #name: self.#name.clone()
            };
        }

        return quote! {
            #name: self.#name.clone().ok_or(anyhow::anyhow!(concat!(stringify!(#name), " is not set")))?
        };
    });

    let builder_methods = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;

        if is_optional(&f) {
            // extract root type
            let option_type = get_option_inner_type(&f);
            return quote! {
                pub fn #name(&mut self, #name: #option_type) -> &mut Self {
                    self.#name = Some(#name);
                    self
                }
            };
        }

        return quote! {
            pub fn #name(&mut self, #name: #ty) -> &mut Self {
                self.#name = Some(#name);
                self
            }
        };
    });
    
    let builder_chaining_hardcoded = quote! {
        .#system_prompt_field(format!("{} | {}", 
            self.#system_prompt_field.clone(),
            format!("Please note that due to context size limitations, the information you are receiving
            is actually the summary of several analyses. The previous analyses you did had the same system prompt
            and the same fields with the exception of {} which will now contain concatenated summaries that were
            your previous analysis.
            ", stringify!(#cf_name))
        ))
        .#user_prompt_field(self.#user_prompt_field.clone())
        .#mf_name(self.#mf_name.clone())
        .#mcf_name(self.#mcf_name.clone())
    };

    let builder_chaining_std_fields = standard_fields.iter().map(|f| {
        let name = &f.ident;

        if is_optional(&f) {
            return quote! {
                .#name(self.#name.clone().ok_or(anyhow::anyhow!(concat!(stringify!(#name), " is not set")))?)
            };
        };

        return quote! {
            .#name(self.#name.clone())
        };
    });

    let expanded = quote! {
        #[derive(serde::Serialize, Clone)]
        pub struct #builder_ident {
            #(#template_fields),*
        }

        impl #builder_ident {
            #(#builder_methods)*

            pub fn build(&self) -> anyhow::Result<#struct_name> {
                Ok(#struct_name {
                    #(#build_fields,)*
                })
            }
        }

        impl Promptize<#struct_name> for #struct_name {
            fn get_model(&self) -> String {
                self.#mf_name.clone()
            }

            fn build_prompt(
                self 
            ) -> anyhow::Result<(std::vec::Vec<std::vec::Vec<tiktoken_rs::ChatCompletionRequestMessage>>, #struct_name)> {
                let model_str = &self.#mf_name.clone();
                let token_limit = promptize_internals::get_context_size(model_str);

                println!("model: {} | token limit: {}", model_str, token_limit);

                let prompt_string = serde_json::to_string(&self)?;
                let total_prompt_tokens: i32 = self.get_prompt_tokens(model_str, &prompt_string)?.try_into()?;
                let chunk_field = self.#cf_name.clone();

                let prompts = match total_prompt_tokens > token_limit {
                    true => self.build_chunked_prompt(model_str, token_limit, self.#mcf_name, chunk_field, prompt_string, total_prompt_tokens)?,
                    false => self.build_single_prompt()
                };
                
                Ok((prompts, self))
            }

            fn reassemble(
                &self, 
                chunk_summary: String, 
            ) -> anyhow::Result<(std::vec::Vec<std::vec::Vec<tiktoken_rs::ChatCompletionRequestMessage>>, #struct_name)> {
                let prompt = #struct_name::builder()
                    #builder_chaining_hardcoded
                    #(#builder_chaining_std_fields)*
                    .#cf_name(chunk_summary)
                    .build()?;

                return Ok(prompt.build_prompt()?);
            }
        }

        impl #struct_name {
            pub fn builder() -> #builder_ident {
                #builder_ident {
                    #(#fields_empty,)*
                }
            }

            fn create_system_prompt(&self) -> tiktoken_rs::ChatCompletionRequestMessage {
                return tiktoken_rs::ChatCompletionRequestMessage {
                    role: "system".to_string(),
                    content: self.#system_prompt_field.clone(),
                    name: None
                };
            }

            fn create_user_prompt(&self, chunkable_field_content: String) -> tiktoken_rs::ChatCompletionRequestMessage {
                return tiktoken_rs::ChatCompletionRequestMessage {
                    role: "user".to_string(),
                    content: format!("{}, {}, {}: {}", self.#user_prompt_field.clone(), #std_fields_fmt_template, stringify!(#cf_name), chunkable_field_content),
                    name: None
                };
            }

            fn build_chunked_prompt(
                &self, 
                model: &str, 
                token_limit: i32,
                maximum_chunk_count: i32,
                chunk_field: #cf_type, 
                prompt_string: String,
                total_prompt_tokens: i32
            ) -> anyhow::Result<std::vec::Vec<std::vec::Vec<tiktoken_rs::ChatCompletionRequestMessage>>> {
                let chunkable_field_tokens: i32 = self.get_prompt_tokens(model, &chunk_field)?.try_into()?;
                let chunk_size_chars = self.get_chunk_size_chars(&prompt_string, chunkable_field_tokens, token_limit, maximum_chunk_count, total_prompt_tokens)?; 
                let string_chunks = self.chunk_string(prompt_string, chunk_size_chars);

                let prompts: Vec<Vec<tiktoken_rs::ChatCompletionRequestMessage>> = string_chunks
                    .iter()
                    .map(|c| {
                        let mut prompt = vec![];
                        let system = self.create_system_prompt();
                        let user = self.create_user_prompt(c.clone());

                        prompt.push(system);
                        prompt.push(user);
                        prompt
                    })
                    .collect();

                Ok(prompts)
            }

            fn build_single_prompt(&self) -> std::vec::Vec<std::vec::Vec<tiktoken_rs::ChatCompletionRequestMessage>> {
                let system = self.create_system_prompt();
                let user = self.create_user_prompt(self.#cf_name.clone());

                vec![vec![system, user]]
            }
            /// Gets the optimal chunk size in Tokens
            fn get_chunk_size_tokens(&self, total: i32, limit: i32) -> i32 {
                let num_chunks = (total as f64 / limit as f64).ceil() as i32;
                let base_chunk_size = total / num_chunks;

                let mut chunk_size = base_chunk_size;
                let mut num_chunks = num_chunks;

                while chunk_size > limit {
                    num_chunks += 1; 
                    chunk_size = total / num_chunks;
                }
                chunk_size
            }

            /// Gets the chunk sizes in chars so that text can be broken up on chars and not tokens
            fn get_chunk_size_chars(
                &self,
                prompt_string: &String,
                chunkable_field_tokens: i32, 
                token_limit: i32, 
                maximum_chunk_count: i32, 
                total_prompt_tokens: i32
            ) -> anyhow::Result<i32> {
                // this represents the tokens left after non chunkable fields are removed
                // since non chunkable fields cannot be changed, this is our "real" limit
                let chunkable_tokens_remaining = token_limit - (total_prompt_tokens - chunkable_field_tokens);
                
                let chunk_size_tokens = self.get_chunk_size_tokens(chunkable_field_tokens, chunkable_tokens_remaining);
                let num_chunks: i32 = (chunkable_field_tokens as f64 / chunk_size_tokens as f64) as i32;

                if num_chunks > maximum_chunk_count {
                    anyhow::bail!("Number of chunks exceeds the maximum allowed chunk count");
                }

                let chunk_ratio = chunk_size_tokens as f64 / chunkable_field_tokens as f64;
                let total_chars = prompt_string.chars().collect::<Vec<char>>().len();
                let chunk_size_chars:i32 = (chunk_ratio * total_chars as f64).ceil() as i32;
                Ok(chunk_size_chars)
            }

            /// Chunks up a string based on chunk_size which is number of chars not tokens
            fn chunk_string(&self, prompt: String, chunk_size: i32) -> std::vec::Vec<String> {
                let chunks = prompt
                    .chars()
                    .collect::<std::vec::Vec<char>>()
                    .chunks(chunk_size as usize)
                    .map(|c| c.iter().collect::<String>())
                    .collect::<std::vec::Vec<String>>();

                chunks
            }

            fn get_prompt_tokens(&self, model: &str, prompt: &str) -> anyhow::Result<usize> {
                let bpe = tiktoken_rs::get_bpe_from_model(model)?;
                let prompt_tokens = bpe.encode_with_special_tokens(prompt).len();
                Ok(prompt_tokens)
            }

        }

    };

    proc_macro::TokenStream::from(expanded)
}

fn get_fields(input_ast: &syn::DeriveInput, struct_name: &syn::Ident) -> Result<syn::punctuated::Punctuated<syn::Field, syn::token::Comma>, proc_macro::TokenStream> {
    let fields = if let syn::Data::Struct(syn::DataStruct { 
        fields: syn::Fields::Named(syn::FieldsNamed { 
            ref named, 
            ..
        }), 
        ..
    }) = input_ast.data {
        named 
    } else {
        let error = syn::Error::new(struct_name.span(), "Proc Macro only supports Structs");
        return Err(error.to_compile_error().into());
    };

    Ok(fields.to_owned())
}

/// This returns all non-prompt fields and all non chunkable fields
fn get_standard_fields(fields: syn::punctuated::Punctuated<syn::Field, syn::token::Comma>) -> Vec<syn::Field> {
    let fields = fields
        .into_iter()
        .filter(|field| {
            if field.attrs.len() == 0 &&
                field.ident.clone().unwrap() != USER_PROMPT_FIELD_NAME &&
                field.ident.clone().unwrap() != SYSTEM_PROMPT_FIELD_NAME 
            {
                return true;
            }
            return false;
        })
        .collect::<Vec<syn::Field>>();

    return fields;
}

fn get_field_ident_by_name(struct_name: &syn::Ident, fields: syn::punctuated::Punctuated<syn::Field, syn::token::Comma>, field_name: &str) -> Result<Option<syn::Ident>, proc_macro::TokenStream> {
    let fields = fields
        .into_iter()
        .filter(|field| {
            if field.ident.clone().unwrap() == field_name
            {
                return true;
            }
            return false;
        })
        .collect::<Vec<syn::Field>>();

    if fields.len() == 0 {
        let error = syn::Error::new(struct_name.span(), format!("Field: {} not found", field_name));
        return Err(error.to_compile_error().into());
    }

    if fields.len() > 1 {
        let error = syn::Error::new(struct_name.span(), format!("Field: {} found more than once", field_name));
        return Err(error.to_compile_error().into());
    }

    return Ok(fields.first().unwrap().to_owned().ident);
}

fn validate_fields(struct_name: &syn::Ident, fields: syn::punctuated::Punctuated<syn::Field, syn::token::Comma>) -> Result<bool, proc_macro::TokenStream> {
    let has_user = fields.iter().any(|f| { f.ident.clone().unwrap() == USER_PROMPT_FIELD_NAME });
    let has_system = fields.iter().any(|f| { f.ident.clone().unwrap() == SYSTEM_PROMPT_FIELD_NAME });

    if !has_user || !has_system {
        let error = syn::Error::new(struct_name.span(), format!("{} and {} fields are required to be defined on struct", USER_PROMPT_FIELD_NAME, SYSTEM_PROMPT_FIELD_NAME));
        return Err(error.to_compile_error().into());
    }

    Ok(true)
}

fn get_and_val_field_by_attribute(
    struct_name: &syn::Ident,
    attr_path_ident:&str,
    field_type:&str,
    fields: syn::punctuated::Punctuated<syn::Field, syn::token::Comma>
) -> Result<syn::Field, proc_macro::TokenStream> {
    let target_fields = extract_fields_by_attribute(attr_path_ident, fields);

    if target_fields.len() > 1 {
        let error = syn::Error::new(struct_name.span(), format!("{} attribute is only supported on one field at a time", attr_path_ident));
        return Err(error.to_compile_error().into());
    }
    let target_field = target_fields.first().unwrap();

    // Ensure chunkable field matches field_type parameter
    match &target_field.ty {
        syn::Type::Path(p) => {
            let mut found_field_type = p.path.segments.first().unwrap().ident.to_string();
            
            // TO DO: This should be more generic but it meets my particular needs for now
            if found_field_type == "Option" {
                let inner_type = get_option_inner_type(target_field);
                found_field_type = format!("Option<{}>", inner_type);
            }

            let is_field_type = found_field_type == field_type;
            
            if !is_field_type {
                let error = syn::Error::new(struct_name.span(), format!("Only {} type supported for {} fields", field_type, attr_path_ident));
                return Err(error.to_compile_error().into());
            }
        },
        _ => panic!("only syn::Type::Path supported on Field Type")
    };

    Ok(target_field.to_owned())
}

fn extract_fields_by_attribute(attr_path_ident: &str, fields: syn::punctuated::Punctuated<syn::Field, syn::token::Comma>) -> Vec<syn::Field> {
    let chunkable = fields
        .into_iter()
        .filter(|field| {
            field.attrs.iter().any(|attr| {
                match attr.meta.clone() {
                    syn::Meta::Path(path) => path.is_ident(attr_path_ident),
                    _ => false,
                }
            })
        })
        .collect::<Vec<syn::Field>>();

    return chunkable;
}

fn get_option_inner_type(field: &syn::Field) -> syn::Ident {
    match &field.ty {
        syn::Type::Path(t_path) => {
            let segments = &t_path.path.segments;
            match &segments[0].arguments {
                syn::PathArguments::AngleBracketed(af) => {
                    let first_arg = af.args.first().unwrap();
                    match first_arg {
                        syn::GenericArgument::Type(arg) => {
                            match arg {
                                syn::Type::Path(p) => {
                                    return p.path.get_ident().unwrap().to_owned();
                                },
                                _ => unimplemented!("Arg not of Type::Path")
                            }
                        },
                        _ => unimplemented!("Path Argument not GenericArgument::Type")
                    }
                },
                _ => unimplemented!("PathArgument not AngleBracketed")
            }
        },
        _ => unimplemented!("Type not a path")
    }
}

fn is_optional(field: &syn::Field) -> bool {
    if let syn::Type::Path(t_path) = &field.ty {
        let segments = &t_path.path.segments;
        if segments.len() == 1 && segments[0].ident == "Option" {
            return true
        }
        return false
    } else {
        panic!("unsupported type path")
    }
}
