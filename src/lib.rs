use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse_macro_input, DeriveInput
};

const USER_PROMPT_FIELD_NAME: &str = "user_prompt";
const SYSTEM_PROMPT_FIELD_NAME: &str = "system_prompt";
const CHUNKABLE_ATTRIBUTE_NAME: &str = "chunkable";

/// Allows fields of a struct to be marked as Chunkable as to denote that they are able to be
/// broken up into chunks when being fed to an LLM to fit inside of a context window.
#[proc_macro_derive(Promptize, attributes(chunkable, prompt))]
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
    let chunk_field = match get_chunkable_fields(struct_name, CHUNKABLE_ATTRIBUTE_NAME, fields.clone()) {
        Ok(chunk_field) => chunk_field,
        Err(err) => return err,
    };
    let cf_name = &chunk_field.ident;
    let cf_type = &chunk_field.ty;

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
            let name = quote! {
                #name:{} // TODO: Figure out why this is outputting foo : "bar" (spaces)
            };
            return name;
        });

    let std_fields_fmt_template = quote! {
        format!(stringify!(#(#fmt_string_braces)*), #(self.#standard_fields_template.clone().unwrap()),*)
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

    let builder_methods = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;

        if is_optional(&f) {
            // extract root type
            let option_type = get_option_type(&f);
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

    let expanded = quote! {
        #[derive(serde::Serialize, Clone)]
        struct #builder_ident {
            #(#template_fields),*
        }
        
        impl #builder_ident {
            #(#builder_methods)*

            pub fn build_prompt(
                &self, 
                model: &str, 
                token_limit: i32,
                maximum_chunk_count: i32
            ) -> Result<
                std::vec::Vec<std::vec::Vec<tiktoken_rs::ChatCompletionRequestMessage>>, 
                std::boxed::Box<dyn std::error::Error>
            > {
                let prompt_string = serde_json::to_string(&self)?;
                let total_prompt_tokens: i32 = get_prompt_tokens(model, &prompt_string)?.try_into()?;
                let chunk_field = self.#cf_name.clone().ok_or(concat!(stringify!(#struct_name), " is not set"))?;

                let prompts = match total_prompt_tokens > token_limit {
                    true => self.build_chunked_prompt(model, token_limit, maximum_chunk_count, chunk_field, prompt_string, total_prompt_tokens)?,
                    false => self.build_single_prompt()
                };
                
                Ok(prompts)
            }
            
            fn create_system_prompt(&self) -> tiktoken_rs::ChatCompletionRequestMessage {
                return tiktoken_rs::ChatCompletionRequestMessage {
                    role: "system".to_string(),
                    content: self.#system_prompt_field.clone().unwrap(),
                    name: None
                };
            }

            fn create_user_prompt(&self, chunkable_field_content: String) -> tiktoken_rs::ChatCompletionRequestMessage {
                return tiktoken_rs::ChatCompletionRequestMessage {
                    role: "user".to_string(),
                    content: format!("{}, {}, {}: {}", self.#user_prompt_field.clone().unwrap(), #std_fields_fmt_template, stringify!(#cf_name), chunkable_field_content),
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
            ) -> Result<std::vec::Vec<std::vec::Vec<tiktoken_rs::ChatCompletionRequestMessage>>, Box<dyn std::error::Error>> {
                let chunkable_field_tokens: i32 = get_prompt_tokens(model, &chunk_field)?.try_into()?;
                let chunk_size_chars = get_chunk_size_chars(&prompt_string, chunkable_field_tokens, token_limit, maximum_chunk_count, total_prompt_tokens)?; 
                let string_chunks = chunk_string(prompt_string, chunk_size_chars);

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
                let user = self.create_user_prompt(self.#cf_name.clone().unwrap());

                vec![vec![system, user]]
            }
        }

        impl #struct_name {
            fn builder() -> #builder_ident {
                #builder_ident {
                    #(#fields_empty,)*
                }
            }
        }

        /// Gets the optimal chunk size in Tokens
        fn get_chunk_size_tokens(total: i32, limit: i32) -> i32 {
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
            prompt_string: &String,
            chunkable_field_tokens: i32, 
            token_limit: i32, 
            maximum_chunk_count: i32, 
            total_prompt_tokens: i32
        ) -> Result<i32, Box<dyn std::error::Error>> {
            // this represents the tokens left after non chunkable fields are removed
            // since non chunkable fields cannot be changed, this is our "real" limit
            let chunkable_tokens_remaining = token_limit - (total_prompt_tokens - chunkable_field_tokens);
            
            let chunk_size_tokens = get_chunk_size_tokens(chunkable_field_tokens, chunkable_tokens_remaining);
            let num_chunks: i32 = (chunkable_field_tokens as f64 / chunk_size_tokens as f64) as i32;

            if num_chunks > maximum_chunk_count {
                return Err("Number of chunks exceeds the maximum allowed chunk count".into());
            }

            let chunk_ratio = chunk_size_tokens as f64 / chunkable_field_tokens as f64;
            let total_chars = prompt_string.chars().collect::<Vec<char>>().len();
            let chunk_size_chars:i32 = (chunk_ratio * total_chars as f64).ceil() as i32;
            Ok(chunk_size_chars)
        }
 
        /// Chunks up a string based on chunk_size which is number of chars not tokens
        fn chunk_string(prompt: String, chunk_size: i32) -> std::vec::Vec<String> {
            let chunks = prompt
                .chars()
                .collect::<std::vec::Vec<char>>()
                .chunks(chunk_size as usize)
                .map(|c| c.iter().collect::<String>())
                .collect::<std::vec::Vec<String>>();

            chunks
        }

        fn get_prompt_tokens(model: &str, prompt: &str) -> Result<usize, std::boxed::Box<dyn std::error::Error>> {
            let bpe = tiktoken_rs::get_bpe_from_model(model)?;
            let prompt_tokens = bpe.encode_with_special_tokens(prompt).len();
            Ok(prompt_tokens)
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
        let error = syn::Error::new(struct_name.span(), format!("{} and {} system_prompt fields are required to be defined on struct", USER_PROMPT_FIELD_NAME, SYSTEM_PROMPT_FIELD_NAME));
        return Err(error.to_compile_error().into());
    }
    Ok(true)
}

fn get_chunkable_fields(
    struct_name: &syn::Ident,
    attr_path_ident:&str,
    fields: syn::punctuated::Punctuated<syn::Field, syn::token::Comma>
) -> Result<syn::Field, proc_macro::TokenStream> {
    let chunkable_fields = extract_fields_by_attribute(attr_path_ident, fields);

    if chunkable_fields.len() > 1 {
        let error = syn::Error::new(struct_name.span(), "chunkable attribute is only supported on one field at a time");
        return Err(error.to_compile_error().into());
    }

    let chunk_field = chunkable_fields.first().unwrap();

    // Ensure chunkable field is of type string
    match &chunk_field.ty {
        syn::Type::Path(p) => {
            let is_string = p.path.segments.first().unwrap().ident == "String";
            
            if !is_string {
                let error = syn::Error::new(struct_name.span(), "Only String type supported for chunkable fields");
                return Err(error.to_compile_error().into());
            }
        },
        _ => panic!("only syn::Type::Path supported on Field Type")
    };

    Ok(chunk_field.to_owned())
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

fn get_option_type(field: &syn::Field) -> syn::Ident {
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
