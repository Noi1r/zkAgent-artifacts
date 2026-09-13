use anyhow::Result;
use quote::ToTokens;
use syn::{parse_file, FnArg, Item, Pat, ReturnType, Type};

#[derive(Debug, Clone)]
pub struct RustFunction {
    pub name: String,
    pub params: Vec<(String, String)>,
    pub return_type: String,
    pub is_public: bool,
    pub body: String,
}

pub fn parse_rust_functions(rust_code: &str) -> Result<Vec<RustFunction>> {
    let file = parse_file(rust_code)?;
    let mut functions = Vec::new();
    for item in file.items {
        if let Item::Fn(func) = item {
            let name = func.sig.ident.to_string();
            if name.starts_with("test_")
                || name.starts_with('_')
                || !func.sig.generics.params.is_empty()
            {
                continue;
            }
            let is_public = matches!(func.vis, syn::Visibility::Public(_));
            let params = func
                .sig
                .inputs
                .iter()
                .filter_map(|arg| {
                    let FnArg::Typed(arg) = arg else {
                        return None;
                    };
                    let Pat::Ident(name) = &*arg.pat else {
                        return None;
                    };
                    Some((name.ident.to_string(), format_type(&arg.ty)))
                })
                .collect();
            let return_type = match &func.sig.output {
                ReturnType::Default => "()".to_string(),
                ReturnType::Type(_, ty) => format_type(ty),
            };
            functions.push(RustFunction {
                name,
                params,
                return_type,
                is_public,
                body: func.block.to_token_stream().to_string(),
            });
        }
    }
    Ok(functions)
}

pub(crate) fn format_type(ty: &Type) -> String {
    let file: syn::File = syn::parse_quote!(type __Py2ZkvmType = #ty;);
    let declaration = prettyplease::unparse(&file);
    declaration
        .split_once('=')
        .unwrap()
        .1
        .trim()
        .strip_suffix(';')
        .unwrap()
        .trim()
        .to_string()
}

pub(crate) fn input_type_for_zkvm(typ: &str) -> Result<Type> {
    Ok(owned_type(&syn::parse_str(typ)?))
}

fn owned_type(ty: &Type) -> Type {
    match ty {
        Type::Reference(reference) => owned_type(&reference.elem),
        Type::Slice(slice) => {
            let element = &slice.elem;
            syn::parse_quote!(Vec<#element>)
        }
        Type::Path(path) if path.path.is_ident("str") => syn::parse_quote!(String),
        Type::Paren(paren) => owned_type(&paren.elem),
        Type::Group(group) => owned_type(&group.elem),
        _ => ty.clone(),
    }
}

fn wrapped_type(ty: &Type, container: &str) -> Option<Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    let names: Vec<_> = path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    let recognized = match container {
        "Result" => {
            names == ["Result"]
                || names == ["std", "result", "Result"]
                || names == ["core", "result", "Result"]
        }
        "Box" => {
            names == ["Box"]
                || names == ["std", "boxed", "Box"]
                || names == ["alloc", "boxed", "Box"]
        }
        _ => false,
    };
    if !recognized {
        return None;
    }
    let arguments = &path.path.segments.last()?.arguments;
    let syn::PathArguments::AngleBracketed(arguments) = arguments else {
        return None;
    };
    match arguments.args.first()? {
        syn::GenericArgument::Type(ty) => Some(ty.clone()),
        _ => None,
    }
}

pub(crate) fn adapt_output(typ: &str, mut expression: syn::Expr) -> Result<(Type, syn::Expr)> {
    let mut ty: Type = syn::parse_str(typ)?;
    loop {
        if let Some(inner) = wrapped_type(&ty, "Result") {
            expression = syn::parse_quote! {
                (#expression).unwrap_or_else(|_| panic!("transpiled function returned an error"))
            };
            ty = inner;
        } else if let Some(inner) = wrapped_type(&ty, "Box") {
            expression = syn::parse_quote! { *(#expression) };
            ty = inner;
        } else {
            match ty {
                Type::Paren(ref paren) => ty = (*paren.elem).clone(),
                Type::Group(ref group) => ty = (*group.elem).clone(),
                _ => break,
            }
        }
    }
    Ok((owned_type(&ty), expression))
}

pub(crate) fn output_type_for_zkvm(typ: &str) -> Result<Type> {
    adapt_output(typ, syn::parse_quote!(__py2zkvm_output)).map(|(ty, _)| ty)
}

pub fn check_zkvm_compatibility(rust_type: &str) -> bool {
    let typ = rust_type.trim();

    if contains_generic_param(typ) {
        return false;
    }

    let primitives = [
        "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "f32", "f64", "bool",
        "()", "usize", "isize",
    ];

    if primitives.contains(&typ) {
        return true;
    }

    if typ == "String" || typ == "&str" || typ == "& str" {
        return true;
    }

    if typ.starts_with("Vec<") || typ.starts_with("& Vec<") {
        let inner = extract_generic_type(typ, "Vec");
        return check_zkvm_compatibility(&inner);
    }

    if typ.starts_with("Option<") {
        let inner = extract_generic_type(typ, "Option");
        return check_zkvm_compatibility(&inner);
    }

    if typ.starts_with("Result<") {
        return false;
    }

    if typ.starts_with('(') && typ.ends_with(')') {
        let inner = &typ[1..typ.len() - 1];
        return inner.split(',').all(|t| check_zkvm_compatibility(t.trim()));
    }

    if typ.starts_with('[') && typ.contains(';') {
        let inner = &typ[1..typ.find(';').unwrap()];
        return check_zkvm_compatibility(inner.trim());
    }

    if typ.starts_with('&') {
        let inner = typ.trim_start_matches('&').trim_start_matches("mut").trim();
        return check_zkvm_compatibility(inner);
    }

    true
}

pub fn contains_generic_param(typ: &str) -> bool {
    typ.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .any(|token| token.len() == 1 && token.chars().all(|c| c.is_ascii_uppercase()))
}

fn extract_generic_type(typ: &str, generic_name: &str) -> String {
    let start = typ.find('<').unwrap_or(0) + 1;
    let end = typ.rfind('>').unwrap_or(typ.len());
    typ[start..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zkvm_compatibility() {
        assert!(check_zkvm_compatibility("i32"));
        assert!(check_zkvm_compatibility("Vec<u8>"));
        assert!(check_zkvm_compatibility("(i32, i32)"));
        assert!(!check_zkvm_compatibility("Result<i32, Error>"));
    }

    #[test]
    fn test_simplify_type() {
        assert_eq!(
            format_type(&input_type_for_zkvm("&Vec<i32>").unwrap()),
            "Vec<i32>"
        );
        assert_eq!(
            format_type(&input_type_for_zkvm("&mut i32").unwrap()),
            "i32"
        );
    }

    #[test]
    fn regression_output_adaptation_handles_spacing_qualified_paths_and_nesting() {
        let cases = [
            (
                "Result < (i64, Vec<i32>), Box < dyn std::error::Error > >",
                "(i64, Vec<i32>)",
                1,
            ),
            (
                "std::result::Result<Vec<(i64, bool)>, String>",
                "Vec<(i64, bool)>",
                1,
            ),
            (
                "Box<core::result::Result<Result<(), String>, String>>",
                "()",
                2,
            ),
            (
                "Result<Vec<Result<i32, String>>, String>",
                "Vec<Result<i32, String>>",
                1,
            ),
        ];
        for (source, expected, unwraps) in cases {
            let (ty, expression) = adapt_output(source, syn::parse_quote!(compute())).unwrap();
            assert_eq!(format_type(&ty), expected);
            assert_eq!(
                expression
                    .to_token_stream()
                    .to_string()
                    .matches("unwrap_or_else")
                    .count(),
                unwraps
            );
        }
    }

    #[test]
    fn regression_type_formatting_keeps_dyn_keyword_separate() {
        let functions = parse_rust_functions(
            "pub fn f() -> Result < i64, Box < dyn std::error::Error > > { Ok(11) }",
        )
        .unwrap();
        assert_eq!(
            functions[0].return_type,
            "Result<i64, Box<dyn std::error::Error>>"
        );
        assert!(parse_rust_functions("pub fn f(").is_err());
    }
}
