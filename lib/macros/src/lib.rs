#![feature(proc_macro_value)]

extern crate proc_macro;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

#[proc_macro_attribute]
pub fn heavy_computation(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ret = TokenStream::new();
    ret.extend([
        TokenTree::Punct(Punct::new('#', Spacing::Joint)),
        TokenTree::Group(Group::new(Delimiter::Bracket, {
            let mut token_stream = TokenStream::new();
            token_stream.extend([
                TokenTree::Ident(Ident::new("must_use", Span::call_site())),
                TokenTree::Punct(Punct::new('=', Spacing::Alone)),
                TokenTree::Literal(Literal::string(
                    "discarding the result of a potentially heavy computation is wasteful",
                )),
            ]);
            token_stream
        })),
    ]);
    ret.extend(item);
    ret
}

#[proc_macro_attribute]
pub fn efficient_alternatives(args: TokenStream, item: TokenStream) -> TokenStream {
    fn error(message: &str) -> TokenStream {
        [
            TokenTree::Ident(Ident::new("compile_error", Span::call_site())),
            TokenTree::Punct(Punct::new('!', Spacing::Joint)),
            TokenTree::Group(Group::new(
                Delimiter::Parenthesis,
                [TokenTree::Literal(Literal::string(message))].into_iter().collect(),
            )),
            TokenTree::Punct(Punct::new(';', Spacing::Alone)),
        ]
        .into_iter()
        .collect()
    }

    let mut message = String::from("consider using ");
    let mut iterator = args.clone().into_iter();
    while let Some(token_tree) = iterator.next() {
        match &token_tree {
            TokenTree::Literal(literal) if let Ok(argument) = literal.str_value() => {
                message.reserve(argument.len() + 2);
                message.push('`');
                message.push_str(argument.as_str());
                message.push('`');
            }
            _ => return error("only string arguments are allowed"),
        }
        match iterator.next() {
            Some(token_tree) => match token_tree {
                TokenTree::Punct(delimiter) if delimiter.as_char() == ',' => {
                    message.push_str(" or ");
                }
                _ => return error("invalid delimiter"),
            },
            None => break,
        }
    }
    message.push_str(" as a more efficient alternative");
    let message = TokenTree::Literal(Literal::string(message.as_str()));

    let mut ret = TokenStream::from_iter(
        [
            TokenTree::Punct(Punct::new('#', Spacing::Joint)),
            TokenTree::Group(Group::new(
                Delimiter::Bracket,
                [
                    TokenTree::Ident(Ident::new("deprecated", Span::call_site())),
                    TokenTree::Punct(Punct::new('=', Spacing::Alone)),
                    message,
                ]
                .into_iter()
                .collect(),
            )),
        ]
        .into_iter(),
    );
    ret.extend(item);
    ret
}
