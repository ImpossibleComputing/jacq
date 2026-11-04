//! Token definitions and the lexer for the Jacquard DSL.
//!
//! Keywords are not distinguished at the lexical level — every bareword is an `Ident`, and the
//! parser decides whether a given identifier introduces a block (`config`, `weights`, …) or
//! names an operation. This keeps the token set small and lets operation names like `rmsnorm`
//! or `silu` be ordinary identifiers.

use logos::Logos;

/// A single lexical token.
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n\f]+")] // whitespace
#[logos(skip r"//[^\n]*")] // line comments
#[logos(skip r"/\*([^*]|\*[^/])*\*/")] // block comments
pub enum Token {
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    // Floats must out-prioritize ints so `10000.0` and `1e-5` lex as one float. A fractional digit
    // is required after the dot (`[0-9]+` not `[0-9]*`) so a range like `0..2` lexes as
    // Int `..` Int rather than the float `0.` swallowing the first dot.
    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+", |lex| lex.slice().to_string(), priority = 3)]
    Float(String),

    #[regex(r"[0-9]+", |lex| lex.slice().to_string(), priority = 2)]
    Int(String),

    #[regex(r#""[^"]*""#, |lex| { let s = lex.slice(); s[1..s.len()-1].to_string() })]
    Str(String),

    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("..")]
    DotDot,
    #[token(":")]
    Colon,
    #[token("=")]
    Eq,
    #[token("@")]
    At,
    #[token("+")]
    Plus,
    #[token("*")]
    Star,
    #[token("->")]
    Arrow,
    #[token(">=")]
    Ge,
    #[token("<=")]
    Le,
    #[token(">")]
    Gt,
    #[token("<")]
    Lt,
    #[token("-")]
    Minus,
}

/// A token paired with its source byte range, for diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned {
    pub token: Token,
    pub span: std::ops::Range<usize>,
}

/// Lex `src` into a vector of spanned tokens, or return the byte offset of the first
/// unrecognized character.
pub fn lex(src: &str) -> Result<Vec<Spanned>, std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut lexer = Token::lexer(src);
    while let Some(result) = lexer.next() {
        match result {
            Ok(token) => out.push(Spanned {
                token,
                span: lexer.span(),
            }),
            Err(_) => return Err(lexer.span()),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Token> {
        lex(src).unwrap().into_iter().map(|s| s.token).collect()
    }

    #[test]
    fn lexes_idents_numbers_strings() {
        let toks = kinds(r#"hidden = 2048 rope_base = 10000.0 eps = 1e-5 src = "model""#);
        assert_eq!(toks[0], Token::Ident("hidden".into()));
        assert_eq!(toks[2], Token::Int("2048".into()));
        assert_eq!(toks[5], Token::Float("10000.0".into()));
        assert_eq!(toks[8], Token::Float("1e-5".into()));
        assert_eq!(toks[11], Token::Str("model".into()));
    }

    #[test]
    fn lexes_layer_range_not_as_float() {
        // `0..2` must be Int, DotDot, Int — the float rule must not swallow `0.`.
        let toks = kinds("prune layers 0..2, 20..21");
        assert_eq!(toks[2], Token::Int("0".into()));
        assert_eq!(toks[3], Token::DotDot);
        assert_eq!(toks[4], Token::Int("2".into()));
        assert_eq!(toks[5], Token::Comma);
        assert_eq!(toks[6], Token::Int("20".into()));
        assert_eq!(toks[7], Token::DotDot);
        assert_eq!(toks[8], Token::Int("21".into()));
    }

    #[test]
    fn lexes_arrow_and_braces_and_skips_comments() {
        let toks = kinds("// a comment\npipeline { embed -> lm_head }");
        assert_eq!(toks[0], Token::Ident("pipeline".into()));
        assert_eq!(toks[1], Token::LBrace);
        assert!(toks.contains(&Token::Arrow));
    }
}
