//! Lexing and parsing for the Jacquard DSL.
//!
//! Jacquard is two languages sharing one file format: a high-level `model` DSL (compiled to a
//! CUDA inference runtime) and a low-level `kernel` DSL (compiled to individual CUDA kernels),
//! plus top-level `strategy`, `package`, `surgery`, and `experiment` definitions.
//!
//! Phase 0 covers the model DSL front-end: lexer ([`token`]), AST ([`ast`]), and a
//! recursive-descent parser ([`parser`]). Other constructs land in later phases.

pub mod ast;
pub mod emit;
pub mod parser;
pub mod token;

pub use ast::{
    ExpKind, Experiment, Kernel, LayerRange, Metric, Model, Subject, SubjectSrc, Surgery,
    SurgeryOp, SweepAxis,
};
pub use emit::model_to_jac;
pub use parser::{parse_experiment, parse_kernel, parse_model, parse_surgery, ParseError};

/// The top-level construct a `.jac` source item can declare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    Model,
    Kernel,
    Strategy,
    Package,
    Surgery,
    Experiment,
}

impl ItemKind {
    /// Map a leading keyword to its item kind, if it introduces a top-level construct.
    pub fn from_keyword(word: &str) -> Option<ItemKind> {
        Some(match word {
            "model" => ItemKind::Model,
            "kernel" => ItemKind::Kernel,
            "strategy" => ItemKind::Strategy,
            "package" => ItemKind::Package,
            "surgery" => ItemKind::Surgery,
            "experiment" => ItemKind::Experiment,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_top_level_keywords() {
        assert_eq!(ItemKind::from_keyword("model"), Some(ItemKind::Model));
        assert_eq!(ItemKind::from_keyword("kernel"), Some(ItemKind::Kernel));
        assert_eq!(
            ItemKind::from_keyword("experiment"),
            Some(ItemKind::Experiment)
        );
        assert_eq!(ItemKind::from_keyword("hidden"), None);
    }
}
