//! Recursive-descent parser for the Jacquard **model** DSL.
//!
//! Parses a single `model { config, weights, layer, pipeline }` definition into [`ast::Model`].
//! Errors carry a source span so the CLI can render a caret pointing at the problem.

use crate::ast::{
    BinOp, CVal, CmpOp, Constraint, ExpKind, Experiment, Expr, FusionDirective, IndexExpr,
    IterSpec, KStmt, Kernel, Layer, LayerRange, LoraSpec, Metric, Model, Phase, StateDecl, Stmt,
    Subject, SubjectSrc, Surgery, SurgeryOp, SweepAxis, TensorDecl, Value,
};
use crate::token::{lex, Token};

/// A parse failure with the byte range it occurred at.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub span: std::ops::Range<usize>,
}

struct Parser {
    toks: Vec<crate::token::Spanned>,
    pos: usize,
    /// End offset of the source, used for "unexpected end of input" spans.
    end: usize,
}

/// Parse a `.jac` source string containing one model definition.
pub fn parse_model(src: &str) -> Result<Model, ParseError> {
    let toks = lex(src).map_err(|span| ParseError {
        message: "unrecognized character".to_string(),
        span,
    })?;
    let mut p = Parser {
        toks,
        pos: 0,
        end: src.len(),
    };
    let model = p.parse_model()?;
    if let Some(s) = p.peek_spanned() {
        return Err(p.err_at(s.span.clone(), "unexpected trailing input after model"));
    }
    Ok(model)
}

/// Parse a `.jac` source string containing one `kernel NAME { ... }` definition (kernel DSL).
pub fn parse_kernel(src: &str) -> Result<Kernel, ParseError> {
    let toks = lex(src).map_err(|span| ParseError {
        message: "unrecognized character".to_string(),
        span,
    })?;
    let mut p = Parser {
        toks,
        pos: 0,
        end: src.len(),
    };
    let kernel = p.parse_kernel()?;
    if let Some(s) = p.peek_spanned() {
        return Err(p.err_at(s.span.clone(), "unexpected trailing input after kernel"));
    }
    Ok(kernel)
}

/// Parse a `.jac` source string containing one `surgery NAME from "PATH" { ops }` definition.
pub fn parse_surgery(src: &str) -> Result<Surgery, ParseError> {
    let toks = lex(src).map_err(|span| ParseError {
        message: "unrecognized character".to_string(),
        span,
    })?;
    let mut p = Parser {
        toks,
        pos: 0,
        end: src.len(),
    };
    let surgery = p.parse_surgery()?;
    if let Some(s) = p.peek_spanned() {
        return Err(p.err_at(s.span.clone(), "unexpected trailing input after surgery"));
    }
    Ok(surgery)
}

/// Parse a `.jac` source string containing one `experiment NAME { ... }` definition.
pub fn parse_experiment(src: &str) -> Result<Experiment, ParseError> {
    let toks = lex(src).map_err(|span| ParseError {
        message: "unrecognized character".to_string(),
        span,
    })?;
    let mut p = Parser {
        toks,
        pos: 0,
        end: src.len(),
    };
    let e = p.parse_experiment()?;
    if let Some(s) = p.peek_spanned() {
        return Err(p.err_at(s.span.clone(), "unexpected trailing input after experiment"));
    }
    Ok(e)
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.toks.get(self.pos).map(|s| &s.token)
    }
    fn peek_spanned(&self) -> Option<&crate::token::Spanned> {
        self.toks.get(self.pos)
    }
    fn span_here(&self) -> std::ops::Range<usize> {
        self.toks
            .get(self.pos)
            .map(|s| s.span.clone())
            .unwrap_or(self.end..self.end)
    }
    fn bump(&mut self) -> Option<Token> {
        let t = self.toks.get(self.pos).map(|s| s.token.clone());
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn err(&self, msg: impl Into<String>) -> ParseError {
        ParseError {
            message: msg.into(),
            span: self.span_here(),
        }
    }
    fn err_at(&self, span: std::ops::Range<usize>, msg: impl Into<String>) -> ParseError {
        ParseError {
            message: msg.into(),
            span,
        }
    }

    fn expect(&mut self, want: &Token) -> Result<(), ParseError> {
        match self.peek() {
            Some(t) if t == want => {
                self.pos += 1;
                Ok(())
            }
            _ => Err(self.err(format!("expected `{want:?}`"))),
        }
    }

    /// Consume and return an identifier, or error.
    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.peek() {
            Some(Token::Ident(s)) => {
                let s = s.clone();
                self.pos += 1;
                Ok(s)
            }
            _ => Err(self.err("expected an identifier")),
        }
    }

    /// If the next token is the identifier `kw`, consume it and return true.
    fn eat_keyword(&mut self, kw: &str) -> bool {
        if matches!(self.peek(), Some(Token::Ident(s)) if s == kw) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_model(&mut self) -> Result<Model, ParseError> {
        if !self.eat_keyword("model") {
            return Err(self.err("expected `model`"));
        }
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;

        let mut model = Model {
            name,
            config: vec![],
            weights: vec![],
            layers: vec![],
            pipeline: vec![],
            lora: None,
        };

        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(kw)) => {
                    let kw = kw.clone();
                    match kw.as_str() {
                        "config" => {
                            self.pos += 1;
                            model.config = self.parse_kv_block()?;
                        }
                        "weights" => {
                            self.pos += 1;
                            model.weights = self.parse_kv_block()?;
                        }
                        "layer" => {
                            self.pos += 1;
                            model.layers.push(self.parse_layer()?);
                        }
                        "pipeline" => {
                            self.pos += 1;
                            model.pipeline = self.parse_pipeline()?;
                        }
                        "lora" => {
                            self.pos += 1;
                            model.lora = Some(self.parse_lora_block()?);
                        }
                        other => {
                            return Err(self.err(format!("unknown model section `{other}`")));
                        }
                    }
                }
                None => return Err(self.err("unexpected end of input inside `model`")),
                _ => return Err(self.err("expected a model section or `}`")),
            }
        }
        Ok(model)
    }

    /// `surgery NAME from "PATH" { op* }`
    fn parse_surgery(&mut self) -> Result<Surgery, ParseError> {
        if !self.eat_keyword("surgery") {
            return Err(self.err("expected `surgery`"));
        }
        let name = self.expect_ident()?;
        if !self.eat_keyword("from") {
            return Err(self.err("expected `from` after surgery name"));
        }
        let from = match self.bump() {
            Some(Token::Str(s)) => s,
            _ => return Err(self.err("expected a quoted source path after `from`")),
        };
        self.expect(&Token::LBrace)?;

        let mut ops = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(kw)) if kw == "prune" => {
                    self.pos += 1;
                    if !self.eat_keyword("layers") {
                        return Err(self.err("expected `layers` after `prune`"));
                    }
                    ops.push(SurgeryOp::PruneLayers(self.parse_layer_ranges()?));
                }
                Some(Token::Ident(kw)) if kw == "cast" => {
                    self.pos += 1;
                    if !self.eat_keyword("weights") {
                        return Err(self.err("expected `weights` after `cast`"));
                    }
                    if !self.eat_keyword("to") {
                        return Err(self.err("expected `to` after `cast weights`"));
                    }
                    let dtype = self.expect_ident()?;
                    ops.push(SurgeryOp::CastWeights(dtype));
                }
                Some(Token::Ident(kw)) if kw == "update" => {
                    self.pos += 1;
                    if !self.eat_keyword("config") {
                        return Err(self.err("expected `config` after `update`"));
                    }
                    ops.push(SurgeryOp::UpdateConfig(self.parse_kv_block()?));
                }
                Some(Token::Ident(kw)) if kw == "merge" => {
                    self.pos += 1;
                    if !self.eat_keyword("layers") {
                        return Err(self.err("expected `layers` after `merge`"));
                    }
                    let ranges = self.parse_layer_ranges()?;
                    if !self.eat_keyword("from") {
                        return Err(
                            self.err("expected `from \"donor.jac\"` after `merge layers <ranges>`")
                        );
                    }
                    let donor = match self.bump() {
                        Some(Token::Str(s)) => s,
                        _ => return Err(self.err("expected a quoted donor path after `from`")),
                    };
                    ops.push(SurgeryOp::MergeLayers { ranges, donor });
                }
                Some(Token::Ident(kw)) if kw == "apply" => {
                    self.pos += 1;
                    if !self.eat_keyword("lora") {
                        return Err(self.err("expected `lora` after `apply`"));
                    }
                    ops.push(SurgeryOp::ApplyLora(self.parse_lora_block()?));
                }
                Some(Token::Ident(other)) => {
                    return Err(self.err(format!(
                        "surgery op `{other}` is not yet supported (Phase 2: prune / cast / update / merge / apply lora)"
                    )));
                }
                None => return Err(self.err("unexpected end of input inside `surgery`")),
                _ => return Err(self.err("expected a surgery op or `}`")),
            }
        }
        Ok(Surgery { name, from, ops })
    }

    /// `A..B, C..D, E` — comma-separated inclusive ranges; a bare `E` means `E..E`.
    fn parse_layer_ranges(&mut self) -> Result<Vec<LayerRange>, ParseError> {
        let mut ranges = vec![self.parse_layer_range()?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.pos += 1;
            ranges.push(self.parse_layer_range()?);
        }
        Ok(ranges)
    }

    fn parse_layer_range(&mut self) -> Result<LayerRange, ParseError> {
        let start = self.expect_int()?;
        let end = if matches!(self.peek(), Some(Token::DotDot)) {
            self.pos += 1;
            self.expect_int()?
        } else {
            start
        };
        if end < start {
            return Err(self.err(format!("layer range {start}..{end} is empty (end < start)")));
        }
        Ok(LayerRange { start, end })
    }

    /// Consume and return a non-negative integer literal.
    fn expect_int(&mut self) -> Result<i64, ParseError> {
        let span = self.span_here();
        match self.bump() {
            Some(Token::Int(s)) => s
                .parse::<i64>()
                .map_err(|_| self.err_at(span, "invalid integer")),
            _ => Err(self.err_at(span, "expected an integer")),
        }
    }

    /// Consume an integer or float literal as an f64.
    fn expect_number(&mut self) -> Result<f64, ParseError> {
        let span = self.span_here();
        match self.bump() {
            Some(Token::Int(s)) => s
                .parse::<f64>()
                .map_err(|_| self.err_at(span, "invalid number")),
            Some(Token::Float(s)) => s
                .parse::<f64>()
                .map_err(|_| self.err_at(span, "invalid number")),
            _ => Err(self.err_at(span, "expected a number")),
        }
    }

    /// `{ rank = N  alpha = M  targets = w_q, w_k, ...  source = "repo" }` (fields in any order).
    fn parse_lora_block(&mut self) -> Result<LoraSpec, ParseError> {
        self.expect(&Token::LBrace)?;
        let (mut rank, mut alpha, mut targets, mut source) = (None, None, None, None);
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(_)) => {
                    let key = self.expect_ident()?;
                    self.expect(&Token::Eq)?;
                    match key.as_str() {
                        "rank" => rank = Some(self.expect_int()?),
                        "alpha" => alpha = Some(self.expect_number()?),
                        "source" => {
                            source = Some(match self.bump() {
                                Some(Token::Str(s)) => s,
                                _ => return Err(self.err("lora `source` must be a quoted string")),
                            })
                        }
                        "targets" => {
                            let mut ts = vec![self.expect_ident()?];
                            while matches!(self.peek(), Some(Token::Comma)) {
                                self.pos += 1;
                                ts.push(self.expect_ident()?);
                            }
                            targets = Some(ts);
                        }
                        other => return Err(self.err(format!("unknown lora field `{other}`"))),
                    }
                }
                _ => return Err(self.err("expected a lora field or `}`")),
            }
        }
        Ok(LoraSpec {
            rank: rank.ok_or_else(|| self.err("lora block missing `rank`"))?,
            alpha: alpha.ok_or_else(|| self.err("lora block missing `alpha`"))?,
            targets: targets.ok_or_else(|| self.err("lora block missing `targets`"))?,
            source: source.ok_or_else(|| self.err("lora block missing `source`"))?,
        })
    }

    // ---- kernel DSL (the second sub-language) ----

    /// `kernel NAME { tensors {...} params {...} constraints {...} phase ... }`
    fn parse_kernel(&mut self) -> Result<Kernel, ParseError> {
        if !self.eat_keyword("kernel") {
            return Err(self.err("expected `kernel`"));
        }
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let mut k = Kernel {
            name,
            tensors: vec![],
            state: vec![],
            params: vec![],
            constraints: vec![],
            phases: vec![],
            fusion: vec![],
        };
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(kw)) => match kw.clone().as_str() {
                    "tensors" => {
                        self.pos += 1;
                        k.tensors = self.parse_tensor_block()?;
                    }
                    "params" => {
                        self.pos += 1;
                        k.params = self.parse_kernel_params()?;
                    }
                    "constraints" => {
                        self.pos += 1;
                        k.constraints = self.parse_constraints()?;
                    }
                    "phase" => {
                        self.pos += 1;
                        k.phases.push(self.parse_phase()?);
                    }
                    "state" => {
                        self.pos += 1;
                        k.state = self.parse_state_block()?;
                    }
                    "fusion" => {
                        self.pos += 1;
                        k.fusion = self.parse_fusion_block()?;
                    }
                    other => return Err(self.err(format!("unknown kernel section `{other}`"))),
                },
                None => return Err(self.err("unexpected end of input inside `kernel`")),
                _ => return Err(self.err("expected a kernel section or `}`")),
            }
        }
        if k.tensors.is_empty() {
            return Err(self.err("kernel requires a `tensors` block"));
        }
        Ok(k)
    }

    /// `tensors { Name[d,...] : dtype @placement [-> access] [-> global] ... }`
    fn parse_tensor_block(&mut self) -> Result<Vec<TensorDecl>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(_)) => out.push(self.parse_tensor_decl()?),
                _ => return Err(self.err("expected a tensor declaration or `}`")),
            }
        }
        Ok(out)
    }

    fn parse_tensor_decl(&mut self) -> Result<TensorDecl, ParseError> {
        let name = self.expect_ident()?;
        self.expect(&Token::LBracket)?;
        let mut dims = vec![self.expect_ident()?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.pos += 1;
            dims.push(self.expect_ident()?);
        }
        self.expect(&Token::RBracket)?;
        self.expect(&Token::Colon)?;
        let dtype = self.expect_ident()?;
        const DTYPES: [&str; 8] = [
            "fp16", "fp32", "bf16", "int8", "int16", "int32", "int64", "bool",
        ];
        if !DTYPES.contains(&dtype.as_str()) {
            return Err(self.err(format!("unknown tensor dtype `{dtype}`")));
        }
        self.expect(&Token::At)?;
        let placement = self.expect_ident()?;
        if !matches!(placement.as_str(), "registers" | "smem" | "global") {
            return Err(self.err(format!(
                "unknown placement `@{placement}` (registers/smem/global)"
            )));
        }
        // Zero, one, or two `-> flow` suffixes: `stream`/`gather` (access) and/or `global` (output).
        let (mut access, mut output) = (None, false);
        while matches!(self.peek(), Some(Token::Arrow)) {
            self.pos += 1;
            let flow = self.expect_ident()?;
            match flow.as_str() {
                "global" => output = true,
                "stream" | "gather" => access = Some(flow),
                other => return Err(self.err(format!("unknown access/output flow `-> {other}`"))),
            }
        }
        Ok(TensorDecl {
            name,
            dims,
            dtype,
            placement,
            access,
            output,
        })
    }

    /// `params { IDENT = INT ... }` — compile-time integer parameters.
    fn parse_kernel_params(&mut self) -> Result<Vec<(String, i64)>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(_)) => {
                    let key = self.expect_ident()?;
                    self.expect(&Token::Eq)?;
                    out.push((key, self.expect_int()?));
                }
                _ => return Err(self.err("expected a `name = int` param or `}`")),
            }
        }
        Ok(out)
    }

    /// `constraints { value op value ... }`.
    fn parse_constraints(&mut self) -> Result<Vec<Constraint>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut out = Vec::new();
        loop {
            if matches!(self.peek(), Some(Token::RBrace)) {
                self.pos += 1;
                break;
            }
            let lhs = self.parse_cval()?;
            let op = match self.bump() {
                Some(Token::Ge) => CmpOp::Ge,
                Some(Token::Le) => CmpOp::Le,
                Some(Token::Eq) => CmpOp::Eq,
                Some(Token::Gt) => CmpOp::Gt,
                Some(Token::Lt) => CmpOp::Lt,
                _ => return Err(self.err("expected a comparison operator (>=, <=, =, >, <)")),
            };
            let rhs = self.parse_cval()?;
            out.push(Constraint { lhs, op, rhs });
        }
        Ok(out)
    }

    /// A constraint value: an integer with an optional glued unit (`48KB`), or a symbol (`sm_80`).
    fn parse_cval(&mut self) -> Result<CVal, ParseError> {
        match self.peek() {
            Some(Token::Int(_)) => {
                let int_span = self.span_here();
                let n = self.expect_int()?;
                // A unit is an identifier glued directly to the number (no whitespace), so the next
                // constraint's left-hand identifier (on a new line) is not mistaken for a unit.
                let unit = match self.peek_spanned() {
                    Some(s)
                        if matches!(s.token, Token::Ident(_)) && s.span.start == int_span.end =>
                    {
                        Some(self.expect_ident()?)
                    }
                    _ => None,
                };
                Ok(CVal::Num(n, unit))
            }
            Some(Token::Ident(_)) => Ok(CVal::Sym(self.expect_ident()?)),
            _ => Err(self.err("expected a constraint value (number or symbol)")),
        }
    }

    /// Consume an identifier and check it names one of the eight scalar dtypes.
    fn expect_dtype(&mut self) -> Result<String, ParseError> {
        const DTYPES: [&str; 8] = [
            "fp16", "fp32", "bf16", "int8", "int16", "int32", "int64", "bool",
        ];
        let d = self.expect_ident()?;
        if !DTYPES.contains(&d.as_str()) {
            return Err(self.err(format!("unknown dtype `{d}`")));
        }
        Ok(d)
    }

    /// `state { NAME ["[" dims "]"] : type @placement qualifier* ... }`
    fn parse_state_block(&mut self) -> Result<Vec<StateDecl>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(_)) => out.push(self.parse_state_decl()?),
                _ => return Err(self.err("expected a state declaration or `}`")),
            }
        }
        Ok(out)
    }

    fn parse_state_decl(&mut self) -> Result<StateDecl, ParseError> {
        let name = self.expect_ident()?;
        let mut dims = Vec::new();
        if matches!(self.peek(), Some(Token::LBracket)) {
            self.pos += 1;
            dims.push(self.expect_ident()?);
            while matches!(self.peek(), Some(Token::Comma)) {
                self.pos += 1;
                dims.push(self.expect_ident()?);
            }
            self.expect(&Token::RBracket)?;
        }
        self.expect(&Token::Colon)?;
        // type_expr: a scalar dtype, or a tuple `(dtype, dtype, ...)`.
        let mut dtype = Vec::new();
        if matches!(self.peek(), Some(Token::LParen)) {
            self.pos += 1;
            dtype.push(self.expect_dtype()?);
            while matches!(self.peek(), Some(Token::Comma)) {
                self.pos += 1;
                dtype.push(self.expect_dtype()?);
            }
            self.expect(&Token::RParen)?;
            if dtype.len() < 2 {
                return Err(self.err("a tuple type needs at least two scalar types"));
            }
        } else {
            dtype.push(self.expect_dtype()?);
        }
        self.expect(&Token::At)?;
        let placement = self.expect_ident()?;
        if !matches!(placement.as_str(), "registers" | "smem" | "global") {
            return Err(self.err(format!(
                "unknown placement `@{placement}` (registers/smem/global)"
            )));
        }
        let mut qualifiers = Vec::new();
        loop {
            match self.peek() {
                Some(Token::Ident(q)) if q == "broadcast" || q == "atomic" => {
                    qualifiers.push(self.expect_ident()?);
                }
                _ => break,
            }
        }
        Ok(StateDecl {
            name,
            dims,
            dtype,
            placement,
            qualifiers,
        })
    }

    /// `fusion { (A + B = single_kernel | a.op overlaps_with b.op) ... }`
    fn parse_fusion_block(&mut self) -> Result<Vec<FusionDirective>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(_)) => out.push(self.parse_fusion_directive()?),
                _ => return Err(self.err("expected a fusion directive or `}`")),
            }
        }
        Ok(out)
    }

    fn parse_fusion_directive(&mut self) -> Result<FusionDirective, ParseError> {
        let first = self.expect_ident()?;
        if matches!(self.peek(), Some(Token::Dot)) {
            // Overlap: `a.op overlaps_with b.op`
            self.pos += 1;
            let a_op = self.expect_ident()?;
            if !self.eat_keyword("overlaps_with") {
                return Err(self.err("expected `overlaps_with` in fusion overlap directive"));
            }
            let b_phase = self.expect_ident()?;
            self.expect(&Token::Dot)?;
            let b_op = self.expect_ident()?;
            return Ok(FusionDirective::Overlap {
                a: (first, a_op),
                b: (b_phase, b_op),
            });
        }
        // Combine: `A + B + ... = single_kernel`
        let mut phases = vec![first];
        while matches!(self.peek(), Some(Token::Plus)) {
            self.pos += 1;
            phases.push(self.expect_ident()?);
        }
        self.expect(&Token::Eq)?;
        let mode = self.expect_ident()?;
        if mode != "single_kernel" {
            return Err(self.err(format!(
                "unknown fusion mode `{mode}` (expected `single_kernel`)"
            )));
        }
        Ok(FusionDirective::Combine { phases, mode })
    }

    /// `phase NAME [over T.dim(D) by TILE] [after OTHER] { ...body... }` — header parsed, body skipped.
    fn parse_phase(&mut self) -> Result<Phase, ParseError> {
        let name = self.expect_ident()?;
        let mut iteration = None;
        let mut after = None;
        if self.eat_keyword("over") {
            let tensor = self.expect_ident()?;
            self.expect(&Token::Dot)?;
            if !self.eat_keyword("dim") {
                return Err(self.err("expected `.dim(DIM)` in phase iteration"));
            }
            self.expect(&Token::LParen)?;
            let dim = self.expect_ident()?;
            self.expect(&Token::RParen)?;
            if !self.eat_keyword("by") {
                return Err(self.err("expected `by TILE` in phase iteration"));
            }
            let by = self.expect_ident()?;
            iteration = Some(IterSpec { tensor, dim, by });
        }
        if self.eat_keyword("after") {
            after = Some(self.expect_ident()?);
        }
        let body = self.parse_phase_body()?;
        Ok(Phase {
            name,
            iteration,
            after,
            body,
        })
    }

    /// `{ stmt* }` — the phase body. Parses `load`, `einsum`, and `compare`; other statement
    /// forms error clearly (deferred to the kernels that need them — no silent drop).
    fn parse_phase_body(&mut self) -> Result<Vec<KStmt>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut body = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(kw)) if kw == "load" => {
                    self.pos += 1;
                    body.push(self.parse_load_stmt()?);
                }
                Some(Token::Ident(kw)) if kw == "online_softmax" => {
                    self.pos += 1;
                    body.push(self.parse_online_softmax_stmt()?);
                }
                Some(Token::Ident(kw)) if matches!(kw.as_str(), "on" | "append" | "drain") => {
                    let kw = kw.clone();
                    return Err(self.err(format!(
                        "phase statement `{kw}` (KV-cache eviction) is not yet supported"
                    )));
                }
                // Otherwise `DST = (einsum(...) | softmax(...) | SRC.gather(...) | LHS op RHS)`.
                Some(Token::Ident(_)) => body.push(self.parse_assign_stmt()?),
                _ => return Err(self.err("expected a phase statement or `}`")),
            }
        }
        Ok(body)
    }

    /// `load DST = SRC[idx, ...] [via NAME(.NAME)*]`
    fn parse_load_stmt(&mut self) -> Result<KStmt, ParseError> {
        let dst = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        let src = self.expect_ident()?;
        self.expect(&Token::LBracket)?;
        let mut index = vec![self.parse_index_expr()?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.pos += 1;
            index.push(self.parse_index_expr()?);
        }
        self.expect(&Token::RBracket)?;
        let via = if self.eat_keyword("via") {
            let mut path = self.expect_ident()?;
            while matches!(self.peek(), Some(Token::Dot)) {
                self.pos += 1;
                path.push('.');
                path.push_str(&self.expect_ident()?);
            }
            Some(path)
        } else {
            None
        };
        Ok(KStmt::Load {
            dst,
            src,
            index,
            via,
        })
    }

    fn parse_index_expr(&mut self) -> Result<IndexExpr, ParseError> {
        match self.peek() {
            Some(Token::Star) => {
                self.pos += 1;
                Ok(IndexExpr::Wildcard)
            }
            Some(Token::Ident(_)) => Ok(IndexExpr::Name(self.expect_ident()?)),
            _ => Err(self.err("expected `*` or an index name")),
        }
    }

    /// `DST = einsum("...", op, ...)` or `DST = LHS <cmp> RHS`.
    fn parse_assign_stmt(&mut self) -> Result<KStmt, ParseError> {
        let dst = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        if self.eat_keyword("einsum") {
            self.expect(&Token::LParen)?;
            let subscripts = match self.bump() {
                Some(Token::Str(s)) => s,
                _ => return Err(self.err("einsum: expected a subscript string")),
            };
            let mut operands = Vec::new();
            while matches!(self.peek(), Some(Token::Comma)) {
                self.pos += 1;
                operands.push(self.expect_ident()?);
            }
            self.expect(&Token::RParen)?;
            if operands.is_empty() {
                return Err(self.err("einsum: expected at least one operand"));
            }
            return Ok(KStmt::Einsum {
                dst,
                subscripts,
                operands,
            });
        }
        if self.eat_keyword("softmax") {
            // `softmax(SRC, dim: DIM)`
            self.expect(&Token::LParen)?;
            let src = self.expect_ident()?;
            self.expect(&Token::Comma)?;
            if !self.eat_keyword("dim") {
                return Err(self.err("softmax: expected `dim: DIM`"));
            }
            self.expect(&Token::Colon)?;
            let dim = self.expect_ident()?;
            self.expect(&Token::RParen)?;
            return Ok(KStmt::Softmax { dst, src, dim });
        }
        let first = self.expect_ident()?;
        if matches!(self.peek(), Some(Token::Dot)) {
            // `SRC.gather(INDEX)`
            self.pos += 1;
            if !self.eat_keyword("gather") {
                return Err(self.err("expected `.gather(INDEX)`"));
            }
            self.expect(&Token::LParen)?;
            let index = self.expect_ident()?;
            self.expect(&Token::RParen)?;
            return Ok(KStmt::Gather {
                dst,
                src: first,
                index,
            });
        }
        // Compare: `DST = LHS <op> RHS`.
        let op = match self.bump() {
            Some(Token::Ge) => CmpOp::Ge,
            Some(Token::Le) => CmpOp::Le,
            Some(Token::Eq) => CmpOp::Eq,
            Some(Token::Gt) => CmpOp::Gt,
            Some(Token::Lt) => CmpOp::Lt,
            _ => {
                return Err(
                    self.err("expected einsum/softmax/.gather or a comparison `LHS <op> RHS`")
                )
            }
        };
        let rhs = self.expect_ident()?;
        Ok(KStmt::Compare {
            dst,
            lhs: first,
            op,
            rhs,
        })
    }

    /// `online_softmax SCORES -> ACC { V = .. running_max = .. running_sum = .. }`
    fn parse_online_softmax_stmt(&mut self) -> Result<KStmt, ParseError> {
        let scores = self.expect_ident()?;
        self.expect(&Token::Arrow)?;
        let acc = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let (mut v, mut running_max, mut running_sum) = (None, None, None);
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(_)) => {
                    let key = self.expect_ident()?;
                    self.expect(&Token::Eq)?;
                    let val = self.expect_ident()?;
                    match key.as_str() {
                        "V" => v = Some(val),
                        "running_max" => running_max = Some(val),
                        "running_sum" => running_sum = Some(val),
                        other => {
                            return Err(self.err(format!("unknown online_softmax field `{other}`")))
                        }
                    }
                }
                _ => return Err(self.err("expected an online_softmax field or `}`")),
            }
        }
        Ok(KStmt::OnlineSoftmax {
            scores,
            acc,
            v: v.ok_or_else(|| self.err("online_softmax: missing `V`"))?,
            running_max: running_max
                .ok_or_else(|| self.err("online_softmax: missing `running_max`"))?,
            running_sum: running_sum
                .ok_or_else(|| self.err("online_softmax: missing `running_sum`"))?,
        })
    }

    /// `{ key = value  key = value ... }`
    fn parse_kv_block(&mut self) -> Result<Vec<(String, Value)>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut entries = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(_)) => {
                    let key = self.expect_ident()?;
                    self.expect(&Token::Eq)?;
                    let value = self.parse_value()?;
                    entries.push((key, value));
                }
                _ => return Err(self.err("expected a `key = value` entry or `}`")),
            }
        }
        Ok(entries)
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        let span = self.span_here();
        match self.bump() {
            Some(Token::Int(s)) => s
                .parse::<i64>()
                .map(Value::Int)
                .map_err(|_| self.err_at(span, "invalid integer")),
            Some(Token::Float(s)) => s
                .parse::<f64>()
                .map(Value::Float)
                .map_err(|_| self.err_at(span, "invalid float")),
            Some(Token::Str(s)) => Ok(Value::Str(s)),
            Some(Token::Ident(s)) => Ok(Value::Ident(s)),
            _ => Err(self.err_at(span, "expected a value (number, string, or identifier)")),
        }
    }

    /// `NAME { stmt* }` where each stmt is `target0, target1, ... = expr`.
    fn parse_layer(&mut self) -> Result<Layer, ParseError> {
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(_)) => stmts.push(self.parse_stmt()?),
                _ => return Err(self.err("expected a statement or `}` in layer")),
            }
        }
        Ok(Layer { name, stmts })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        let mut targets = vec![self.expect_ident()?];
        while matches!(self.peek(), Some(Token::Comma)) {
            self.pos += 1;
            targets.push(self.expect_ident()?);
        }
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Stmt { targets, value })
    }

    /// `step -> step -> ...`, each step a call or bareword.
    fn parse_pipeline(&mut self) -> Result<Vec<Expr>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut steps = Vec::new();
        if matches!(self.peek(), Some(Token::RBrace)) {
            self.pos += 1;
            return Ok(steps);
        }
        steps.push(self.parse_primary()?);
        loop {
            match self.peek() {
                Some(Token::Arrow) => {
                    self.pos += 1;
                    steps.push(self.parse_primary()?);
                }
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.err("expected `->` or `}` in pipeline")),
            }
        }
        Ok(steps)
    }

    // Expression grammar: add (lowest) -> mul -> primary.
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_add()
    }
    fn parse_add(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_mul()?;
        while matches!(self.peek(), Some(Token::Plus)) {
            self.pos += 1;
            let rhs = self.parse_mul()?;
            lhs = Expr::Binary {
                op: BinOp::Add,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }
    fn parse_mul(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_primary()?;
        while matches!(self.peek(), Some(Token::Star)) {
            self.pos += 1;
            let rhs = self.parse_primary()?;
            lhs = Expr::Binary {
                op: BinOp::Mul,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }
    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let span = self.span_here();
        match self.peek().cloned() {
            Some(Token::Int(s)) => {
                self.pos += 1;
                s.parse::<i64>()
                    .map(Expr::Int)
                    .map_err(|_| self.err_at(span, "invalid integer"))
            }
            Some(Token::Float(s)) => {
                self.pos += 1;
                s.parse::<f64>()
                    .map(Expr::Float)
                    .map_err(|_| self.err_at(span, "invalid float"))
            }
            Some(Token::Str(s)) => {
                self.pos += 1;
                Ok(Expr::Str(s))
            }
            Some(Token::Ident(name)) => {
                self.pos += 1;
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.pos += 1;
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Token::RParen)) {
                        args.push(self.parse_expr()?);
                        while matches!(self.peek(), Some(Token::Comma)) {
                            self.pos += 1;
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(&Token::RParen)?;
                    Ok(Expr::Call { name, args })
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            _ => Err(self.err_at(span, "expected an expression")),
        }
    }

    // ---- experiment DSL ----

    fn parse_experiment(&mut self) -> Result<Experiment, ParseError> {
        if !self.eat_keyword("experiment") {
            return Err(self.err("expected `experiment`"));
        }
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let (mut kind, mut fixture, mut subjects, mut reference, mut sweep, mut metrics) =
            (None, vec![], vec![], None, vec![], vec![]);
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(kw)) => match kw.clone().as_str() {
                    "kind" => {
                        self.pos += 1;
                        self.expect(&Token::Eq)?;
                        let k = self.expect_ident()?;
                        kind = Some(match k.as_str() {
                            "kernel" => ExpKind::Kernel,
                            other => {
                                return Err(self.err(format!(
                                    "unknown experiment kind `{other}` (expected `kernel`)"
                                )))
                            }
                        });
                    }
                    "fixture" => {
                        self.pos += 1;
                        fixture = self.parse_kv_block()?;
                    }
                    "subjects" => {
                        self.pos += 1;
                        subjects = self.parse_subjects_block()?;
                    }
                    "reference" => {
                        self.pos += 1;
                        self.expect(&Token::Eq)?;
                        reference = Some(self.expect_ident()?);
                    }
                    "sweep" => {
                        self.pos += 1;
                        sweep = self.parse_sweep_block()?;
                    }
                    "metrics" => {
                        self.pos += 1;
                        self.expect(&Token::Eq)?;
                        metrics = self.parse_metric_list()?;
                    }
                    other => return Err(self.err(format!("unknown experiment section `{other}`"))),
                },
                None => return Err(self.err("unexpected end of input inside `experiment`")),
                _ => return Err(self.err("expected an experiment section or `}`")),
            }
        }
        Ok(Experiment {
            name,
            kind: kind.ok_or_else(|| self.err("experiment missing `kind`"))?,
            fixture,
            subjects,
            reference: reference.ok_or_else(|| self.err("experiment missing `reference`"))?,
            sweep,
            metrics,
        })
    }

    /// "{ name = jac(variant) | cuda(\"path\") ... }"
    fn parse_subjects_block(&mut self) -> Result<Vec<Subject>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(_)) => {
                    let name = self.expect_ident()?;
                    self.expect(&Token::Eq)?;
                    let ctor = self.expect_ident()?;
                    self.expect(&Token::LParen)?;
                    let src = match ctor.as_str() {
                        "jac" => SubjectSrc::Jac(self.expect_ident()?),
                        "cuda" => match self.bump() {
                            Some(Token::Str(s)) => SubjectSrc::Cuda(s),
                            _ => return Err(self.err("cuda(...) expects a quoted path")),
                        },
                        other => {
                            return Err(self.err(format!(
                                "unknown subject constructor `{other}` (expected jac/cuda)"
                            )))
                        }
                    };
                    self.expect(&Token::RParen)?;
                    out.push(Subject { name, src });
                }
                _ => return Err(self.err("expected a subject (`name = jac(..)|cuda(..)`) or `}`")),
            }
        }
        Ok(out)
    }

    /// "{ NAME = [int, int, ...] ... }"
    fn parse_sweep_block(&mut self) -> Result<Vec<SweepAxis>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.pos += 1;
                    break;
                }
                Some(Token::Ident(_)) => {
                    let name = self.expect_ident()?;
                    self.expect(&Token::Eq)?;
                    self.expect(&Token::LBracket)?;
                    let mut values = Vec::new();
                    while !matches!(self.peek(), Some(Token::RBracket)) {
                        values.push(self.expect_int()?);
                        if matches!(self.peek(), Some(Token::Comma)) {
                            self.pos += 1;
                        }
                    }
                    self.expect(&Token::RBracket)?;
                    if values.is_empty() {
                        return Err(self.err("sweep axis must have at least one value"));
                    }
                    out.push(SweepAxis { name, values });
                }
                _ => return Err(self.err("expected a sweep axis (`NAME = [ints]`) or `}`")),
            }
        }
        Ok(out)
    }

    /// "[ name | name(key: value, ...) , ... ]"
    fn parse_metric_list(&mut self) -> Result<Vec<Metric>, ParseError> {
        self.expect(&Token::LBracket)?;
        let mut out = Vec::new();
        if matches!(self.peek(), Some(Token::RBracket)) {
            self.pos += 1;
            return Ok(out);
        }
        loop {
            let name = self.expect_ident()?;
            let mut args = Vec::new();
            if matches!(self.peek(), Some(Token::LParen)) {
                self.pos += 1;
                loop {
                    match self.peek() {
                        Some(Token::RParen) => break,
                        None => {
                            return Err(self.err("unexpected end of input in metric argument list"))
                        }
                        _ => {}
                    }
                    let k = self.expect_ident()?;
                    self.expect(&Token::Colon)?;
                    args.push((k, self.parse_value()?));
                    if matches!(self.peek(), Some(Token::Comma)) {
                        self.pos += 1;
                    }
                }
                self.expect(&Token::RParen)?;
            }
            out.push(Metric { name, args });
            match self.peek() {
                Some(Token::Comma) => {
                    self.pos += 1;
                }
                Some(Token::RBracket) => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.err("expected `,` or `]` in metrics list")),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_minimal_model() {
        let src = r#"
        model tiny {
          config { hidden = 8  layers = 2  norm_eps = 1e-5 }
          weights { format = huggingface  source = "org/model"  dtype = fp16 }
          layer decoder {
            normed = rmsnorm(input, norm_weight, norm_eps)
            q, k, v = linear(normed, w_q, w_k, w_v)
            ffn_mid = silu(gate) * up
            output = residual(hidden, linear(ffn_mid, w_down))
          }
          pipeline {
            embed(token, embed_weights)
            -> repeat(decoder, layers)
            -> lm_head(lm_head_weights)
            -> greedy_sample
          }
        }
        "#;
        let m = parse_model(src).expect("should parse");
        assert_eq!(m.name, "tiny");
        assert_eq!(m.config_value("hidden"), Some(&Value::Int(8)));
        assert_eq!(
            m.weights_value("format"),
            Some(&Value::Ident("huggingface".into()))
        );
        assert_eq!(m.layers.len(), 1);
        assert_eq!(m.layers[0].stmts.len(), 4);
        // multi-target fused op
        assert_eq!(m.layers[0].stmts[1].targets, vec!["q", "k", "v"]);
        // pipeline: embed -> repeat -> lm_head -> greedy_sample
        assert_eq!(m.pipeline.len(), 4);
        assert!(matches!(&m.pipeline[3], Expr::Ident(s) if s == "greedy_sample"));
    }

    #[test]
    fn reports_unknown_section() {
        let err = parse_model("model m { bogus { } }").unwrap_err();
        assert!(err.message.contains("unknown model section"));
    }

    const GEMV: &str = r#"
    kernel gemv {
      tensors {
        x[N]      : fp16  @registers
        W[M, N]   : fp16  @global -> stream
        y[M]      : fp32  @registers -> global
      }
      params {
        M      = 4096
        N      = 128
        tile_m = 128
      }
      constraints {
        arch >= sm_80
        registers_per_thread <= 255
        smem_per_block <= 48KB
        occupancy >= 2
      }
      phase compute over W.dim(M) by tile_m {
        load W_tile = W[tile, *]  via cp.async
        y_tile = einsum("n, tn -> t", x, W_tile)
      }
    }"#;

    #[test]
    fn parses_gemv_kernel() {
        let k = parse_kernel(GEMV).expect("gemv should parse");
        assert_eq!(k.name, "gemv");
        assert_eq!(k.tensors.len(), 3);
        // x[N] : fp16 @registers
        assert_eq!(k.tensors[0].name, "x");
        assert_eq!(k.tensors[0].dims, vec!["N"]);
        assert_eq!(k.tensors[0].placement, "registers");
        // W[M,N] : fp16 @global -> stream
        assert_eq!(k.tensors[1].dims, vec!["M", "N"]);
        assert_eq!(k.tensors[1].access.as_deref(), Some("stream"));
        // y[M] : fp32 @registers -> global  (output)
        assert!(k.tensors[2].output);
        assert_eq!(k.tensors[2].dtype, "fp32");
        assert_eq!(
            k.params,
            vec![
                ("M".into(), 4096),
                ("N".into(), 128),
                ("tile_m".into(), 128)
            ]
        );
        assert_eq!(k.constraints.len(), 4);
        // smem_per_block <= 48KB  — the unit must glue to the number, not swallow the next line.
        match (&k.constraints[2].lhs, &k.constraints[2].rhs) {
            (CVal::Sym(l), CVal::Num(48, Some(u))) => {
                assert_eq!(l, "smem_per_block");
                assert_eq!(u, "KB");
            }
            other => panic!("unexpected smem constraint: {other:?}"),
        }
        // phase header + body parsed.
        assert_eq!(k.phases.len(), 1);
        let ph = &k.phases[0];
        let it = ph.iteration.as_ref().unwrap();
        assert_eq!(
            (it.tensor.as_str(), it.dim.as_str(), it.by.as_str()),
            ("W", "M", "tile_m")
        );
        assert_eq!(ph.body.len(), 2);
        match &ph.body[0] {
            KStmt::Load {
                dst,
                src,
                index,
                via,
            } => {
                assert_eq!((dst.as_str(), src.as_str()), ("W_tile", "W"));
                assert_eq!(
                    index,
                    &[IndexExpr::Name("tile".into()), IndexExpr::Wildcard]
                );
                assert_eq!(via.as_deref(), Some("cp.async"));
            }
            other => panic!("expected Load, got {other:?}"),
        }
        match &ph.body[1] {
            KStmt::Einsum {
                dst,
                subscripts,
                operands,
            } => {
                assert_eq!(dst, "y_tile");
                assert_eq!(subscripts, "n, tn -> t");
                assert_eq!(operands, &["x", "W_tile"]);
            }
            other => panic!("expected Einsum, got {other:?}"),
        }
    }

    #[test]
    fn parses_compare_statement() {
        let k = parse_kernel(
            r#"kernel k { tensors { s[N] : fp32 @registers } phase p { exceeds = s > threshold } }"#,
        )
        .unwrap();
        match &k.phases[0].body[0] {
            KStmt::Compare { dst, lhs, op, rhs } => {
                assert_eq!(
                    (dst.as_str(), lhs.as_str(), rhs.as_str()),
                    ("exceeds", "s", "threshold")
                );
                assert_eq!(*op, CmpOp::Gt);
            }
            other => panic!("expected Compare, got {other:?}"),
        }
    }

    #[test]
    fn parses_online_softmax() {
        let k = parse_kernel(
            r#"kernel k { tensors { s[N] : fp32 @registers }
               phase p { online_softmax scores -> acc { V = V_tile  running_max = rmax  running_sum = rsum } } }"#,
        )
        .unwrap();
        match &k.phases[0].body[0] {
            KStmt::OnlineSoftmax {
                scores,
                acc,
                v,
                running_max,
                running_sum,
            } => {
                assert_eq!(scores, "scores");
                assert_eq!(acc, "acc");
                assert_eq!(v, "V_tile");
                assert_eq!(running_max, "rmax");
                assert_eq!(running_sum, "rsum");
            }
            other => panic!("expected OnlineSoftmax, got {other:?}"),
        }
    }

    #[test]
    fn parses_softmax_and_gather() {
        let k = parse_kernel(
            r#"kernel k { tensors { x[N] : fp16 @registers }
               phase p { p1 = softmax(scores, dim: kv)  g1 = K.gather(idx) } }"#,
        )
        .unwrap();
        match &k.phases[0].body[0] {
            KStmt::Softmax { dst, src, dim } => assert_eq!(
                (dst.as_str(), src.as_str(), dim.as_str()),
                ("p1", "scores", "kv")
            ),
            other => panic!("expected Softmax, got {other:?}"),
        }
        match &k.phases[0].body[1] {
            KStmt::Gather { dst, src, index } => assert_eq!(
                (dst.as_str(), src.as_str(), index.as_str()),
                ("g1", "K", "idx")
            ),
            other => panic!("expected Gather, got {other:?}"),
        }
    }

    #[test]
    fn phase_body_defers_eviction_statements() {
        let err = parse_kernel(
            r#"kernel k { tensors { s[N] : fp32 @registers } phase p { drain staging -> kv { } } }"#,
        )
        .unwrap_err();
        assert!(
            err.message.contains("not yet supported"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn kernel_requires_tensors() {
        let err = parse_kernel("kernel k { params { M = 8 } }").unwrap_err();
        assert!(err.message.contains("tensors"), "got: {}", err.message);
    }

    #[test]
    fn kernel_rejects_unknown_dtype_and_placement() {
        assert!(
            parse_kernel("kernel k { tensors { x[N] : fp9 @registers } }")
                .unwrap_err()
                .message
                .contains("dtype")
        );
        assert!(parse_kernel("kernel k { tensors { x[N] : fp16 @vram } }")
            .unwrap_err()
            .message
            .contains("placement"));
    }

    #[test]
    fn parses_state_block() {
        // The flagship attention state: scalars + an array + (separately) a tuple type.
        let k = parse_kernel(
            r#"kernel k {
              tensors { x[N] : fp16 @registers }
              state {
                running_max         : fp32 @registers
                acc[d_head]         : fp32 @smem
                staging[tile]       : (fp32, int32) @smem broadcast atomic
              }
            }"#,
        )
        .unwrap();
        assert_eq!(k.state.len(), 3);
        assert_eq!(k.state[0].name, "running_max");
        assert!(k.state[0].dims.is_empty()); // scalar
        assert_eq!(k.state[1].dims, vec!["d_head"]);
        assert_eq!(k.state[1].placement, "smem");
        assert_eq!(k.state[2].dtype, vec!["fp32", "int32"]); // tuple
        assert_eq!(k.state[2].qualifiers, vec!["broadcast", "atomic"]);
    }

    #[test]
    fn parses_fusion_block() {
        let k = parse_kernel(
            r#"kernel k {
              tensors { x[N] : fp16 @registers }
              fusion { scan + gather = single_kernel  gather.V_load overlaps_with scan.last_drain }
            }"#,
        )
        .unwrap();
        assert_eq!(k.fusion.len(), 2);
        match &k.fusion[0] {
            FusionDirective::Combine { phases, mode } => {
                assert_eq!(phases, &["scan", "gather"]);
                assert_eq!(mode, "single_kernel");
            }
            other => panic!("expected Combine, got {other:?}"),
        }
        match &k.fusion[1] {
            FusionDirective::Overlap { a, b } => {
                assert_eq!(a, &("gather".to_string(), "V_load".to_string()));
                assert_eq!(b, &("scan".to_string(), "last_drain".to_string()));
            }
            other => panic!("expected Overlap, got {other:?}"),
        }
    }

    #[test]
    fn parses_prune_surgery() {
        let s = parse_surgery(
            r#"surgery slim from "specs/tinyllama.jac" { prune layers 0..2, 20..21 }"#,
        )
        .expect("should parse");
        assert_eq!(s.name, "slim");
        assert_eq!(s.from, "specs/tinyllama.jac");
        assert_eq!(s.ops.len(), 1);
        match &s.ops[0] {
            SurgeryOp::PruneLayers(ranges) => {
                assert_eq!(
                    ranges,
                    &[
                        LayerRange { start: 0, end: 2 },
                        LayerRange { start: 20, end: 21 }
                    ]
                );
            }
            other => panic!("expected PruneLayers, got {other:?}"),
        }
    }

    #[test]
    fn prune_accepts_single_index() {
        let s = parse_surgery(r#"surgery s from "m.jac" { prune layers 5 }"#).unwrap();
        match &s.ops[0] {
            SurgeryOp::PruneLayers(r) => assert_eq!(r, &[LayerRange { start: 5, end: 5 }]),
            other => panic!("expected PruneLayers, got {other:?}"),
        }
    }

    #[test]
    fn parses_cast_and_update_and_compose() {
        let s = parse_surgery(
            r#"surgery s from "m.jac" {
                prune layers 0..1
                cast weights to int8
                update config { max_seq = 4096 }
            }"#,
        )
        .expect("should parse");
        assert_eq!(s.ops.len(), 3);
        assert!(matches!(&s.ops[0], SurgeryOp::PruneLayers(_)));
        assert!(matches!(&s.ops[1], SurgeryOp::CastWeights(d) if d == "int8"));
        match &s.ops[2] {
            SurgeryOp::UpdateConfig(kvs) => {
                assert_eq!(kvs, &[("max_seq".to_string(), Value::Int(4096))]);
            }
            _ => panic!("expected UpdateConfig"),
        }
    }

    #[test]
    fn parses_merge_layers() {
        let s = parse_surgery(r#"surgery s from "m.jac" { merge layers 11..16 from "donor.jac" }"#)
            .unwrap();
        match &s.ops[0] {
            SurgeryOp::MergeLayers { ranges, donor } => {
                assert_eq!(ranges, &[LayerRange { start: 11, end: 16 }]);
                assert_eq!(donor, "donor.jac");
            }
            other => panic!("expected MergeLayers, got {other:?}"),
        }
    }

    #[test]
    fn merge_requires_from() {
        let err = parse_surgery(r#"surgery s from "m.jac" { merge layers 0..1 }"#).unwrap_err();
        assert!(err.message.contains("from"), "got: {}", err.message);
    }

    #[test]
    fn rejects_unsupported_surgery_op() {
        let err = parse_surgery(r#"surgery s from "m.jac" { frobnicate weights }"#).unwrap_err();
        assert!(
            err.message.contains("not yet supported"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn parses_apply_lora() {
        let s = parse_surgery(
            r#"surgery s from "m.jac" { apply lora { rank = 16  alpha = 32  targets = w_q, w_k, w_v, w_o  source = "org/adapter" } }"#,
        )
        .unwrap();
        match &s.ops[0] {
            SurgeryOp::ApplyLora(l) => {
                assert_eq!(l.rank, 16);
                assert_eq!(l.alpha, 32.0);
                assert_eq!(l.targets, vec!["w_q", "w_k", "w_v", "w_o"]);
                assert_eq!(l.source, "org/adapter");
            }
            other => panic!("expected ApplyLora, got {other:?}"),
        }
    }

    #[test]
    fn lora_block_missing_field_errors() {
        let err = parse_surgery(
            r#"surgery s from "m.jac" { apply lora { rank = 8  targets = w_q  source = "a" } }"#,
        )
        .unwrap_err();
        assert!(err.message.contains("alpha"), "got: {}", err.message);
    }

    #[test]
    fn parses_model_level_lora_block() {
        let src = r#"model m {
          config { hidden = 8 layers = 1 heads = 2 head_dim = 4 vocab = 10 }
          weights { format = huggingface source = "x" dtype = fp16 }
          layer d { x = rmsnorm(a, b, c) }
          pipeline { greedy_sample }
          lora { rank = 8  alpha = 16  targets = w_q, w_o  source = "org/a" }
        }"#;
        let m = parse_model(src).unwrap();
        let l = m.lora.expect("lora block should parse");
        assert_eq!(l.rank, 8);
        assert_eq!(l.targets, vec!["w_q", "w_o"]);
    }

    #[test]
    fn rejects_backwards_range() {
        let err = parse_surgery(r#"surgery s from "m.jac" { prune layers 5..2 }"#).unwrap_err();
        assert!(err.message.contains("empty"));
    }

    const EXP: &str = r#"
experiment gemv_study {
  kind = kernel
  fixture   { op = "specs/gemv.jac" }
  subjects  { naive = jac(naive)  cublas = jac(cublas)  mine = cuda("kernels/my_gemv.cu") }
  reference = cublas
  sweep     { M = [1024, 4096]  N = [128, 512] }
  metrics   = [correctness, latency, memory]
}"#;

    #[test]
    fn parses_experiment() {
        use crate::ast::{ExpKind, SubjectSrc, SweepAxis};
        let e = parse_experiment(EXP).expect("should parse");
        assert_eq!(e.name, "gemv_study");
        assert_eq!(e.kind, ExpKind::Kernel);
        assert_eq!(
            e.fixture,
            vec![("op".to_string(), Value::Str("specs/gemv.jac".into()))]
        );
        assert_eq!(e.subjects.len(), 3);
        assert_eq!(e.subjects[0].name, "naive");
        assert_eq!(e.subjects[0].src, SubjectSrc::Jac("naive".into()));
        assert_eq!(
            e.subjects[2].src,
            SubjectSrc::Cuda("kernels/my_gemv.cu".into())
        );
        assert_eq!(e.reference, "cublas");
        assert_eq!(
            e.sweep,
            vec![
                SweepAxis {
                    name: "M".into(),
                    values: vec![1024, 4096]
                },
                SweepAxis {
                    name: "N".into(),
                    values: vec![128, 512]
                },
            ]
        );
        assert_eq!(
            e.metrics
                .iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>(),
            vec!["correctness", "latency", "memory"]
        );
    }

    #[test]
    fn experiment_requires_known_kind() {
        let err = parse_experiment("experiment e { kind = bogus  fixture {}  subjects {}  reference = x  sweep {}  metrics = [] }").unwrap_err();
        assert!(err.message.contains("kind"), "got: {}", err.message);
    }

    #[test]
    fn experiment_rejects_unknown_subject_ctor() {
        let err = parse_experiment(r#"experiment e { kind = kernel  fixture { op = "k.jac" }  subjects { a = magic(x) }  reference = a  sweep { M = [1] }  metrics = [latency] }"#).unwrap_err();
        assert!(err.message.contains("subject"), "got: {}", err.message);
    }

    #[test]
    fn experiment_rejects_empty_sweep_axis() {
        let err = parse_experiment(r#"experiment e { kind = kernel  fixture { op = "k.jac" }  subjects { a = jac(naive) }  reference = a  sweep { M = [] }  metrics = [latency] }"#).unwrap_err();
        assert!(
            err.message.contains("at least one value"),
            "got: {}",
            err.message
        );
    }
}
