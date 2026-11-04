//! Abstract syntax tree for the Jacquard **model** DSL.
//!
//! Scope for Phase 0: `model` blocks with `config`, `weights`, `layer`, and `pipeline`
//! sections. The kernel DSL and the strategy/surgery/experiment constructs land in later
//! phases and will extend this module.

/// A literal value as it appears in `config`/`weights` entries.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    /// A bareword such as `huggingface`, `fp16`, `silu`.
    Ident(String),
    Str(String),
}

/// A whole `model NAME { ... }` definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub name: String,
    /// `config { ... }` entries, in source order.
    pub config: Vec<(String, Value)>,
    /// `weights { ... }` entries, in source order.
    pub weights: Vec<(String, Value)>,
    /// `layer NAME { ... }` definitions.
    pub layers: Vec<Layer>,
    /// `pipeline { step -> step -> ... }`, each step an expression (call or bareword).
    pub pipeline: Vec<Expr>,
    /// Optional `lora { ... }` block — low-rank adapters injected after target linears.
    pub lora: Option<LoraSpec>,
}

/// A `lora { rank, alpha, targets, source }` adapter specification (also produced by the
/// `apply lora` surgery op). `targets` names base linears: `w_q`, `w_k`, `w_v`, `w_o`.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraSpec {
    pub rank: i64,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub source: String,
}

impl Model {
    /// Look up a config entry by key.
    pub fn config_value(&self, key: &str) -> Option<&Value> {
        self.config.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Look up a weights entry by key.
    pub fn weights_value(&self, key: &str) -> Option<&Value> {
        self.weights.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

/// A `layer NAME { ... }` block: a sequence of single-assignment statements.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub name: String,
    pub stmts: Vec<Stmt>,
}

/// One layer statement: `lhs0, lhs1, ... = expr`.
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    /// Assignment targets (more than one for fused ops like `q, k, v = linear(...)`).
    pub targets: Vec<String>,
    pub value: Expr,
}

/// Binary operators used in layer expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Mul,
}

/// A `surgery NAME from "PATH" { ops }` model-transformation spec.
#[derive(Debug, Clone, PartialEq)]
pub struct Surgery {
    pub name: String,
    /// The source spec path as written (may begin with `@` for project-root-relative).
    pub from: String,
    pub ops: Vec<SurgeryOp>,
}

/// A single surgery operation. Phase 2 implements `prune layers`, `cast weights to <dtype>`, and
/// `update config { ... }`; remaining ops (`merge`, `apply lora`) parse-error for now.
#[derive(Debug, Clone, PartialEq)]
pub enum SurgeryOp {
    /// `prune layers 0..2, 20..21` — remove these (inclusive) layer-index ranges.
    PruneLayers(Vec<LayerRange>),
    /// `cast weights to int8` — change the weights dtype (re-quantizing existing local weights).
    CastWeights(String),
    /// `update config { max_seq = 4096 }` — override config entries.
    UpdateConfig(Vec<(String, Value)>),
    /// `merge layers 11..16 from "donor.jac"` — replace these (inclusive) layer ranges with the
    /// donor model's weights at the same indices. Structure is unchanged; only weights differ.
    MergeLayers {
        ranges: Vec<LayerRange>,
        donor: String,
    },
    /// `apply lora { rank, alpha, targets, source }` — attach a LoRA adapter to the model.
    ApplyLora(LoraSpec),
}

/// An inclusive range of layer indices, e.g. `0..2` is layers 0, 1, and 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRange {
    pub start: i64,
    pub end: i64,
}

/// A `kernel NAME { ... }` definition in the kernel DSL (the second sub-language).
/// Phase 2 front-end: `tensors`, `params`, `constraints`, and phase headers are parsed; phase
/// bodies and the `state`/`fusion` sections land with kernel codegen.
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub tensors: Vec<TensorDecl>,
    pub state: Vec<StateDecl>,
    pub params: Vec<(String, i64)>,
    pub constraints: Vec<Constraint>,
    pub phases: Vec<Phase>,
    pub fusion: Vec<FusionDirective>,
}

/// A `state` declaration: `NAME ["[" dims "]"] : type @placement qualifier*`. `dtype` holds one
/// scalar type, or several for a tuple type `(fp32, int32)`. Dims empty = scalar.
#[derive(Debug, Clone, PartialEq)]
pub struct StateDecl {
    pub name: String,
    pub dims: Vec<String>,
    pub dtype: Vec<String>,
    pub placement: String,
    pub qualifiers: Vec<String>, // broadcast / atomic
}

/// A `fusion` directive: combine phases into one launch, or overlap operations across phases.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionDirective {
    /// `phaseA + phaseB + ... = single_kernel`
    Combine { phases: Vec<String>, mode: String },
    /// `phase.op overlaps_with phase.op`
    Overlap {
        a: (String, String),
        b: (String, String),
    },
}

/// A tensor declaration: `Name[d0, d1] : dtype @placement [-> access] [-> global]`.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDecl {
    pub name: String,
    pub dims: Vec<String>,
    pub dtype: String,
    pub placement: String,      // registers | smem | global
    pub access: Option<String>, // stream | gather (for @global inputs)
    pub output: bool,           // `-> global` output suffix
}

/// Comparison operator in a constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Ge,
    Le,
    Eq,
    Gt,
    Lt,
}

/// One side of a constraint: a number with an optional unit (e.g. `48KB`), or a symbol (`sm_80`).
#[derive(Debug, Clone, PartialEq)]
pub enum CVal {
    Num(i64, Option<String>),
    Sym(String),
}

/// A kernel constraint, e.g. `arch >= sm_80` or `smem_per_block <= 48KB`.
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub lhs: CVal,
    pub op: CmpOp,
    pub rhs: CVal,
}

/// A phase: `phase NAME [over T.dim(D) by TILE] [after OTHER] { stmt* }`.
#[derive(Debug, Clone, PartialEq)]
pub struct Phase {
    pub name: String,
    pub iteration: Option<IterSpec>,
    pub after: Option<String>,
    pub body: Vec<KStmt>,
}

/// `over TENSOR.dim(DIM) by TILE` — tile a phase over a tensor dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct IterSpec {
    pub tensor: String,
    pub dim: String,
    pub by: String,
}

/// One element of a `load` index list: a wildcard `*` or an index name (`tile`, a dim, …).
#[derive(Debug, Clone, PartialEq)]
pub enum IndexExpr {
    Wildcard,
    Name(String),
}

/// A statement inside a phase body. Phase 2 parses `load`, `einsum`, and `compare`; the streaming
/// statements (`online_softmax`, `gather`, `append`, …) land with the kernels that need them.
#[derive(Debug, Clone, PartialEq)]
pub enum KStmt {
    /// `load DST = SRC[idx, ...] [via PATH]` — load a tile slice into a local.
    Load {
        dst: String,
        src: String,
        index: Vec<IndexExpr>,
        via: Option<String>,
    },
    /// `DST = einsum("SUBSCRIPTS", op, ...)` — an Einstein-summation contraction.
    Einsum {
        dst: String,
        subscripts: String,
        operands: Vec<String>,
    },
    /// `DST = LHS <op> RHS` — a boolean mask from a relational comparison.
    Compare {
        dst: String,
        lhs: String,
        op: CmpOp,
        rhs: String,
    },
    /// `online_softmax SCORES -> ACC { V = .. running_max = .. running_sum = .. }` — streaming
    /// (flash-style) softmax with running max/sum, the core of single-pass decode attention.
    OnlineSoftmax {
        scores: String,
        acc: String,
        v: String,
        running_max: String,
        running_sum: String,
    },
    /// `DST = SRC.gather(INDEX)` — sparse indexed gather from a `@global -> gather` tensor.
    Gather {
        dst: String,
        src: String,
        index: String,
    },
    /// `DST = softmax(SRC, dim: DIM)` — numerically-stable softmax along a dimension.
    Softmax {
        dst: String,
        src: String,
        dim: String,
    },
}

/// An `experiment NAME { kind, fixture, subjects, reference, sweep, metrics }` definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Experiment {
    pub name: String,
    pub kind: ExpKind,
    /// Held-constant environment, as raw key→value (provider-interpreted). For `kind = kernel`
    /// this holds `op = "<path>"`.
    pub fixture: Vec<(String, Value)>,
    pub subjects: Vec<Subject>,
    pub reference: String,
    pub sweep: Vec<SweepAxis>,
    pub metrics: Vec<Metric>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpKind {
    Kernel,
}

/// A named subject and where its implementation comes from.
#[derive(Debug, Clone, PartialEq)]
pub struct Subject {
    pub name: String,
    pub src: SubjectSrc,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SubjectSrc {
    /// `jac(<variant>)` — a jac-generated variant of the fixture op (`naive`/`cublas`).
    Jac(String),
    /// `cuda("<path>")` — a user .cu implementing the op ABI.
    Cuda(String),
}

/// One sweep axis. Phase A: a literal integer list.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepAxis {
    pub name: String,
    pub values: Vec<i64>,
}

/// A metric request; `args` are reserved for parameterized metrics (e.g. `quality(judge: ...)`).
#[derive(Debug, Clone, PartialEq)]
pub struct Metric {
    pub name: String,
    pub args: Vec<(String, Value)>,
}

/// An expression in a layer body or pipeline step.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A bare identifier (variable, weight name, or nullary step like `greedy_sample`).
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
    /// `name(arg, arg, ...)` — an operation or pipeline step.
    Call {
        name: String,
        args: Vec<Expr>,
    },
    /// `lhs <op> rhs`, e.g. `silu(gate) * up`.
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
}
