// Model surgery: attach a LoRA adapter to TinyLlama's attention projections.
//   jacq --surgery specs/tinyllama_lora.jac   ->   tinyllama_lora.jac (+ convert script)
// NOTE: LoRA codegen is not yet implemented — `jacq --emit-model tinyllama_lora.jac` currently
// reports "lora codegen is not yet implemented (front-end only)". The spec/IR carry the adapter;
// runtime injection lands in a follow-up (see docs/PROGRESS.md LoRA plan).
surgery tinyllama_lora from "@specs/tinyllama.jac" {
  apply lora {
    rank    = 16
    alpha   = 32
    targets = w_q, w_k, w_v, w_o
    source  = "TinyLlama/TinyLlama-lora-example"
  }
}
