// Model surgery: prune TinyLlama from 22 to 17 layers by removing the first 3 and last 2.
//   jacq --surgery specs/slim_tinyllama.jac
//   -> slim_tinyllama.jac, slim_tinyllama_convert.py
surgery slim_tinyllama from "@specs/tinyllama.jac" {
  prune layers 0..2, 20..21
}
