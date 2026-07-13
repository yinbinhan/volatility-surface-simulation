"""Fold LoRA-adapted fine-tuned checkpoint into a plain diffusion state_dict.

The fine-tuner wraps selected Linear layers with LoRALinear (base + low-rank
B@A*scaling). backtest_diffusion.py loads a plain ConditionalTransformer, so we
merge each LoRA update back into the base weight and strip the ".base" prefix,
yielding a checkpoint loadable by the unmodified backtest model.
"""
import argparse, torch

def fold_state_dict(sd, alpha=16.0, rank=8):
    scaling = alpha / rank
    out = {}
    # group lora keys by module prefix
    lora_prefixes = {k[:-len(".lora_A")] for k in sd if k.endswith(".lora_A")}
    consumed = set()
    for pref in lora_prefixes:
        A = sd[pref + ".lora_A"]; B = sd[pref + ".lora_B"]
        W = sd[pref + ".base.weight"]
        merged = W + scaling * (B @ A)
        out[pref + ".weight"] = merged
        consumed |= {pref + ".lora_A", pref + ".lora_B", pref + ".base.weight"}
        if pref + ".base.bias" in sd:
            out[pref + ".weight" if False else pref + ".bias"] = sd[pref + ".base.bias"]
            consumed.add(pref + ".base.bias")
    for k, v in sd.items():
        if k in consumed:
            continue
        out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--rank", type=int, default=8)
    a = ap.parse_args()
    try:
        ckpt = torch.load(a.inp, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(a.inp, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    folded = fold_state_dict(sd, a.alpha, a.rank)
    torch.save({"model": folded}, a.out)
    print(f"folded {len(sd)} -> {len(folded)} keys, saved {a.out}")

if __name__ == "__main__":
    main()
