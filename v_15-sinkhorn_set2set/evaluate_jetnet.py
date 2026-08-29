"""
evaluate_jet_pipeline.py
================================================================================
Unified evaluation for the AE + latent-flow jet-generation pipeline.

Run the SAME script on every trained variant (ChebNet K=1,2,3,5,10; single-hop
GCN/SAGE; different pooling/decoder) to get a directly comparable set of metrics.

It computes FOUR groups of numbers, because the ablation lives at four levels:

  (A) Standard JetNet generative metrics  -> comparability with the literature
        W1-M, W1-P (per feature), W1-EFP, FPD, KPD, FPND (JetNet-30 only),
        coverage & MMD.
  (B) Long-range / substructure observables -> your ACTUAL thesis metrics
        girth (jet width), pT-dispersion, mean pairwise angular scale (EEC-like),
        two-point energy correlator, + optional D2 via energyflow.
        The claim "ChebNet captures long-range correlations" must be shown HERE,
        not on marginal eta/phi/pt (which every model gets right).
  (C) AE reconstruction ceiling            -> encode real -> decode, no flow.
        Separates "bad reconstruction" from "bad generation".
  (D) Latent quality                       -> is the flow sampling the right z?
        per-dim W1(z_gen, z_enc), latent Frechet distance, and a CLASSIFIER
        TWO-SAMPLE TEST (AUC ~0.5 => flow latents indistinguishable from real).

Everything is wrapped in try/except so one API mismatch (jetnet version drift)
records NaN instead of killing the whole run on the cluster.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  1. Make your training script importable as a module (e.g. save it as
     `jet_pipeline.py` next to this file) so we can reuse the model classes and
     `sample_flow`. Only the CLASS DEFINITIONS and sample_flow are imported;
     no training runs on import (they are guarded by `if __name__ == "__main__"`).
  2. Register each trained variant in MODEL_REGISTRY below (name -> how to build
     the encoder + where the weights live).
  3. Run:
        python evaluate_jet_pipeline.py --model chebnet_K10 --num-particles 150
        python evaluate_jet_pipeline.py --model chebnet_K2  --num-particles 150
        ...
        python evaluate_jet_pipeline.py --aggregate      # build comparison table + plots
--------------------------------------------------------------------------------
"""

import os
import json
import argparse
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 0. IMPORT YOUR MODEL CLASSES  (edit the module name to match your file)
# ============================================================================
# These names must match the classes in your training script.
from gsoc_sinkhorn_set2set_ import (          # <-- rename `jet_pipeline` to your file's module name
    SpatialSet2SetChebNet,
    SlotDecoder,
    ConditionalFlowModel,
    sample_flow,
    LATENT_DIM,
    TOTAL_PARTICLES as TRAIN_TOTAL_PARTICLES,
)
from jetnet.datasets import JetNet
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data, Batch
from scipy.stats import wasserstein_distance

# jetnet metrics — imported lazily/defensively inside functions so a version
# mismatch on one metric doesn't block the others.
import jetnet


# ============================================================================
# 1. CONFIG
# ============================================================================
RESULTS_DIR = "./eval_results"          # where JSON + CSV + plots are written
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "comparison_table.csv")

N_EVAL          = 25000     # jets to generate/evaluate (>=50k ideal for FPD; 25k ok for dev)
KNN_K           = 10        # must match how each encoder was trained
BATCH_FLOW      = 2048
FLOW_STEPS      = 750
JET_TYPES       = ["g", "q", "t"]
TYPE_TO_ID      = {"g": 0, "q": 1, "t": 2}

# ---------------------------------------------------------------------------
# MODEL REGISTRY
# Each entry says: how to build the encoder, and where the 3 weight files are.
# `build_encoder` is a zero-arg callable returning an (untrained) encoder whose
# architecture matches the saved weights. Add one entry per ablation variant.
# ---------------------------------------------------------------------------
def _cheb_encoder(K):
    # Same SpatialSet2SetChebNet, only the Chebyshev order K changes.
    return lambda: SpatialSet2SetChebNet(K=K)

MODEL_REGISTRY = {
    "v_15": {
        "build_encoder": _cheb_encoder(10),   # K=10, matches v_15 training config
        "weights_dir":   "/pscratch/sd/r/rushil13/gsoc_rushil/v_15-sinkhorn_set2set/results_v15/weights",
        "note":          "v15 base, softmax-pT, ±0.8 eta/phi bounds",
    },
}


# ============================================================================
# 2. DATA + CONDITIONING
#    We evaluate CONDITIONALLY: take N real test jets, read off their
#    (pt, mass, mult, type) as flow conditioning, generate, and compare
#    real-vs-generated distributions like-for-like.
# ============================================================================
def load_test_jets(num_particles, data_dir):
    """Return (particle_data[N,P,4], jet_data[N,4], type_ids[N]) for the test split."""
    all_p, all_j, all_t = [], [], []
    for jtype in JET_TYPES:
        p, j = JetNet.getData(
            jet_type=[jtype], data_dir=data_dir,
            particle_features=["etarel", "phirel", "ptrel", "mask"],
            jet_features=["pt", "eta", "mass", "num_particles"],
            num_particles=num_particles, split="test", download=True,
        )
        all_p.append(p); all_j.append(j)
        all_t.append(np.full(len(p), TYPE_TO_ID[jtype], dtype=np.int64))
    return (np.concatenate(all_p), np.concatenate(all_j), np.concatenate(all_t))


def build_conditioning(jet_data, type_ids, num_particles):
    """Reproduce the training-time conditioning vector [minmax(pt), minmax(mass), mult/P]."""
    jet_pt, jet_mass, jet_mult = jet_data[:, 0], jet_data[:, 2], jet_data[:, 3] / num_particles
    mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    # NOTE: for perfect fidelity, save the train-set min/max and reuse them here.
    # Recomputing on the eval set is fine for RELATIVE model comparison.
    cont = np.stack([mm(jet_pt), mm(jet_mass), jet_mult], axis=1).astype(np.float32)
    return cont, type_ids.astype(np.int64)


def real_to_padded(particle_data, idx, num_particles):
    """Real jets -> [n, P, 4] padded (eta, phi, pt, mask), pt-sorted."""
    out = []
    for i in idx:
        jp = particle_data[i]
        m  = jp[:, 3] > 0.5
        vp = jp[m]
        vp = vp[np.argsort(-vp[:, 2])]
        pad = np.zeros((num_particles, 4), dtype=np.float32)
        pad[:len(vp)] = vp
        pad[:len(vp), 3] = 1.0
        out.append(pad)
    return np.stack(out)


def build_graph(hits):
    """kNN graph on (eta, phi) for a single jet's active particles."""
    k = KNN_K if len(hits) > KNN_K else len(hits) - 1
    adj = kneighbors_graph(hits[:, :2], n_neighbors=max(k, 1),
                           mode="connectivity", include_self=False)
    ei = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    return Data(x=torch.tensor(hits, dtype=torch.float), edge_index=ei)


# ============================================================================
# 3. LOAD A TRAINED PIPELINE
# ============================================================================
def load_pipeline(cfg):
    enc = cfg["build_encoder"]().to(DEVICE)
    dec = SlotDecoder(latent_dim=LATENT_DIM).to(DEVICE)
    wd  = cfg["weights_dir"]
    enc.load_state_dict(torch.load(os.path.join(wd, "gnn_encoder.pth"), map_location=DEVICE), strict=False)
    dec.load_state_dict(torch.load(os.path.join(wd, "slot_decoder.pth"), map_location=DEVICE), strict=False)
    flow = ConditionalFlowModel(emb_dim=LATENT_DIM).to(DEVICE)
    flow.load_state_dict(torch.load(os.path.join(wd, "cond_flow.pth"), map_location=DEVICE), strict=False)
    z_mean = np.load(os.path.join(wd, "z_mean.npy"))
    z_std  = np.load(os.path.join(wd, "z_std.npy"))
    enc.eval(); dec.eval(); flow.eval()
    return enc, dec, flow, z_mean, z_std


# ============================================================================
# 4. GENERATION  +  ENCODER LATENTS
# ============================================================================
@torch.no_grad()
def generate(dec, flow, cont, type_ids, z_mean, z_std):
    z = sample_flow(flow, cont, type_ids, n_steps=FLOW_STEPS, batch_size=BATCH_FLOW)
    z = z * z_std + z_mean
    parts = []
    for i in range(0, len(z), BATCH_FLOW):
        zb = torch.tensor(z[i:i+BATCH_FLOW], dtype=torch.float32, device=DEVICE)
        parts.append(dec(zb).cpu().numpy())
    return np.concatenate(parts, 0), z            # gen jets [N,P,4], gen latents [N,512] (denormed)


@torch.no_grad()
def encode_real(enc, particle_data, idx, num_particles):
    """Encoder latents for real jets (for latent two-sample test)."""
    valid = []
    for i in idx:
        jp = particle_data[i]
        m  = jp[:, 3] > 0.5
        vp = jp[m]; vp = vp[np.argsort(-vp[:, 2])]
        if len(vp) < 2:
            continue
        valid.append(build_graph(vp[:, :3]))
    z_list = []
    for i in range(0, len(valid), BATCH_FLOW):
        b = Batch.from_data_list(valid[i:i+BATCH_FLOW]).to(DEVICE)
        z_list.append(enc(b.x, b.edge_index, b.batch).cpu().numpy())
    return np.concatenate(z_list, 0) if z_list else np.zeros((0, LATENT_DIM))


@torch.no_grad()
def reconstruct(enc, dec, particle_data, idx, num_particles):
    """Encode->decode real jets (no flow): the AE reconstruction ceiling."""
    recon = []
    for i in idx:
        jp = particle_data[i]; m = jp[:, 3] > 0.5
        vp = jp[m]; vp = vp[np.argsort(-vp[:, 2])]
        if len(vp) < 2:
            recon.append(np.zeros((num_particles, 4), np.float32)); continue
        g = Batch.from_data_list([build_graph(vp[:, :3])]).to(DEVICE)
        z = enc(g.x, g.edge_index, g.batch)
        recon.append(dec(z).cpu().numpy()[0])
    return np.stack(recon)


# ============================================================================
# 5. FORMAT: [N,P,4]->[N,P,3] (eta,phi,pt), inactive zeroed, pt-sorted
# ============================================================================
def to_jetnet(parts):
    out = parts[:, :, :3].copy()
    mask = parts[:, :, 3] > 0.5
    out[~mask] = 0.0
    order = np.argsort(-out[:, :, 2], axis=1)          # pt-descending
    out = np.take_along_axis(out, order[:, :, None], axis=1)
    return out.astype(np.float32)


# ============================================================================
# 6A. STANDARD JETNET METRICS
# ============================================================================
def standard_metrics(real3, gen3, num_particles, type_ids_real=None):
    R = {}
    def safe(name, fn):
        try:
            R[name] = float(np.atleast_1d(fn())[0])
        except Exception as e:
            R[name] = float("nan"); R[name + "_err"] = str(e)[:120]

    safe("w1m", lambda: jetnet.evaluation.w1m(gen3, real3, num_eval_samples=min(N_EVAL, len(gen3))))
    # w1p returns per-feature W1 (eta, phi, pt) in recent versions -> average ourselves
    try:
        w1p = jetnet.evaluation.w1p(gen3, real3, num_eval_samples=min(N_EVAL, len(gen3)))
        w1p = np.atleast_1d(w1p[0] if isinstance(w1p, tuple) else w1p)
        R["w1p_avg"] = float(np.mean(w1p))
        for k, v in zip(["w1p_eta", "w1p_phi", "w1p_pt"], np.ravel(w1p)):
            R[k] = float(v)
    except Exception as e:
        R["w1p_avg"] = float("nan"); R["w1p_err"] = str(e)[:120]
    safe("w1efp", lambda: np.mean(np.atleast_1d(
        jetnet.evaluation.w1efp(gen3, real3, num_eval_samples=min(N_EVAL, len(gen3)))[0])))

    # FPD / KPD operate on EFP features; compute via jetnet.utils.efps then pass in.
    try:
        efp_r = jetnet.utils.efps(real3, efpset_args=[("n==", 4), ("d==", 4)])
        efp_g = jetnet.utils.efps(gen3,  efpset_args=[("n==", 4), ("d==", 4)])
        R["fpd"] = float(np.atleast_1d(jetnet.evaluation.fpd(efp_r, efp_g)[0]))
        R["kpd"] = float(np.atleast_1d(jetnet.evaluation.kpd(efp_r, efp_g)[0]))
    except Exception as e:
        R["fpd"] = float("nan"); R["kpd"] = float("nan"); R["fpd_err"] = str(e)[:120]

    try:
        cov, mmd = jetnet.evaluation.cov_mmd(real3, gen3, num_eval_samples=min(100, len(gen3)))
        R["coverage"] = float(cov); R["mmd"] = float(mmd)
    except Exception as e:
        R["coverage"] = float("nan"); R["mmd"] = float("nan")

    # FPND: JetNet-30 only, per single jet type.
    if num_particles == 30 and type_ids_real is not None:
        for jt, tid in TYPE_TO_ID.items():
            sel = np.where(type_ids_real == tid)[0]
            if len(sel) < 5000:
                R[f"fpnd_{jt}"] = float("nan"); continue
            try:
                R[f"fpnd_{jt}"] = float(jetnet.evaluation.fpnd(gen3[sel], jet_type=jt))
            except Exception as e:
                R[f"fpnd_{jt}"] = float("nan"); R[f"fpnd_{jt}_err"] = str(e)[:120]
    return R


# ============================================================================
# 6B. LONG-RANGE / SUBSTRUCTURE OBSERVABLES  (your thesis metrics)
#     All numpy-only and dependency-free except the optional energyflow block.
# ============================================================================
def _per_jet_longrange(jets3):
    """Return dict of per-jet arrays: girth, ptD, mean angular scale, EEC scalar."""
    girth, ptD, ang, eec = [], [], [], []
    for j in jets3:
        m = j[:, 2] > 0
        eta, phi, pt = j[m, 0], j[m, 1], j[m, 2]
        if len(pt) < 2:
            continue
        pt = pt / (pt.sum() + 1e-12)
        r = np.sqrt(eta**2 + phi**2)                       # dist from jet axis (rel coords)
        girth.append(float((pt * r).sum()))                # jet width / girth
        ptD.append(float(np.sqrt((pt**2).sum())))          # pT dispersion
        # pairwise angular structure (two-point / EEC-like), pt*pt weighted dR
        de = eta[:, None] - eta[None, :]
        dp = phi[:, None] - phi[None, :]
        dR = np.sqrt(de**2 + dp**2)
        w  = pt[:, None] * pt[None, :]
        iu = np.triu_indices(len(pt), k=1)
        wsum = w[iu].sum() + 1e-12
        ang.append(float((w[iu] * dR[iu]).sum() / wsum))   # pt-weighted mean pairwise dR
        eec.append(float((w[iu] * dR[iu]).sum()))          # unnormalized 2-pt energy correlator
    return {"girth": np.array(girth), "ptD": np.array(ptD),
            "ang_scale": np.array(ang), "eec": np.array(eec)}


def longrange_metrics(real3, gen3):
    R = {}
    rr = _per_jet_longrange(real3)
    gg = _per_jet_longrange(gen3)
    for key in rr:
        try:
            R[f"w1_{key}"] = float(wasserstein_distance(rr[key], gg[key]))
            R[f"w1_{key}_norm"] = float(wasserstein_distance(rr[key], gg[key]) / (rr[key].std() + 1e-9))
        except Exception:
            R[f"w1_{key}"] = float("nan")

    # OPTIONAL: D2 via energyflow (guarded). D2 = e3 / e2^3 (ECF ratio, long-range sensitive).
    try:
        import energyflow as ef
        def ecf_ratios(jets):
            d2 = []
            ecf2 = ef.EFPSet("n==2", "d==2", measure="hadr")   # 2-pt
            ecf3 = ef.EFPSet("n==3", "d==3", measure="hadr")   # 3-pt
            for j in jets:
                m = j[:, 2] > 0
                z = j[m][:, [2, 0, 1]]                          # (pt, eta, phi)
                if len(z) < 3:
                    continue
                e2 = ecf2.compute(z); e3 = ecf3.compute(z)
                e2v = float(np.atleast_1d(e2).sum()); e3v = float(np.atleast_1d(e3).sum())
                if e2v > 0:
                    d2.append(e3v / (e2v**3 + 1e-12))
            return np.array(d2)
        R["w1_D2"] = float(wasserstein_distance(ecf_ratios(real3), ecf_ratios(gen3)))
    except Exception as e:
        R["w1_D2"] = float("nan"); R["D2_note"] = "energyflow not available / adjust ECF call"
    return R


# ============================================================================
# 6C. AE RECONSTRUCTION CEILING
# ============================================================================
def reconstruction_metrics(real3, recon3):
    R = {}
    try:
        R["recon_w1m"] = float(jetnet.evaluation.w1m(recon3, real3,
                              num_eval_samples=min(N_EVAL, len(recon3)))[0])
    except Exception:
        R["recon_w1m"] = float("nan")
    # per-feature reconstruction W1 on pooled particle clouds
    for fi, nm in enumerate(["eta", "phi", "pt"]):
        r = real3[:, :, fi][real3[:, :, 2] > 0]
        g = recon3[:, :, fi][recon3[:, :, 2] > 0]
        try:
            R[f"recon_w1_{nm}"] = float(wasserstein_distance(r, g))
        except Exception:
            R[f"recon_w1_{nm}"] = float("nan")
    return R


# ============================================================================
# 6D. LATENT QUALITY  ("is the flow producing good latents?")
# ============================================================================
def latent_metrics(z_enc, z_gen):
    R = {}
    n = min(len(z_enc), len(z_gen))
    if n < 100:
        return {"latent_note": "too few latents"}
    z_enc, z_gen = z_enc[:n], z_gen[:n]

    # per-dimension W1 (averaged over 512 dims), on standardized latents
    mu, sd = z_enc.mean(0), z_enc.std(0) + 1e-9
    ze, zg = (z_enc - mu) / sd, (z_gen - mu) / sd
    R["latent_w1_mean"] = float(np.mean([wasserstein_distance(ze[:, d], zg[:, d])
                                         for d in range(ze.shape[1])]))

    # latent Frechet distance (FID-style in latent space)
    def frechet(a, b):
        from scipy.linalg import sqrtm
        ma, mb = a.mean(0), b.mean(0)
        ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
        cc = sqrtm(ca @ cb)
        if np.iscomplexobj(cc):
            cc = cc.real
        return float(((ma - mb) ** 2).sum() + np.trace(ca + cb - 2 * cc))
    try:
        R["latent_frechet"] = frechet(z_enc, z_gen)
    except Exception:
        R["latent_frechet"] = float("nan")

    # CLASSIFIER TWO-SAMPLE TEST: AUC ~0.5 => flow latents indistinguishable.
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        X = np.vstack([ze, zg]); y = np.r_[np.zeros(n), np.ones(n)]
        clf = LogisticRegression(max_iter=2000)
        R["latent_c2st_auc"] = float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())
    except Exception as e:
        R["latent_c2st_auc"] = float("nan"); R["c2st_err"] = str(e)[:120]
    return R


# ============================================================================
# 7. RUN ONE MODEL
# ============================================================================
def evaluate_model(name, num_particles, data_dir):
    cfg = MODEL_REGISTRY[name]
    print(f"\n=== Evaluating {name}  (P={num_particles})  [{cfg.get('note','')}] ===")

    particle_data, jet_data, type_ids = load_test_jets(num_particles, data_dir)
    N = min(N_EVAL, len(particle_data))
    idx = np.random.RandomState(0).choice(len(particle_data), N, replace=False)
    particle_data = particle_data[idx]; jet_data = jet_data[idx]; type_ids = type_ids[idx]

    cont, tids = build_conditioning(jet_data, type_ids, num_particles)
    enc, dec, flow, z_mean, z_std = load_pipeline(cfg)

    gen_parts, z_gen = generate(dec, flow, cont, tids, z_mean, z_std)
    real_parts = real_to_padded(particle_data, np.arange(N), num_particles)
    recon_parts = reconstruct(enc, dec, particle_data, np.arange(N), num_particles)
    z_enc = encode_real(enc, particle_data, np.arange(N), num_particles)

    real3, gen3, recon3 = to_jetnet(real_parts), to_jetnet(gen_parts), to_jetnet(recon_parts)

    results = {"model": name, "num_particles": num_particles, "n_eval": int(N)}
    results.update(standard_metrics(real3, gen3, num_particles, type_ids))
    results.update(longrange_metrics(real3, gen3))
    results.update(reconstruction_metrics(real3, recon3))
    results.update(latent_metrics(z_enc, z_gen))

    with open(os.path.join(RESULTS_DIR, f"{name}_P{num_particles}.json"), "w") as f:
        json.dump(results, f, indent=2)
    _append_csv(results)
    print(json.dumps({k: v for k, v in results.items() if not k.endswith("err")}, indent=2))
    return results


def _append_csv(row):
    import csv
    exists = os.path.exists(CSV_PATH)
    prior_keys = []
    if exists:
        with open(CSV_PATH) as f:
            prior_keys = next(csv.reader(f), [])
    keys = sorted(set(prior_keys) | set(row.keys()))
    rows = []
    if exists:
        with open(CSV_PATH) as f:
            import csv as _csv
            rows = list(_csv.DictReader(f))
    rows.append({k: row.get(k, "") for k in keys})
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# ============================================================================
# 8. AGGREGATE -> comparison table + metric-vs-K plot
# ============================================================================
def aggregate():
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(CSV_PATH)
    key_metrics = ["w1m", "w1p_avg", "w1efp", "fpd", "kpd",
                   "w1_girth", "w1_ang_scale", "w1_eec", "w1_D2",
                   "recon_w1m", "latent_c2st_auc"]
    key_metrics = [m for m in key_metrics if m in df.columns]
    print("\n=== COMPARISON TABLE ===")
    print(df[["model", "num_particles"] + key_metrics].to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "comparison_clean.csv"), index=False)

    # metric-vs-K plot (parses K from chebnet_K<k> names); the money figure is
    # a long-range metric where the JetNet-150 curve should drop faster with K.
    def get_K(nm):
        import re
        m = re.search(r"K(\d+)", str(nm)); return int(m.group(1)) if m else None
    df["K"] = df["model"].map(get_K)
    sub = df.dropna(subset=["K"])
    if len(sub):
        for metric in ["w1_ang_scale", "w1_eec", "fpd", "w1m"]:
            if metric not in sub.columns:
                continue
            plt.figure(figsize=(7, 5))
            for P, g in sub.groupby("num_particles"):
                g = g.sort_values("K")
                plt.plot(g["K"], g[metric], "o-", label=f"JetNet-{int(P)}")
            plt.xlabel("Chebyshev order K (receptive field)"); plt.ylabel(metric)
            plt.title(f"{metric} vs K  — thesis: 150 curve drops faster with K")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"vsK_{metric}.png"), dpi=120)
            plt.close()
    print(f"\nWrote tables + plots to {RESULTS_DIR}/")


# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="key in MODEL_REGISTRY")
    ap.add_argument("--num-particles", type=int, default=150, choices=[30, 150])
    ap.add_argument("--data-dir", type=str, default="/pscratch/sd/r/rushil13/jetnet_data")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
    elif args.model:
        evaluate_model(args.model, args.num_particles, args.data_dir)
    else:
        # evaluate everything registered, both datasets
        for name in MODEL_REGISTRY:
            for P in (30, 150):
                try:
                    evaluate_model(name, P, args.data_dir)
                except Exception as e:
                    print(f"[skip] {name} P={P}: {e}")
        aggregate()