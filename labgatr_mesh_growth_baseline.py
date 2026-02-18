#!/usr/bin/env python3
"""
LaBGATr baseline for aorta mesh growth prediction (OBJ directory scan + time conditioning)

High-level idea
--------------
We want a baseline model that learns a vertex-wise deformation:
    V_src (N,3)  --->  V_tgt (N,3)

Your dataset:
- Each patient has multiple scans (meshes).
- Earliest scan t1 is treated as "source".
- Any later scan t2 is treated as a "target".
- Time interval dt (in months) is computed from (t2 - t1) and used to condition the model.

Why LaBGATr works here
----------------------
LaBGATr consumes point clouds (PyG Data) and predicts a per-token output.
We treat the mesh vertices as points and provide:
- pos: vertex coordinates
- orientation: per-vertex normals (used as an extra geometric feature)
- scalar_feature: dt_norm (time interval, normalized), repeated for all vertices

Important: LaBGATr expects pooling/tokenization metadata in PyG Data.
So we ALWAYS apply PointCloudPoolingScales to generate fields like:
- scale0_sampling_index
without which LaBGATr will crash.

Assumptions
-----------
- Vertex correspondence between source and target (same N and same ordering).
- Faces may differ between source and target. We do not require face equality.

Loss
----
- vertex L1 loss:
    loss = L1(V_pred, V_tgt)
"""

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torch_geometric as pyg

from lab_gatr import LaBGATr, PointCloudPoolingScales
from gatr.interface import embed_oriented_plane, extract_translation


# -------------------------
# OBJ IO
# -------------------------

def load_obj(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minimal OBJ loader.

    Supports:
      v x y z
      f i j k       (triangles)
      f i/... j/... k/... (OBJ indices may contain slashes)

    If a face has >3 vertices, we triangulate it by fan triangulation.

    Returns:
      V: (N,3) float32 vertices
      F: (M,3) int64 faces, 0-based indices
    """
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Vertex line
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])

            # Face line
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx: List[int] = []
                for p in parts:
                    # OBJ faces may be like "12/34/56" or "12//56" etc.
                    i = p.split("/")[0]
                    idx.append(int(i) - 1)  # OBJ is 1-based

                # If already triangle
                if len(idx) == 3:
                    faces.append(idx)
                else:
                    # Fan triangulation: (0,1,2), (0,2,3), ...
                    for t in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[t], idx[t + 1]])

    V = torch.tensor(verts, dtype=torch.float32)
    Fcs = torch.tensor(faces, dtype=torch.long)
    return V, Fcs


def save_obj(path: str, V: torch.Tensor, Fcs: torch.Tensor) -> None:
    """
    Save an OBJ mesh.
    We write:
      - all vertices
      - faces

    Note:
    If target faces differ, we still save using SOURCE faces (Fcs from source),
    because this baseline only predicts vertices.
    """
    V = V.detach().cpu()
    Fcs = Fcs.detach().cpu()

    with open(path, "w") as f:
        for i in range(V.shape[0]):
            x, y, z = V[i].tolist()
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        for i in range(Fcs.shape[0]):
            a, b, c = Fcs[i].tolist()
            f.write(f"f {a+1} {b+1} {c+1}\n")


# -------------------------
# Geometry helpers
# -------------------------

def vertex_normals(V: torch.Tensor, Fcs: torch.Tensor) -> torch.Tensor:
    """
    Compute per-vertex normals from faces.

    Why normals:
      LaBGATr expects an "orientation" vector per point for embed_oriented_plane().
      For meshes, a simple way is to use vertex normals.

    How it works:
      - compute per-face normal via cross product
      - accumulate face normals to vertices (index_add_)
      - normalize

    Note:
      If faces differ between timepoints, we compute normals using SOURCE faces only.
      That’s fine because normals are input features, and you only require vertex correspondence.
    """
    v0 = V[Fcs[:, 0]]
    v1 = V[Fcs[:, 1]]
    v2 = V[Fcs[:, 2]]

    # Face normals
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn = F.normalize(fn, dim=-1)

    # Accumulate to vertices
    vn = torch.zeros_like(V)
    vn.index_add_(0, Fcs[:, 0], fn)
    vn.index_add_(0, Fcs[:, 1], fn)
    vn.index_add_(0, Fcs[:, 2], fn)

    # Normalize per-vertex
    vn = F.normalize(vn, dim=-1)
    return vn


# -------------------------
# Time handling
# -------------------------

def parse_yyyymmdd(s: str) -> date:
    """Parse 'YYYYMMDD' string into python date."""
    return datetime.strptime(s, "%Y%m%d").date()


def months_between(d1: date, d2: date) -> float:
    """
    Compute a floating month difference using average month length.

    Why average month:
      It gives a smooth scalar (months) rather than integer months.
    """
    days = (d2 - d1).days
    return float(days) / 30.436875  # average days/month


def dt_norm_from_months(dt_months: float, dt_scale_months: float = 12.0) -> float:
    """
    Normalize the time interval.

    Example:
      dt_months=6, dt_scale_months=12  => dt_norm=0.5

    Why normalize:
      A scalar in a small range (e.g., 0..2) is easier for networks than raw months.
    """
    return float(dt_months) / float(dt_scale_months)


# -------------------------
# Scan directory
# -------------------------

# Matches e.g. "PTAAP013_20190903.obj"
NAME_RE = re.compile(r"^(?P<pid>.+?)_(?P<date>\d{8})\.obj$", re.IGNORECASE)

@dataclass(frozen=True)
class MeshRecord:
    """
    Represents one mesh file on disk, with parsed patient_id and acquisition date.
    """
    patient_id: str
    acq_date: date
    path: str

@dataclass(frozen=True)
class PairItem:
    """
    Represents one training pair (src -> tgt) and its time interval dt_months.
    """
    patient_id: str
    src_path: str
    tgt_path: str
    dt_months: float


def scan_mesh_dir(mesh_dir: str) -> List[MeshRecord]:
    """
    Scan mesh_dir for OBJ files matching <pid>_<yyyymmdd>.obj and parse them.
    """
    recs: List[MeshRecord] = []
    for fn in os.listdir(mesh_dir):
        m = NAME_RE.match(fn)
        if not m:
            continue
        pid = m.group("pid")
        d = parse_yyyymmdd(m.group("date"))
        recs.append(MeshRecord(patient_id=pid, acq_date=d, path=os.path.join(mesh_dir, fn)))

    if not recs:
        raise RuntimeError(f"No OBJ files matched '<patient>_YYYYMMDD.obj' in: {mesh_dir}")
    return recs


def build_pairs(records: List[MeshRecord]) -> List[PairItem]:
    """
    Build training pairs per patient:
      src = earliest timepoint
      tgt = every later timepoint
    """
    by_pid: Dict[str, List[MeshRecord]] = {}
    for r in records:
        by_pid.setdefault(r.patient_id, []).append(r)

    pairs: List[PairItem] = []
    for pid, lst in by_pid.items():
        lst_sorted = sorted(lst, key=lambda x: x.acq_date)
        if len(lst_sorted) < 2:
            # patient with only one scan provides no (src->tgt) training pairs
            continue

        src = lst_sorted[0]
        for tgt in lst_sorted[1:]:
            dtm = months_between(src.acq_date, tgt.acq_date)
            pairs.append(PairItem(pid, src.path, tgt.path, dtm))

    if not pairs:
        raise RuntimeError("No training pairs found (need patients with >=2 timepoints).")
    return pairs


def choose_patient_target(
    records: List[MeshRecord], patient_id: str, target_date: Optional[str]
) -> Tuple[MeshRecord, MeshRecord, float]:
    """
    For inference:
      - src = earliest scan of this patient
      - tgt = latest scan if target_date is None, else the scan with that date

    Returns: (src_record, tgt_record, dt_months)
    """
    pts = [r for r in records if r.patient_id == patient_id]
    if not pts:
        raise ValueError(f"No meshes found for patient_id='{patient_id}'")

    pts = sorted(pts, key=lambda x: x.acq_date)
    src = pts[0]

    if target_date is None:
        tgt = pts[-1]
    else:
        td = parse_yyyymmdd(target_date)
        exact = [r for r in pts if r.acq_date == td]
        if not exact:
            available = ", ".join([p.acq_date.strftime("%Y%m%d") for p in pts])
            raise ValueError(f"target_date {target_date} not found for {patient_id}. Available: {available}")
        tgt = exact[0]

    dtm = months_between(src.acq_date, tgt.acq_date)
    return src, tgt, dtm


# -------------------------
# LaBGATr interface
# -------------------------

class MeshGrowthGAInterface:
    """
    This interface is the key “adapter” between your PyG Data and the GATr/LaBGATr internals.

    LaBGATr internally operates on:
      multivectors: (N, C, 16)  -> 16D representation of geometric algebra elements
      scalars:      (N, S)      -> extra scalar channels

    We define:
      - how to embed your mesh vertices into multivectors + scalars (embed)
      - how to decode model outputs back to Euclidean displacement (dislodge)

    Time conditioning:
      - scalars contain dt_norm (same scalar repeated for all vertices)
      - model predicts displacement *rate* (per unit dt_norm)
      - final displacement = rate * dt_norm
    """
    num_input_channels = 1
    num_output_channels = 1
    num_input_scalars = 1
    num_output_scalars = 1

    @staticmethod
    @torch.no_grad()
    def embed(data: pyg.data.Data):
        """
        Convert PyG data fields to GA inputs.

        - embed_oriented_plane(normal, position) produces a (N,16) multivector
          representing an oriented plane at each point.
        - We then add a channel dimension -> (N,1,16)

        scalars:
        - scalar_feature is dt_norm repeated per vertex -> (N,1)
        """
        mv = embed_oriented_plane(normal=data.orientation, position=data.pos)  # (N,16)
        multivectors = mv.view(-1, 1, 16)
        scalars = data.scalar_feature.view(-1, 1)  # dt_norm repeated
        return multivectors, scalars

    @staticmethod
    def dislodge(multivectors: torch.Tensor, scalars: torch.Tensor):
        """
        Decode model outputs.

        extract_translation(multivectors) converts a multivector to a 3D translation vector.

        We interpret that output as a displacement RATE, then scale by dt_norm:

          disp = disp_rate * dt_norm
        """
        disp_rate = extract_translation(multivectors).squeeze()  # (N,3)
        dt_norm = scalars[:, 0:1]                                # (N,1)
        return disp_rate * dt_norm


def make_pyg_data(
    V_src: torch.Tensor,
    F_src: torch.Tensor,
    dt_months: float,
    device: torch.device,
    dt_scale_months: float,
) -> pyg.data.Data:
    """
    Build the PyG Data object that LaBGATr expects.

    Fields:
      pos            : (N,3) vertex coordinates (source)
      orientation    : (N,3) vertex normals from source faces
      scalar_feature : (N,)  dt_norm repeated

    We also attach:
      faces : source faces (only used for normals and saving output)
    """
    V_src = V_src.to(device)
    F_src = F_src.to(device)

    ori = vertex_normals(V_src, F_src)

    dt_norm = dt_norm_from_months(dt_months, dt_scale_months=dt_scale_months)
    scalar_feature = torch.full((V_src.shape[0],), float(dt_norm), dtype=torch.float32, device=device)

    data = pyg.data.Data(pos=V_src, orientation=ori, scalar_feature=scalar_feature)

    # Keep SOURCE faces only for saving output / computing normals
    data.faces = F_src
    return data


# -------------------------
# Dataset
# -------------------------

class MeshPairDataset(torch.utils.data.Dataset):
    """
    Dataset yields ONE pair:
      (V_src, V_tgt, F_src, dt_months, patient_id)

    Note:
      We do NOT return date objects because DataLoader's default collate can’t handle them.
      Here we only return numeric dt_months (float).
    """
    def __init__(self, pairs: List[PairItem]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        V_src, F_src = load_obj(p.src_path)
        V_tgt, _F_tgt = load_obj(p.tgt_path)

        # Require vertex correspondence only
        if V_src.shape != V_tgt.shape:
            raise ValueError(f"[{p.patient_id}] vertex mismatch: {V_src.shape} vs {V_tgt.shape}")

        return V_src, V_tgt, F_src, float(p.dt_months), p.patient_id


# -------------------------
# Train / Infer
# -------------------------

def build_model(device: torch.device, d_model: int, num_blocks: int, num_heads: int) -> torch.nn.Module:
    """
    Construct the LaBGATr model.

    Key hyperparameters:
      d_model       : hidden width (bigger = more capacity, slower)
      num_blocks    : transformer depth
      num_attn_heads: multi-head attention heads
      use_class_token=False:
        - we want per-vertex outputs (a displacement for each vertex)
    """
    return LaBGATr(
        MeshGrowthGAInterface,
        d_model=d_model,
        num_blocks=num_blocks,
        num_attn_heads=num_heads,
        use_class_token=False,
    ).to(device)


def train(
    mesh_dir: str,
    out_dir: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    d_model: int,
    num_blocks: int,
    num_heads: int,
    pool_ratio: float,
    dt_scale_months: float,
    device_str: str,
):
    """
    Training loop.

    Important detail:
      LaBGATr's internal tokeniser expects fields like `scale0_sampling_index`,
      so we ALWAYS apply PointCloudPoolingScales to create them.

    pool_ratio:
      - if 1.0, you *keep all points as tokens*, but still create metadata fields
      - if <1.0, you subsample points and interpolate features (faster/cheaper)
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    records = scan_mesh_dir(mesh_dir)
    pairs = build_pairs(records)
    print(f"Found {len(records)} meshes, {len(pairs)} (source->target) pairs across patients.")

    ds = MeshPairDataset(pairs)

    # batch_size=1 is safest because different meshes could have different sizes in other datasets.
    # collate_fn=lambda x: x returns raw list; we use batch[0].
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, collate_fn=lambda x: x)

    # Always apply pooling so LaBGATr gets required metadata fields
    pool = PointCloudPoolingScales(rel_sampling_ratios=(pool_ratio,), interp_simplex="triangle")

    model = build_model(device, d_model=d_model, num_blocks=num_blocks, num_heads=num_heads)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for batch in dl:
            V_src, V_tgt, F_src, dt_months, pid = batch[0]

            # Make PyG Data (pos/orientation/scalar_feature)
            data = make_pyg_data(V_src, F_src, dt_months, device=device, dt_scale_months=dt_scale_months)

            # Create pooling/tokenization metadata expected by LaBGATr
            data_in = pool(data)

            # Target vertices on device
            V_tgt = V_tgt.to(device)

            # Forward: model outputs per-vertex displacement (N,3)
            disp = model(data_in)
            V_pred = data_in.pos + disp

            # ONLY L1 vertex loss (no regularizers)
            loss = F.l1_loss(V_pred, V_tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())

        avg = running / max(1, len(dl))
        print(f"Epoch {ep:03d} | loss={avg:.6f}")

        if avg < best:
            best = avg
            torch.save({"state_dict": model.state_dict()}, os.path.join(out_dir, "labgatr_best.pth"))

    torch.save({"state_dict": model.state_dict()}, os.path.join(out_dir, "labgatr_last.pth"))
    print("Saved checkpoints to:", out_dir)

# # Inference for one patient
# infer --mesh_dir /mnt/storage/home/lchen6/lchen6/data/TAAMesh/MeshALL_10000 --ckpt runs/labgatr_growth/labgatr_best.pth --patient_id PTAAP013 --out_obj pred_PTAAP013_latest.obj --device cuda
@torch.no_grad()
def infer(
    mesh_dir: str,
    ckpt_path: str,
    patient_id: str,
    out_obj: str,
    target_date: Optional[str],
    d_model: int,
    num_blocks: int,
    num_heads: int,
    pool_ratio: float,
    dt_scale_months: float,
    device_str: str,
):
    """
    Inference / prediction:
      - load checkpoint
      - choose src mesh (earliest) and dt months to target (given date or latest)
      - run model and save predicted vertices as OBJ
    """
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    records = scan_mesh_dir(mesh_dir)
    src_rec, tgt_rec, dt_months = choose_patient_target(records, patient_id=patient_id, target_date=target_date)
    print(f"[{patient_id}] source={src_rec.acq_date}  target={tgt_rec.acq_date}  dt≈{dt_months:.2f} months")

    V_src, F_src = load_obj(src_rec.path)

    model = build_model(device, d_model=d_model, num_blocks=num_blocks, num_heads=num_heads)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    pool = PointCloudPoolingScales(rel_sampling_ratios=(pool_ratio,), interp_simplex="triangle")

    data = make_pyg_data(V_src, F_src, dt_months, device=device, dt_scale_months=dt_scale_months)
    data_in = pool(data)

    disp = model(data_in)
    V_pred = data_in.pos + disp

    # Save with SOURCE faces (target faces may differ)
    save_obj(out_obj, V_pred, F_src)
    print("Wrote predicted mesh to:", out_obj)



@torch.no_grad()
def infer_all_patients(
    mesh_dir: str,
    ckpt_path: str,
    out_dir: str,
    d_model: int,
    num_blocks: int,
    num_heads: int,
    pool_ratio: float,
    dt_scale_months: float,
    device_str: str,
):
    """
    Batch inference over ALL patients.

    For each patient:
      - pick earliest scan as source t1
      - for each later scan tk (k>=2):
          * compute dt_months = months_between(t1, tk)
          * predict V_pred from V_src(t1) with dt condition
          * save as: <PATIENTID>_<YYYYMMDD>_pred.obj
            where YYYYMMDD is the TARGET scan date (tk)

    Output meshes are written using SOURCE faces (t1 faces).
    """
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    # 1) Scan directory and group by patient
    records = scan_mesh_dir(mesh_dir)

    by_pid: Dict[str, List[MeshRecord]] = {}
    for r in records:
        by_pid.setdefault(r.patient_id, []).append(r)

    # 2) Load model checkpoint
    model = build_model(device, d_model=d_model, num_blocks=num_blocks, num_heads=num_heads)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3) Pool transform (ALWAYS needed for LaBGATr)
    pool = PointCloudPoolingScales(rel_sampling_ratios=(pool_ratio,), interp_simplex="triangle")

    total_preds = 0
    total_patients = 0

    # 4) Iterate patients
    for pid, lst in sorted(by_pid.items(), key=lambda x: x[0]):
        lst_sorted = sorted(lst, key=lambda x: x.acq_date)
        if len(lst_sorted) < 2:
            # no targets available for this patient
            continue

        total_patients += 1

        # Source is earliest scan t1
        src = lst_sorted[0]
        V_src, F_src = load_obj(src.path)

        # Predict for each later scan date
        for tgt in lst_sorted[1:]:
            dt_months = months_between(src.acq_date, tgt.acq_date)

            # Build input PyG Data (source vertices + normals + dt scalar)
            data = make_pyg_data(
                V_src=V_src,
                F_src=F_src,
                dt_months=dt_months,
                device=device,
                dt_scale_months=dt_scale_months,
            )

            # Add pooling/tokenization metadata required by LaBGATr
            data_in = pool(data)

            # Forward -> displacement -> predicted vertices
            disp = model(data_in)               # (N,3)
            V_pred = data_in.pos + disp         # (N,3)

            # Save: PATIENTID_YYYYMMDD_pred.obj (target date)
            out_name = f"{pid}_{tgt.acq_date.strftime('%Y%m%d')}_pred.obj"
            out_path = os.path.join(out_dir, out_name)
            save_obj(out_path, V_pred, F_src)

            total_preds += 1

        print(f"[{pid}] wrote {len(lst_sorted) - 1} predictions.")

    print(f"Done. Patients processed: {total_patients}, predictions saved: {total_preds}")
    print("Output directory:", out_dir)




def main():
    """
    CLI entry point.

    train:
      python labgatr_mesh_growth_baseline.py train --mesh_dir ... --out_dir ...

    infer:
      python labgatr_mesh_growth_baseline.py infer --mesh_dir ... --ckpt ... --patient_id ...
    """
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train")
    ap_tr.add_argument("--mesh_dir", required=True)
    ap_tr.add_argument("--out_dir", required=True)
    ap_tr.add_argument("--epochs", type=int, default=50)
    ap_tr.add_argument("--lr", type=float, default=1e-4)
    ap_tr.add_argument("--weight_decay", type=float, default=1e-4)
    ap_tr.add_argument("--d_model", type=int, default=64)
    ap_tr.add_argument("--num_blocks", type=int, default=6)
    ap_tr.add_argument("--num_heads", type=int, default=4)
    ap_tr.add_argument("--pool_ratio", type=float, default=1.0)
    ap_tr.add_argument("--dt_scale_months", type=float, default=12.0)
    ap_tr.add_argument("--device", type=str, default="cuda")

    ap_in = sub.add_parser("infer")
    ap_in.add_argument("--mesh_dir", required=True)
    ap_in.add_argument("--ckpt", required=True)
    ap_in.add_argument("--out_dir", required=True)
    ap_in.add_argument("--d_model", type=int, default=64)
    ap_in.add_argument("--num_blocks", type=int, default=6)
    ap_in.add_argument("--num_heads", type=int, default=4)
    ap_in.add_argument("--pool_ratio", type=float, default=1.0)
    ap_in.add_argument("--dt_scale_months", type=float, default=12.0)
    ap_in.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    if args.cmd == "train":
        train(
            mesh_dir=args.mesh_dir,
            out_dir=args.out_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            d_model=args.d_model,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            pool_ratio=args.pool_ratio,
            dt_scale_months=args.dt_scale_months,
            device_str=args.device,
        )
    else:
        infer_all_patients(
            mesh_dir=args.mesh_dir,
            ckpt_path=args.ckpt,
            out_dir=args.out_dir,
            d_model=args.d_model,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            pool_ratio=args.pool_ratio,
            dt_scale_months=args.dt_scale_months,
            device_str=args.device,
        )




if __name__ == "__main__":
    main()

# Example commands (Linux paths):
# infer --mesh_dir /mnt/storage/home/lchen6/lchen6/data/TAAMesh/MeshALL_10000 --ckpt runs/labgatr_growth/labgatr_best.pth --out_dir runs/labgatr_growth/preds_all --device cuda
# train --mesh_dir /mnt/storage/home/lchen6/lchen6/data/TAAMesh/MeshALL_10000 --out_dir runs/labgatr_growth865 --pool_ratio 1.0 --epochs 50 --device cuda


# train --mesh_dir /mnt/storage/home/lchen6/lchen6/data/TAAMesh/MeshALL_10000 \
#       --out_dir runs/labgatr_growth --pool_ratio 1.0 --epochs 50 --device cuda
#
# infer --mesh_dir /mnt/storage/home/lchen6/lchen6/data/TAAMesh/MeshALL_10000 \
#       --ckpt runs/labgatr_growth/labgatr_best.pth --patient_id PTAAP013 --target_date 20200310 \
#       --out_obj pred_PTAAP013_20200310.obj --device cuda
#
# infer --mesh_dir /mnt/storage/home/lchen6/lchen6/data/TAAMesh/MeshALL_10000 \
#       --ckpt runs/labgatr_growth/labgatr_best.pth --patient_id PTAAP013 \
#       --out_obj pred_PTAAP013_latest.obj --device cuda
