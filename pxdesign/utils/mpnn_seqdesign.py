# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Post-diffusion MPNN sequence design with optional per-position bias.
Runs ProteinMPNN on diffused backbone CIFs and writes summary.csv for CASCADE.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from biotite.structure import get_residues
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile

logger = logging.getLogger(__name__)


def _cif_to_pdb(cif_path: str, pdb_path: str) -> None:
    """Convert CIF to PDB using biotite."""
    cif = pdbx.CIFFile.read(cif_path)
    atom_array = pdbx.get_structure(cif, model=1)
    pdb_file = PDBFile()
    pdb_file.set_structure(atom_array)
    pdb_file.write(pdb_path)


def _get_chain_ids(atom_array) -> list[str]:
    """Return sorted list of chain IDs."""
    chain_ids = []
    for res in get_residues(atom_array):
        cid = str(res.chain_id[0])
        if cid not in chain_ids:
            chain_ids.append(cid)
    return chain_ids


def _run_proteinmpnn(
    pdb_path: str,
    out_folder: str,
    path_to_model_weights: str,
    designed_chain_id: str,
    all_chain_ids: list[str],
    bias_by_res_jsonl: str | None = None,
    num_seq: int = 1,
    sampling_temp: str = "0.1",
) -> list[str]:
    """
    Run ProteinMPNN and return designed sequences.
    Returns list of sequence strings from the output FASTA.
    """
    mpnn_run = Path(path_to_model_weights) / "protein_mpnn_run.py"
    if not mpnn_run.exists():
        raise FileNotFoundError(
            f"ProteinMPNN run script not found at {mpnn_run}. "
            "Run download_tool_weights.sh to fetch it."
        )
    model_dir = str(Path(path_to_model_weights).parent)

    pdb_name = Path(pdb_path).stem
    fixed_chains = [c for c in all_chain_ids if c != designed_chain_id]
    chain_id_dict = {pdb_name: ([designed_chain_id], fixed_chains)}
    chain_id_path = os.path.join(out_folder, "chain_id.jsonl")
    with open(chain_id_path, "w") as f:
        f.write(json.dumps(chain_id_dict) + "\n")

    cmd = [
        "python",
        str(mpnn_run),
        "--pdb_path",
        pdb_path,
        "--out_folder",
        out_folder,
        "--path_to_model_weights",
        os.path.join(model_dir, "vanilla_model_weights") + "/",
        "--pdb_path_chains",
        designed_chain_id,
        "--chain_id_jsonl",
        chain_id_path,
        "--num_seq_per_target",
        str(num_seq),
        "--sampling_temp",
        sampling_temp,
    ]
    if bias_by_res_jsonl and os.path.exists(bias_by_res_jsonl):
        cmd.extend(["--bias_by_res_jsonl", bias_by_res_jsonl])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=model_dir,
    )
    if result.returncode != 0:
        logger.warning(f"ProteinMPNN stderr: {result.stderr}")
        raise RuntimeError(f"ProteinMPNN failed: {result.stderr[:500]}")

    seqs_path = os.path.join(out_folder, "seqs", f"{pdb_name}.fa")
    if not os.path.exists(seqs_path):
        raise FileNotFoundError(f"ProteinMPNN did not produce {seqs_path}")

    sequences = []
    with open(seqs_path) as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq = line.strip().replace("/", "")
            if seq and len(seq) >= 5:
                sequences.append(seq)
    return sequences


def run_mpnn_on_predictions(
    predictions_dir: str,
    sample_name: str,
    bias_by_res_jsonl: str | None = None,
    num_seq_per_cif: int = 1,
) -> str | None:
    """
    Run MPNN on all CIFs in predictions_dir and write summary.csv.
    The binder is the last chain in each CIF.
    Returns path to summary.csv, or None if failed.
    """
    tool_root = os.environ.get("TOOL_WEIGHTS_ROOT", "")
    if not tool_root:
        logger.warning("TOOL_WEIGHTS_ROOT not set; skipping MPNN sequence design")
        return None
    mpnn_weights = os.path.join(tool_root, "mpnn")
    if not os.path.isdir(mpnn_weights):
        logger.warning(f"MPNN weights dir not found at {mpnn_weights}; skipping")
        return None

    cif_files = sorted(Path(predictions_dir).glob("*.cif"))
    if not cif_files:
        logger.warning(f"No CIF files in {predictions_dir}")
        return None

    rows = []
    for cif_path in cif_files:
        with tempfile.TemporaryDirectory() as tmp:
            # Use CIF stem so bias file key can match (bias uses structure name)
            pdb_stem = cif_path.stem
            pdb_path = os.path.join(tmp, f"{pdb_stem}.pdb")
            try:
                _cif_to_pdb(str(cif_path), pdb_path)
            except Exception as e:
                logger.warning(f"CIF to PDB failed for {cif_path}: {e}")
                continue

            from biotite.structure.io import pdbx as pdbx_mod
            atom_array = pdbx_mod.get_structure(
                pdbx_mod.CIFFile.read(str(cif_path)), model=1
            )
            chain_ids = _get_chain_ids(atom_array)
            if not chain_ids:
                continue
            binder_chain = chain_ids[-1]

            # Adapt bias file: map structure name to pdb_stem, "binder" to actual chain
            bias_path = bias_by_res_jsonl
            if bias_by_res_jsonl and os.path.exists(bias_by_res_jsonl):
                try:
                    with open(bias_by_res_jsonl) as f:
                        bias_data = json.load(f)
                    # Get bias array from first struct/chain; support "*" wildcard
                    arr = None
                    for skey, chains in bias_data.items():
                        for ckey, val in chains.items():
                            if isinstance(val, (list, tuple)) or hasattr(val, "tolist"):
                                arr = val
                                break
                        if arr is not None:
                            break
                    if arr is not None:
                        chain_bias = {binder_chain: arr}
                        rewritten = {pdb_stem: chain_bias}
                        bias_path = os.path.join(tmp, "bias.jsonl")
                        with open(bias_path, "w") as f:
                            f.write(json.dumps(rewritten) + "\n")
                except Exception as e:
                    logger.warning(f"Could not adapt bias file: {e}")

            mpnn_out = os.path.join(tmp, "mpnn_out")
            os.makedirs(mpnn_out, exist_ok=True)
            try:
                seqs = _run_proteinmpnn(
                    pdb_path=pdb_path,
                    out_folder=mpnn_out,
                    path_to_model_weights=mpnn_weights,
                    designed_chain_id=binder_chain,
                    all_chain_ids=chain_ids,
                    bias_by_res_jsonl=bias_path,
                    num_seq=num_seq_per_cif,
                )
            except Exception as e:
                logger.warning(f"ProteinMPNN failed for {cif_path.name}: {e}")
                continue
            if seqs:
                rows.append({"sequence": seqs[0]})

    if not rows:
        return None

    # Write to design_outputs/sample_name/summary.csv so CASCADE wrapper finds it
    # predictions_dir = output_dir/sample_name/seed_X/predictions
    output_dir = Path(predictions_dir).parent.parent.parent
    design_out = output_dir / "design_outputs" / sample_name
    design_out.mkdir(parents=True, exist_ok=True)
    summary_path = str(design_out / "summary.csv")
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    logger.info(f"Wrote MPNN summary.csv with {len(rows)} sequences to {summary_path}")
    return summary_path
