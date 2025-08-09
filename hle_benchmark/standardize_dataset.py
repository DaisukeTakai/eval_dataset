"""
HuggingFace Datasets 標準化モジュール（args指定版）

args に以下のキー/属性を持たせて渡す:
    dataset        (必須) str
    question_col   (任意) str
    answer_col     (任意) str
    thinking_col   (任意) str
    id_col         (任意) str または None

戻り値:
    dict[Optional[str], DatasetDict]  # config名 -> DatasetDict
"""

from __future__ import annotations
from typing import Optional, Dict, Union
from datasets import load_dataset, DatasetDict, Dataset, get_dataset_config_names
import logging


def standardize_dataset(args: Union[dict, "argparse.Namespace"]) -> Dict[Optional[str], DatasetDict]:
    # Namespace → dict に統一
    if not isinstance(args, dict):
        args = vars(args)

    dataset_name = args.get("dataset")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("'dataset' が未指定です")

    question_col = args.get("question_col")
    answer_col = args.get("answer_col")
    thinking_col = args.get("thinking_col")
    id_col = args.get("id_col")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logger = logging.getLogger(__name__)

    def _safe_rename(ds: Dataset, src: str, dst: str) -> Dataset:
        if src == dst:
            return ds
        col_map = {c.lower(): c for c in ds.column_names}
        if dst.lower() in col_map and col_map[dst.lower()] != src:
            current = col_map[dst.lower()]
            base = f"{dst}_orig"
            candidate = base
            idx = 1
            while candidate.lower() in col_map:
                candidate = f"{base}{idx}"
                idx += 1
            logger.warning(f"[rename-collision] {dst} 列を退避: {current} -> {candidate}")
            ds = ds.rename_column(current, candidate)
        logger.info(f"rename {src} -> {dst}")
        return ds.rename_column(src, dst)

    def _add_sequential_id(ds: Dataset) -> Dataset:
        return ds.map(lambda ex, idx: {"id": str(idx + 1)}, with_indices=True)

    logger.info(f"[start] dataset={dataset_name}")
    configs = get_dataset_config_names(dataset_name)
    if not configs:
        datasets_by_cfg = {None: load_dataset(dataset_name)}
    else:
        datasets_by_cfg = {cfg: load_dataset(dataset_name, cfg) for cfg in configs}

    results: Dict[Optional[str], DatasetDict] = {}
    for cfg, ds_or_dd in datasets_by_cfg.items():
        if isinstance(ds_or_dd, Dataset):
            dd = DatasetDict({"train": ds_or_dd})
        else:
            dd = ds_or_dd

        std_splits = {}
        for split, ds in dd.items():
            if not isinstance(ds, Dataset):
                continue

            def resolve_override(name: Optional[str]) -> Optional[str]:
                if not name:
                    return None
                col_map_local = {c.lower(): c for c in ds.column_names}
                return col_map_local.get(name.lower())

            q_col_real = resolve_override(question_col)
            a_col_real = resolve_override(answer_col)
            t_col_real = resolve_override(thinking_col)
            id_col_real = resolve_override(id_col) if id_col is not None else None

            if "id" not in {c.lower() for c in ds.column_names}:
                if id_col is not None and id_col_real and id_col_real != "id":
                    ds = ds.rename_column(id_col_real, "id")
                else:
                    ds = _add_sequential_id(ds)

            if q_col_real and q_col_real != "question":
                ds = _safe_rename(ds, q_col_real, "question")
            if a_col_real and a_col_real != "answer":
                ds = _safe_rename(ds, a_col_real, "answer")
            if t_col_real and t_col_real != "thinking":
                ds = _safe_rename(ds, t_col_real, "thinking")

            std_splits[split] = ds

        results[cfg] = DatasetDict(std_splits)

    logger.info("[done]")
    return results


if __name__ == "__main__":
    import yaml
    with open("conf/config.yaml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f) or {}

    datasets = standardize_dataset(args)
    print(datasets)
