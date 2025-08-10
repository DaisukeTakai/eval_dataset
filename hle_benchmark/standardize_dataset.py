from datasets import load_dataset, DatasetDict, Dataset, get_dataset_config_names, concatenate_datasets
from typing import Optional, Union
import logging
from omegaconf import DictConfig
import argparse


def standardize_dataset(
    args: Union[dict, "argparse.Namespace"],
    split: Optional[str] = None,
) -> DatasetDict:
    # Namespace / DictConfig → dict に統一
    if isinstance(args, DictConfig):
        args = dict(args)
    elif not isinstance(args, dict):
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
        # 1始まりの文字列ID
        return ds.map(lambda ex, idx: {"id": str(idx + 1)}, with_indices=True)

    def _concat_if_same_features(parts: list[Dataset]) -> Dataset:
        """featuresが全て一致している場合のみconcatenate"""
        if len(parts) == 1:
            return parts[0]
        ref_features = parts[0].features
        for p in parts[1:]:
            assert p.features == ref_features, (
                f"[features mismatch]\nref={ref_features}\ncur={p.features}"
            )
        return concatenate_datasets(parts)

    logger.info(f"[start] dataset={dataset_name} split={split!r}")
    configs = get_dataset_config_names(dataset_name)

    # configごとにロード
    if not configs:
        datasets_by_cfg = {
            None: load_dataset(dataset_name, split=split) if split else load_dataset(dataset_name)
        }
    else:
        datasets_by_cfg = {
            cfg: load_dataset(dataset_name, cfg, split=split) if split else load_dataset(dataset_name, cfg)
            for cfg in configs
        }

    # 前処理（列の標準化）を config × split で実施
    prepared_by_cfg: dict[Optional[str], DatasetDict] = {}
    for cfg, ds_or_dd in datasets_by_cfg.items():
        if isinstance(ds_or_dd, Dataset):
            dd = DatasetDict({(split or "train"): ds_or_dd})
        else:
            dd = ds_or_dd

        std_splits = {}
        for split_name, ds in dd.items():
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

            # id 列の整備（なければ付与／別名ならrename）
            lower_names = {c.lower() for c in ds.column_names}
            if "id" not in lower_names:
                if id_col is not None and id_col_real and id_col_real != "id":
                    ds = ds.rename_column(id_col_real, "id")
                else:
                    ds = _add_sequential_id(ds)

            # 他の列の標準化
            if q_col_real and q_col_real != "question":
                ds = _safe_rename(ds, q_col_real, "question")
            if a_col_real and a_col_real != "answer":
                ds = _safe_rename(ds, a_col_real, "answer")
            if t_col_real and t_col_real != "thinking":
                ds = _safe_rename(ds, t_col_real, "thinking")

            std_splits[split_name] = ds

        prepared_by_cfg[cfg] = DatasetDict(std_splits)

    # —— 結合ロジック ——
    if split:
        # 指定splitのみ、全configをconcat
        parts = [dd[split] for dd in prepared_by_cfg.values() if split in dd]
        if not parts:
            raise ValueError(f"要求された split='{split}' はいずれのconfigにも存在しません")
        merged = _concat_if_same_features(parts)
        logger.info(f"[done] return merged split='{split}' ({len(parts)} parts)")
        return DatasetDict({split: merged})

    # split未指定：存在する全split名を集約し、各splitごとに全configをconcat
    all_split_names = set()
    for dd in prepared_by_cfg.values():
        all_split_names.update(dd.keys())
    if not all_split_names:
        raise RuntimeError("結合対象の split が見つかりませんでした")

    out = {}
    for s in sorted(all_split_names):
        parts = [dd[s] for dd in prepared_by_cfg.values() if s in dd]
        merged = _concat_if_same_features(parts)
        out[s] = merged
        logger.info(f"[merge] split='{s}' parts={len(parts)}")

    logger.info("[done] return merged all splits")
    return DatasetDict(out)
