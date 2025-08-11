from datasets import (
    load_dataset,
    DatasetDict,
    Dataset,
    get_dataset_config_names,
    concatenate_datasets,
    Value,
)
from typing import Optional, Union, Dict, List
import logging
from omegaconf import DictConfig
import argparse


def standardize_dataset(
    args: Union[dict, "argparse.Namespace"],
    split: Optional[str] = None,
) -> DatasetDict:
    """
    指定の Hugging Face Dataset を config ごとに取得し、列名を標準化。
    全ての config / split で id を「config名-id」に置き換える。
    （config名は None の場合は空文字になる）

    args で使うキー:
      - dataset        (必須) str
      - question_col   (任意) str
      - answer_col     (任意) str
      - thinking_col   (任意) str
      - id_col         (任意) str または None
    """

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

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # ---------- helpers ----------
    def _colmap(ds: Dataset) -> Dict[str, str]:
        return {c.lower(): c for c in ds.column_names}

    def _find_free_name(existing_lower: set, base: str) -> str:
        candidate = base
        idx = 1
        while candidate.lower() in existing_lower:
            candidate = f"{base}{idx}"
            idx += 1
        return candidate

    def _safe_rename(ds: Dataset, src: str, dst: str) -> Dataset:
        if src == dst:
            return ds
        cmap = _colmap(ds)
        if dst.lower() in cmap and cmap[dst.lower()] != src:
            current = cmap[dst.lower()]
            free = _find_free_name(
                set(n.lower() for n in ds.column_names), f"{dst}_orig"
            )
            logger.warning(f"[rename-collision] {dst} 列を退避: {current} -> {free}")
            ds = ds.rename_column(current, free)
        logger.info(f"rename {src} -> {dst}")
        return ds.rename_column(src, dst)

    def _resolve_override(ds: Dataset, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        return _colmap(ds).get(name.lower())

    def _ensure_id_exists(ds: Dataset, id_col_real: Optional[str]) -> Dataset:
        """
        id が無ければ付与。id_col が指定されていて存在するなら 'id' に rename。
        最終的に id は string 型にキャスト。
        """
        cmap = _colmap(ds)
        if "id" in cmap:
            if cmap["id"] != "id":
                ds = _safe_rename(ds, cmap["id"], "id")
        elif id_col_real:
            ds = _safe_rename(ds, id_col_real, "id")
        else:
            # 1始まりの文字列IDを追加
            ds = ds.map(lambda _, idx: {"id": str(idx + 1)}, with_indices=True)

        # 文字列化（既に string なら no-op）
        try:
            ds = ds.cast_column("id", Value("string"))
        except Exception:
            ds = ds.map(lambda ex: {"id": str(ex["id"])})
        return ds

    def _concat_if_same_features(parts: List[Dataset]) -> Dataset:
        if len(parts) == 1:
            return parts[0]
        ref_features = parts[0].features
        for p in parts[1:]:
            assert (
                p.features == ref_features
            ), f"[features mismatch]\nref={ref_features}\ncur={p.features}"
        return concatenate_datasets(parts)

    def _prefix_id(ds: Dataset, tmp_col: str) -> Dataset:
        """全行の id を config名-id に置換（config名が空の場合はそのまま）"""

        def _mapper(ex):
            cfgv = ex.get(tmp_col, "")
            if cfgv:
                return {"id": f"{cfgv}-{ex['id']}"}
            else:
                return {"id": ex["id"]}

        ds = ds.map(_mapper)
        if tmp_col in ds.column_names:
            ds = ds.remove_columns([tmp_col])
        return ds

    # ---------- load ----------
    logger.info(f"[start] dataset={dataset_name} split={split!r}")
    configs = get_dataset_config_names(dataset_name)

    if not configs:
        datasets_by_cfg: Dict[Optional[str], Union[Dataset, DatasetDict]] = {
            None: (
                load_dataset(dataset_name, split=split, trust_remote_code=True)
                if split
                else load_dataset(dataset_name, trust_remote_code=True)
            )
        }
    else:
        datasets_by_cfg = {
            cfg: (
                load_dataset(dataset_name, cfg, split=split, trust_remote_code=True)
                if split
                else load_dataset(dataset_name, cfg, trust_remote_code=True)
            )
            for cfg in configs
        }

    # ---------- preprocess ----------
    prepared_by_cfg: Dict[Optional[str], DatasetDict] = {}
    for cfg, ds_or_dd in datasets_by_cfg.items():
        if isinstance(ds_or_dd, Dataset):
            dd = DatasetDict({(split or "train"): ds_or_dd})
        else:
            dd = ds_or_dd

        std_splits: Dict[str, Dataset] = {}
        for split_name, ds in dd.items():
            if not isinstance(ds, Dataset):
                continue

            q_col_real = _resolve_override(ds, question_col)
            a_col_real = _resolve_override(ds, answer_col)
            t_col_real = _resolve_override(ds, thinking_col)
            id_col_real = _resolve_override(ds, id_col) if id_col is not None else None

            # 列標準化
            if q_col_real and q_col_real != "question":
                ds = _safe_rename(ds, q_col_real, "question")
            if a_col_real and a_col_real != "answer":
                ds = _safe_rename(ds, a_col_real, "answer")
            if t_col_real and t_col_real != "thinking":
                ds = _safe_rename(ds, t_col_real, "thinking")

            # id の存在保証
            ds = _ensure_id_exists(ds, id_col_real)

            # config名列を付与
            cfg_label = str(cfg) if cfg is not None else ""
            tmp_col = "__cfg__"
            if tmp_col.lower() in _colmap(ds):
                tmp_col = _find_free_name(
                    set(n.lower() for n in ds.column_names), "__cfg__"
                )
            ds = ds.map(lambda _: {tmp_col: cfg_label})

            std_splits[split_name] = ds

        prepared_by_cfg[cfg] = DatasetDict(std_splits)

    # ---------- merge ----------
    if split:
        parts = []
        tmp_col_name = None
        for cfg, dd in prepared_by_cfg.items():
            if split in dd:
                parts.append(dd[split])
                if tmp_col_name is None:
                    cands = [
                        c for c in dd[split].column_names if c.startswith("__cfg__")
                    ]
                    tmp_col_name = cands[0] if cands else "__cfg__"

        if not parts:
            raise ValueError(
                f"要求された split='{split}' はいずれのconfigにも存在しません"
            )

        merged = _concat_if_same_features(parts)
        merged = _prefix_id(merged, tmp_col_name)

        logger.info(
            f"[done] return split='{split}' parts={len(parts)} size={len(merged)}"
        )
        return DatasetDict({split: merged})

    all_split_names = set()
    for dd in prepared_by_cfg.values():
        all_split_names.update(dd.keys())
    if not all_split_names:
        raise RuntimeError("結合対象の split が見つかりませんでした")

    out: Dict[str, Dataset] = {}
    for s in sorted(all_split_names):
        parts = []
        tmp_col_name = None
        for cfg, dd in prepared_by_cfg.items():
            if s in dd:
                parts.append(dd[s])
                if tmp_col_name is None:
                    cands = [c for c in dd[s].column_names if c.startswith("__cfg__")]
                    tmp_col_name = cands[0] if cands else "__cfg__"

        merged = _concat_if_same_features(parts)
        merged = _prefix_id(merged, tmp_col_name)

        out[s] = merged
        logger.info(f"[merge] split='{s}' parts={len(parts)} size={len(merged)}")

    logger.info("[done] return all splits with prefixed ids")
    return DatasetDict(out)
