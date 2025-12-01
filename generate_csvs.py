#!/usr/bin/env python3

import argparse
import re
import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

SECURITY_KEYWORDS = [
    "race","racy","buffer","overflow","stack","integer","signedness","underflow",
    "improper","unauthenticated","gain access","permission","cross site","css","xss",
    "denial service","dos","crash","deadlock","injection","request forgery","csrf","xsrf",
    "forged","security","vulnerability","vulnerable","exploit","attack","bypass",
    "backdoor","threat","expose","breach","violate","fatal","blacklist","overrun",
    "insecure"
]

SEC_KW_REGEX = re.compile(
    "(" + "|".join(re.escape(k) for k in SECURITY_KEYWORDS) + ")",
    flags=re.IGNORECASE
)

def clean_diff_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\t", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def export_table_to_csv(dataset, table_name, columns, outpath, batch_size=5000):
    first_write = True
    rows = []
    count = 0
    print(f"Exporting {table_name} -> {outpath}")

    for ex in tqdm(dataset, desc=f"processing {table_name}"):
        out_row = {}
        for out_col, in_field, transform in columns:
            val = ex.get(in_field) if in_field else None
            if transform:
                val = transform(val)
            out_row[out_col] = val
        rows.append(out_row)
        count += 1

        if len(rows) >= batch_size:
            pd.DataFrame(rows).to_csv(
                outpath, mode="a", index=False, header=first_write, encoding="utf-8"
            )
            first_write = False
            rows = []

    if rows:
        pd.DataFrame(rows).to_csv(
            outpath, mode="a", index=False, header=first_write, encoding="utf-8"
        )

    print(f"Finished {table_name}: exported {count} rows")

def main(args):
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    print("Loading AIDev dataset tables...")

    all_pull_request = load_dataset("hao-li/AIDev", "all_pull_request", split="train")
    all_repository = load_dataset("hao-li/AIDev", "all_repository", split="train")
    pr_task_type = load_dataset("hao-li/AIDev", "pr_task_type", split="train")
    pr_commit_details = load_dataset("hao-li/AIDev", "pr_commit_details", split="train")

    # TASK 1
    def t1_transform_body(x):
        if x is None:
            return ""
        return str(x).replace("\n", " ").strip()

    def agent_name(x):
        if isinstance(x, dict):
            return x.get("name", "")
        return x

    t1_path = os.path.join(outdir, "task1_all_pull_request.csv")

    export_table_to_csv(
        all_pull_request,
        "all_pull_request",
        columns=[
            ("TITLE", "title", None),
            ("ID", "id", None),
            ("AGENTNAME", "agent", agent_name),
            ("BODYSTRING", "body", t1_transform_body),
            ("REPOID", "repo_id", None),
            ("REPOURL", "repo_url", None),
        ],
        outpath=t1_path,
        batch_size=args.batch_size
    )

    # TASK 2
    t2_path = os.path.join(outdir, "task2_all_repository.csv")

    export_table_to_csv(
        all_repository,
        "all_repository",
        columns=[
            ("REPOID", "id", None),
            ("LANG", "language", None),
            ("STARS", "stars", None),
            ("REPOURL", "url", None),
        ],
        outpath=t2_path,
        batch_size=args.batch_size
    )

    # TASK 3 
    t3_path = os.path.join(outdir, "task3_pr_task_type.csv")

    export_table_to_csv(
        pr_task_type,
        "pr_task_type",
        columns=[
            ("PRID", "id", None),
            ("PRTITLE", "title", None),
            ("PRREASON", "reason", None),
            ("PRTYPE", "type", None),
            ("CONFIDENCE", "confidence", None),
        ],
        outpath=t3_path,
        batch_size=args.batch_size
    )

    # TASK 4
    t4_path = os.path.join(outdir, "task4_pr_commit_details.csv")

    export_table_to_csv(
        pr_commit_details,
        "pr_commit_details",
        columns=[
            ("PRID", "pr_id", None),
            ("PRSHA", "sha", None),
            ("PRCOMMITMESSAGE", "message", None),
            ("PRFILE", "filename", None),
            ("PRSTATUS", "status", None),
            ("PRADDS", "additions", None),
            ("PRDELSS", "deletions", None),
            ("PRCHANGECOUNT", "changes", None),
            ("PRDIFF", "patch", clean_diff_text),
        ],
        outpath=t4_path,
        batch_size=args.batch_size
    )

    # TASK 5
    print("Building Task-5 merged security CSV...")

    df_pr = pd.read_csv(t1_path, dtype=str)
    df_pr_task = pd.read_csv(t3_path, dtype=str)

    df_pr_task["CONFIDENCE"] = pd.to_numeric(
        df_pr_task["CONFIDENCE"], errors="coerce"
    ).fillna(0.0)

    df_task_best = (
        df_pr_task.sort_values("CONFIDENCE", ascending=False)
        .drop_duplicates(subset=["PRID"], keep="first")
    )

    df_merged = df_pr.merge(
        df_task_best[["PRID", "PRTYPE", "CONFIDENCE"]],
        left_on="ID",
        right_on="PRID",
        how="left"
    )

    def compute_security_flag(row):
        text = ""
        if pd.notna(row.get("TITLE")):
            text += str(row["TITLE"]) + " "
        if pd.notna(row.get("BODYSTRING")):
            text += str(row["BODYSTRING"])
        return 1 if SEC_KW_REGEX.search(text) else 0

    df_merged["SECURITY"] = df_merged.apply(compute_security_flag, axis=1)

    final_df = pd.DataFrame({
        "ID": df_merged["ID"],
        "AGENT": df_merged["AGENTNAME"],
        "TYPE": df_merged["PRTYPE"],
        "CONFIDENCE": df_merged["CONFIDENCE"],
        "SECURITY": df_merged["SECURITY"],
    })

    t5_path = os.path.join(outdir, "task5_combined_security.csv")
    final_df.to_csv(t5_path, index=False, encoding="utf-8")

    print(f"Task-5 written to {t5_path}")
    print("ALL TASKS COMPLETE SUCCESSFULLY")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    main(args)
