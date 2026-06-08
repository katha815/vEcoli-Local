#!/usr/bin/env python3

from pathlib import Path
import json
import ssl
import urllib.parse
import urllib.request
from urllib.error import URLError
import argparse

import certifi
import pandas as pd


# ---------- SSL-safe request ----------
def safe_urlopen(req, timeout=60, allow_insecure_ssl_fallback=True):
    strict_ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=strict_ctx)
    except URLError as e:
        if not allow_insecure_ssl_fallback:
            raise
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise

        print("Warning: SSL verification failed; using insecure fallback.")
        insecure_ctx = ssl._create_unverified_context()
        return urllib.request.urlopen(req, timeout=timeout, context=insecure_ctx)


# ---------- MyGene query ----------
def query_mygene_by_symbol(symbol, species="511145"):
    base_url = "https://mygene.info/v3/query"
    q = f"symbol:{symbol} AND taxid:{species}"

    params = {
        "q": q,
        "fields": "symbol,name,entrezgene,go.BP.term,go.CC.term,go.MF.term,summary",
        "size": "1",
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, method="GET")

    with safe_urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    hits = data.get("hits", [])
    return hits[0] if hits else {}


def query_mygene_symbols(symbols, species="511145"):
    best = {}
    for sym in symbols:
        hit = query_mygene_by_symbol(sym, species=species)
        if hit:
            best[sym] = hit
    return best


# ---------- GO extraction ----------
def _to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def extract_go_terms(hit):
    go = hit.get("go", {}) if isinstance(hit, dict) else {}
    terms = []

    for namespace in ["BP", "CC", "MF"]:
        block = go.get(namespace)
        for item in _to_list(block):
            if isinstance(item, dict) and item.get("term"):
                terms.append(item["term"])

    return sorted(set(terms))


# ---------- Main pipeline ----------
def run(input_csv, output_path, output_name, max_genes=None, species="511145"):
    input_csv = Path(input_csv)

    df = pd.read_csv(input_csv)
    if "gene_symbol" not in df.columns:
        raise ValueError("Input CSV must contain 'gene_symbol' column")

    symbols = df["gene_symbol"].dropna().astype(str).str.strip()
    symbols = symbols[symbols != ""].unique().tolist()

    if max_genes is not None:
        symbols = symbols[:max_genes]

    print(f"Querying MyGene for {len(symbols)} symbols...")

    hits = query_mygene_symbols(symbols, species=species)

    rows = []
    for sym in symbols:
        hit = hits.get(sym, {})
        go_terms = extract_go_terms(hit)

        rows.append(
            {
                "gene_symbol": sym,
                "mygene_found": bool(hit),
                "mygene_name": hit.get("name"),
                "entrezgene": hit.get("entrezgene"),
                "n_go_terms": len(go_terms),
                "go_terms": "; ".join(go_terms),
            }
        )

    anno_df = pd.DataFrame(rows)
    result_df = df.merge(anno_df, on="gene_symbol", how="left")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    out_file = output_path / output_name
    result_df.to_csv(out_file, index=False)

    print(f"Wrote: {out_file}")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Gene_GO_query.py",
        description=(
            "Query MyGene.info for GO annotations and save gene-level summaries.\n\n"
            "This script:\n"
            "- Reads a CSV containing a 'gene_symbol' column\n"
            "- Queries MyGene.info for GO terms\n"
            "- Extracts BP/CC/MF GO annotations\n"
            "- Writes annotated results to disk"
        ),
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file containing a 'gene_symbol' column",
    )

    parser.add_argument(
        "--out_dir", required=True, help="Directory where output CSV will be saved"
    )

    parser.add_argument(
        "--out_name", required=True, help="Output filename (e.g. annotated_genes.csv)"
    )

    parser.add_argument(
        "--max_genes",
        type=int,
        default=None,
        help="Optional limit for number of genes (useful for testing)",
    )

    parser.add_argument(
        "--species", default="511145", help="NCBI taxid (default: 511145 = E. coli)"
    )

    args = parser.parse_args()

    run(
        input_csv=args.input,
        output_path=args.out_dir,
        output_name=args.out_name,
        max_genes=args.max_genes,
        species=args.species,
    )
