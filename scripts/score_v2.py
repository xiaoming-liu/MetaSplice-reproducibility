#!/usr/bin/env python3
"""
MetaSplice v2.0: pathogenicity of splice-region SNVs outside exons, scored on GRCh38.

Input:  VCF or TSV/CSV of SNVs (chr, pos, ref, alt). GRCh38 by default; --genome hg19 lifts
        GRCh37 input to GRCh38 with the Ensembl chain, and reports variants it cannot lift.
Output: TSV with the score, region and gene of each in-scope variant. --features-out also writes
        the 38-feature matrix; every v2.0 training and evaluation table is built that way, so
        training and scoring share one feature code path (docs/METASPLICE_V2_PLAN.md).

Annotation: GENCODE v50. MANE Select + MANE Plus Clinical transcripts define regions, genes, exon
context and the exon-overlap filter; a variant outside every MANE transcript of its gene is out of
scope (decision O1). A Deep Intronic position inside an exon of any GENCODE basic transcript is out
of scope too (decision O5); variants in MANE splice regions stay in scope whatever overlaps them.
Pass the full GENCODE v50 GTF: the MANE subset lacks the basic transcripts rule O5 needs, and is refused.

Changes from v1 (the 53-feature model, scorer v1.0.x, GRCh37):
  - 38 features: no allele frequency (reported, not used), RVIS, Eigen, GERP++, ESE/ESS or SMS.
  - Everything is computed on GRCh38 except DANN, which exists only for GRCh37 and is looked up
    by lifting each position back, with the orientation checked.
  - Gene-level features (dbNSFP gene file) are looked up by Ensembl gene ID; a missing LOEUF is
    missing (NaN), not 0.0.
  - SpliceTransformer and Pangolin scores are matched to variants by chr:pos:ref:alt, not by row.
  - Nothing is skipped silently. Features are computed for in-scope variants only; a source file or
    contig missing for a chromosome an in-scope variant needs raises an error; gnomAD AF 0.0 means
    the lookup ran and found no variant (a variant not looked up gets NaN). Scoring needs every
    source of the 38 features; --features-only writes features without a model or scores.
"""
import argparse
import bisect
import csv
import functools
import gzip
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
VERSION = "2.0.0"

TISSUES = ["Adipose Tissue", "Blood", "Blood Vessel", "Brain", "Colon", "Heart", "Kidney", "Liver",
           "Lung", "Muscle", "Nerve", "Small Intestine", "Skin", "Spleen", "Stomach"]
REGION_ONEHOT = ["Acceptor Region", "Branch Point Region", "Canonical Acceptor Site", "Canonical Donor Site",
                 "Deep Intronic", "Donor Region", "PPT Region"]
CONSERVATION = ["phyloP100way_vertebrate", "phyloP470way_mammalian", "phastCons100way_vertebrate",
                "phastCons470way_mammalian"]
GENE_LEVEL = ["gnomAD_pLI", "gnomAD_LOEUF", "ClinGen_Haploinsufficiency_Score"]
EXON_CONTEXT = ["exon_length", "exon_length_mod3", "dist_to_boundary"]
FEATURES = (CONSERVATION + GENE_LEVEL + ["DANN_score"] + EXON_CONTEXT + ["spt_orig"]
            + [f"spt_o_{t}" for t in TISSUES] + ["pangolin_inc", "pangolin_dec", "pangolin_max", "gpn_msa_score"]
            + [f"r_{r}" for r in REGION_ONEHOT])       # model order: the 38 features of the G1 clinical model
assert len(FEATURES) == 38

COMP = str.maketrans("ACGT", "TGCA")
BASES = {"A", "C", "G", "T"}
MANE_TAGS = ('tag "MANE_Select"', 'tag "MANE_Plus_Clinical"')
BASIC_TAG = 'tag "basic"'

PRIORITY = {
    "Canonical Acceptor Site": 1, "Canonical Donor Site": 2,
    "Acceptor Region": 3, "Donor Region": 4,
    "PPT Region": 7, "Branch Point Region": 8, "Deep Intronic": 9,
}
TRAIN_PRIORITY = {
    "Canonical Acceptor Site": 1, "Canonical Donor Site": 2,
    "Acceptor Region": 3, "Donor Region": 4,
    "Exonic Acceptor Region": 5, "Exonic Donor Region": 6,
    "PPT Region": 7, "Branch Point Region": 8,
    "Deep Intronic": 9, "Exonic": 10,
}
EXONIC_LABELS = {"Exonic", "Exonic Acceptor Region", "Exonic Donor Region"}


def _attr(field, key):
    m = re.search(key + r' "([^"]+)"', field)
    return m.group(1) if m else None


# ============================================================
# Annotation and region classification (the training rule of v1.0.4, on MANE transcripts)
# ============================================================
def parse_gene_exons(gtf_path, mane_only=False):
    """gene_exons[chrom][gene_name] = [(strand, start, end), ...] in GTF order. With mane_only, only
    exons of MANE Select and MANE Plus Clinical transcripts; otherwise every exon (V02 uses that for
    the canonical-in-any-transcript flag of the deep-intronic sensitivity result)."""
    gene_exons = defaultdict(lambda: defaultdict(list))
    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            x = line.rstrip("\n").split("\t")
            if len(x) < 9 or x[2] != "exon" or (mane_only and not any(t in x[8] for t in MANE_TAGS)):
                continue
            gene = _attr(x[8], "gene_name")
            if gene:
                gene_exons[x[0]][gene].append((x[6], int(x[3]), int(x[4])))
    return gene_exons


def _merge(intervals):
    """Sorted, merged (starts, ends) numpy arrays."""
    iv = sorted(intervals)
    starts, ends = [], []
    for s_, e_ in iv:
        if starts and s_ <= ends[-1] + 1:
            ends[-1] = max(ends[-1], e_)
        else:
            starts.append(s_)
            ends.append(e_)
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


@functools.lru_cache(maxsize=4)
def parse_mane_gtf(gtf_path):
    """MANE Select and MANE Plus Clinical transcripts of a GENCODE GTF, and its basic exons, in one pass:
      gene_exons   {chrom: {gene_name: [(strand, start, end)]}}, GTF order (region classification)
      exons_by_chr {chrom: [(start, end, strand, gene_name)]}           (exon context)
      gene_id      {(chrom, gene_name): gene_id}
      spans        {gene_id: [(chrom, start, end)]}, one per MANE transcript (scope rule O1)
      basic_exons  {chrom: (starts, ends)}, merged exons of every basic transcript (scope rule O5)
      basic_non_mane  number of basic transcripts that are not MANE (0 for the MANE subset)
    Raises ValueError for a GTF without MANE tags: v2.0 is defined on MANE transcripts only."""
    gene_exons = defaultdict(lambda: defaultdict(list))
    exons_by_chr, gene_id, spans = defaultdict(list), {}, defaultdict(list)
    basic, basic_non_mane = defaultdict(list), 0
    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            x = line.rstrip("\n").split("\t")
            if len(x) < 9 or x[2] not in ("exon", "transcript"):
                continue
            mane = any(t in x[8] for t in MANE_TAGS)
            if BASIC_TAG in x[8]:
                if x[2] == "exon":
                    basic[x[0]].append((int(x[3]), int(x[4])))
                elif not mane:
                    basic_non_mane += 1
            if not mane:
                continue
            name, gid = _attr(x[8], "gene_name"), _attr(x[8], "gene_id")
            if not name or not gid:
                continue
            s, e = int(x[3]), int(x[4])
            if x[2] == "transcript":
                spans[gid].append((x[0], s, e))
            else:
                gene_exons[x[0]][name].append((x[6], s, e))
                exons_by_chr[x[0]].append((s, e, x[6], name))
                if gene_id.setdefault((x[0], name), gid) != gid:
                    raise ValueError(f"{name} on {x[0]} has two gene IDs in {gtf_path}")
    if not gene_exons:
        raise ValueError(f"{gtf_path} has no exons of MANE_Select or MANE_Plus_Clinical transcripts; "
                         "MetaSplice v2.0 needs the full GENCODE v50 GTF")
    return {"gene_exons": gene_exons, "exons_by_chr": exons_by_chr, "gene_id": gene_id, "spans": spans,
            "basic_exons": {c: _merge(v) for c, v in basic.items()}, "basic_non_mane": basic_non_mane}


def classify_position(pos, exons_for_gene):
    """Most splice-proximal region of pos relative to the given exons (training rule)."""
    best_r, best_p = None, 999
    for strand, s, e in exons_for_gene:
        if s <= pos <= e:
            from_start, from_end = (pos - s, e - pos) if strand == "+" else (e - pos, pos - s)
            r = ("Exonic Donor Region" if from_end <= 2 else
                 "Exonic Acceptor Region" if from_start <= 2 else "Exonic")
        elif (strand == "+" and pos < s) or (strand == "-" and pos > e):
            d = s - pos if strand == "+" else pos - e
            r = ("Canonical Acceptor Site" if d <= 2 else "Acceptor Region" if d <= 10 else
                 "PPT Region" if d <= 25 else "Branch Point Region" if d <= 44 else "Deep Intronic")
        else:
            d = pos - e if strand == "+" else s - pos
            r = "Canonical Donor Site" if d <= 2 else "Donor Region" if d <= 8 else "Deep Intronic"
        p = TRAIN_PRIORITY[r]
        if p < best_p:
            best_p, best_r = p, r
    return best_r


def build_training_index(gene_exons_chr):
    """Boundary and containment indices for one chromosome, ordered as in training."""
    boundaries, exons_by_start = [], []
    for gene, exons in gene_exons_chr.items():
        for strand, s, e in exons:
            boundaries.append((s, s, e, strand, gene))
            boundaries.append((e, s, e, strand, gene))
            exons_by_start.append((s, e, strand, gene))
    boundaries.sort(key=lambda x: x[0])        # stable: ties keep GTF order, as in training
    exons_by_start.sort(key=lambda x: x[0])
    max_ends, running = [], 0
    for s, e, _, _ in exons_by_start:
        running = max(running, e)
        max_ends.append(running)
    return (boundaries, [b[0] for b in boundaries], exons_by_start,
            [x[0] for x in exons_by_start], max_ends)


def classify_variant_training(pos, index):
    """(region, gene) by the training rule: boundary search, then containment search."""
    boundaries, bnd_positions, exons_by_start, starts, max_ends = index
    best = {"r": None, "p": 999, "g": None, "d": float("inf")}
    seen = set()

    def try_exon(s, e, strand, gene):
        key = (s, e, strand, gene)
        if key in seen:
            return
        seen.add(key)
        r = classify_position(pos, [(strand, s, e)])
        if r is None:
            return
        d, p = min(abs(pos - s), abs(pos - e)), TRAIN_PRIORITY[r]
        if p < best["p"] or (p == best["p"] and d < best["d"]):
            best.update(r=r, p=p, g=gene, d=d)

    idx = bisect.bisect_left(bnd_positions, pos)
    for direction in (-1, 1):
        i = idx if direction == 1 else idx - 1
        while 0 <= i < len(boundaries):
            bp, s, e, strand, gene = boundaries[i]
            if abs(pos - bp) > max(best["d"], 44) and best["r"] is not None:
                break
            try_exon(s, e, strand, gene)
            i += direction
    j = bisect.bisect_right(starts, pos)
    if j > 0 and max_ends[j - 1] >= pos:
        for k in range(j - 1, -1, -1):
            if max_ends[k] < pos:
                break
            s, e, strand, gene = exons_by_start[k]
            if e >= pos:
                try_exon(s, e, strand, gene)
    return best["r"], best["g"]


def build_merged_exons(gene_exons):
    """Merged exon intervals per chromosome, for the exon-overlap filter."""
    merged = {}
    for chrom, genes in gene_exons.items():
        iv = sorted((s, e) for exons in genes.values() for _, s, e in exons)
        starts, ends = [], []
        for s, e in iv:
            if starts and s <= ends[-1] + 1:
                ends[-1] = max(ends[-1], e)
            else:
                starts.append(s)
                ends.append(e)
        merged[chrom] = (starts, ends)
    return merged


def assign_regions(variants, gtf_path):
    """Regions and genes by the training rule on MANE transcripts.

    Returns ({i: region} for the seven in-scope regions, {i: (gene_name, gene_id)} for every
    classified variant). A variant overlapping a MANE exon is Exonic and out of scope, whatever
    splice-region label another MANE transcript would give it. Entries of `variants` that are None
    (not lifted) are skipped. The span rule (O1) is applied by in_mane_transcript / scope."""
    g = parse_mane_gtf(gtf_path)
    index = {c: build_training_index(g["gene_exons"][c]) for c in g["gene_exons"]}
    merged = build_merged_exons(g["gene_exons"])
    regions, genes = {}, {}
    for i, v in enumerate(variants):
        if v is None or v[0] not in index:
            continue
        chrom, pos = v[0], v[1]
        r, name = classify_variant_training(pos, index[chrom])
        if r is not None and r not in EXONIC_LABELS:
            starts, ends = merged[chrom]
            k = bisect.bisect_right(starts, pos) - 1
            if k >= 0 and pos <= ends[k]:
                r = "Exonic"
        if r in PRIORITY:
            regions[i] = r
        if name:
            genes[i] = (name, g["gene_id"][(chrom, name)])
    return regions, genes


def in_mane_transcript(variants, genes, gtf_path):
    """{i: True when the variant lies within the span of a MANE transcript of its assigned gene}."""
    spans = parse_mane_gtf(gtf_path)["spans"]
    return {i: any(c == variants[i][0] and s <= variants[i][1] <= e for c, s, e in spans.get(gid, []))
            for i, (_, gid) in genes.items()}


def in_basic_exon(variants, idx, gtf_path):
    """{i: True when the position lies inside an exon of any GENCODE basic transcript}, for i in idx."""
    basic = parse_mane_gtf(gtf_path)["basic_exons"]
    out = {}
    for i in idx:
        c, p = variants[i][0], variants[i][1]
        if c in basic:
            starts, ends = basic[c]
            k = bisect.bisect_right(starts, p) - 1
            out[i] = bool(k >= 0 and p <= ends[k])
        else:
            out[i] = False
    return out


def scope(variants, gtf_path):
    """({i: region} in scope, {i: (gene_name, gene_id)}, {i: reason} for classified variants out of scope).

    In scope: one of the seven regions, inside a MANE transcript of the assigned gene (O1), and, for
    Deep Intronic, not inside an exon of any GENCODE basic transcript (O5). Needs the full GENCODE GTF:
    a GTF with no basic transcript outside MANE (the MANE subset) would skip rule O5 without notice."""
    g = parse_mane_gtf(gtf_path)
    if not g["basic_non_mane"]:
        raise ValueError(f"{gtf_path} has no basic transcript outside MANE; scope rule O5 needs the full "
                         "GENCODE v50 GTF, not its MANE subset")
    regions, genes = assign_regions(variants, gtf_path)
    inside = in_mane_transcript(variants, genes, gtf_path)
    basic = in_basic_exon(variants, [i for i, r in regions.items() if r == "Deep Intronic"], gtf_path)
    in_scope, why = {}, {}
    for i, v in enumerate(variants):
        if v is None:
            continue
        if i in regions and inside.get(i) and not basic.get(i):
            in_scope[i] = regions[i]
        elif i in regions and not inside.get(i):
            why[i] = "outside MANE transcripts"
        elif i in regions:
            why[i] = "inside a GENCODE basic exon"
        elif i in genes:
            why[i] = "exonic"
        else:
            why[i] = "no MANE transcript on this chromosome"
    return in_scope, genes, why


# ============================================================
# Liftover (Ensembl chain tables; both directions)
# ============================================================
def load_one_to_one(chrom, table_dir, positions):
    """{source position: (target chrom, target position)} for `positions`, from a per-position chain
    table (00_build_liftover_tables.py: columns chrom, pos, target chrom, target pos; no strand).
    Source positions with more than one target, and targets reached from more than one of the loaded
    positions, are dropped (S11a_gnomad_reference_sample.hg38_to_hg19). The builder asserts that the
    chain blocks do not overlap, so the tables are one-to-one throughout."""
    files = [f for f in os.listdir(table_dir) if f.startswith(f"chr{chrom}.") and f.endswith(".gz")]
    if not files or not positions:
        return {}
    need, hits = set(positions), []
    for chunk in pd.read_csv(os.path.join(table_dir, files[0]), sep="\t", header=0, names=["c1", "p1", "c2", "p2"],
                             dtype={"c1": str, "p1": np.int64, "c2": str, "p2": np.int64}, chunksize=10_000_000):
        hits.append(chunk[chunk.p1.isin(need)])
    m = pd.concat(hits)
    m = m[~m.p1.duplicated(keep=False)]
    m = m[~m.duplicated(["c2", "p2"], keep=False)]
    return dict(zip(m.p1.tolist(), zip(m.c2.tolist(), m.p2.tolist())))


def lift_variants(variants, table_dir, target_fasta, source_fasta=None):
    """Lift (chrom, pos, ref, alt) to the other build with per-position chain tables.

    With source_fasta, REF must first equal the source build's base; user input (--genome hg19) is
    checked this way, so that an input REF that is wrong for GRCh37 is not lifted merely because it
    matches GRCh38 (release 2.0.0). Only one-to-one mappings are used. Orientation comes from the
    neighbouring positions: p+1 -> q+1 is forward, p+1 -> q-1 inverted. Forward mappings need REF
    equal to the target base; inverted ones need its complement, and REF and ALT are then
    complemented. Anything else is not lifted.
    Returns ({i: (chrom, pos, ref, alt, strand)}, {i: reason}); None entries are skipped."""
    fa = pysam.FastaFile(target_fasta)
    refs = set(fa.references)
    src = pysam.FastaFile(source_fasta) if source_fasta else None
    src_refs = set(src.references) if src else set()
    by_chr = defaultdict(list)
    for i, v in enumerate(variants):
        if v is not None:
            by_chr[v[0].replace("chr", "")].append(i)
    lifted, failed = {}, {}
    for c, idx in by_chr.items():
        mp = load_one_to_one(c, table_dir, {variants[i][1] + d for i in idx for d in (-1, 0, 1)})
        for i in idx:
            c0, p, r, a = variants[i]
            if src is not None:
                if c0 not in src_refs:
                    failed[i] = f"{c0} not in {os.path.basename(source_fasta)}"
                    continue
                if src.fetch(c0, p - 1, p).upper() != r:
                    failed[i] = f"REF does not match {os.path.basename(source_fasta)}"
                    continue
            if p not in mp:
                failed[i] = "no one-to-one position in the other build"
                continue
            tc, q = mp[p]
            votes = {(mp[p + d][1] - q) * d for d in (1, -1)
                     if p + d in mp and mp[p + d][0] == tc and abs(mp[p + d][1] - q) == 1}
            if len(votes) != 1:
                failed[i] = "orientation not settled by neighbouring positions"
                continue
            strand = votes.pop()
            tc = tc if tc.startswith("chr") else f"chr{tc}"
            if tc not in refs:
                failed[i] = f"{tc} not in {os.path.basename(target_fasta)}"
                continue
            base = fa.fetch(tc, q - 1, q).upper()
            r2, a2 = (r, a) if strand == 1 else (r.translate(COMP), a.translate(COMP))
            if base != r2:
                failed[i] = "REF differs between builds"
                continue
            lifted[i] = (tc, q, r2, a2, strand)
    fa.close()
    if src is not None:
        src.close()
    return lifted, failed


# ============================================================
# Features
# ============================================================
def scan_dann(v19, dann_dir):
    """DANN scores {i: score} for {i: (chrom, pos, ref, alt)} in GRCh37 plus-strand alleles, by one
    sorted scan per chromosome file (the DANN files are position-sorted)."""
    scores = {}
    by_chr = defaultdict(list)
    for i, (chrom, pos, ref, alt) in v19.items():
        by_chr[chrom.replace("chr", "")].append((i, pos, ref, alt))
    for chrom_num, vlist in by_chr.items():
        files = [f for f in os.listdir(dann_dir) if f"chr{chrom_num}." in f and f.endswith(".gz")]
        if not files:
            raise FileNotFoundError(f"no DANN file for chr{chrom_num} in {dann_dir} ({len(vlist)} variants need it)")
        lookup = defaultdict(list)                  # a variant may be listed more than once
        for i, pos, ref, alt in vlist:
            lookup[(pos, ref, alt)].append(i)
        lo, hi = min(p for _, p, _, _ in vlist), max(p for _, p, _, _ in vlist)
        with gzip.open(os.path.join(dann_dir, files[0]), "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 5:
                    continue
                pos = int(parts[1])
                if pos < lo:
                    continue
                if pos > hi:
                    break
                for i in lookup.pop((pos, parts[2], parts[3]), []):
                    val = parts[4].strip()
                    if val != ".":
                        scores[i] = float(val)
                if not lookup:
                    break
    return scores


def compute_dann(variants, dann_dir, rev_table_dir, hg19_fasta):
    """DANN for GRCh38 variants: lift each position to GRCh37 (orientation checked), then look up the
    GRCh37 plus-strand alleles. Returns ({i: score}, {i: reason not lifted})."""
    lifted, failed = lift_variants(variants, rev_table_dir, hg19_fasta)
    return scan_dann({i: v[:4] for i, v in lifted.items()}, dann_dir), failed


def compute_conservation(variants, hg38_map, phylop_dir, phastcons_dir):
    """phyloP and phastCons from GRCh38 wigFix files (v1.0.4)."""
    results = {k: {} for k in CONSERVATION}
    hg38_by_chr = defaultdict(set)
    for i, (hc, hp) in hg38_map.items():
        hg38_by_chr[f"chr{hc}"].add(hp)
    for feat_name, feat_dir, pattern in [
        ("phyloP100way_vertebrate", phylop_dir, "phyloP100way"),
        ("phyloP470way_mammalian", phylop_dir, "phyloP470way"),
        ("phastCons100way_vertebrate", phastcons_dir, "phastCons100way"),
        ("phastCons470way_mammalian", phastcons_dir, "phastCons470way"),
    ]:
        if not feat_dir:
            continue
        for chrom, positions in hg38_by_chr.items():
            path = os.path.join(feat_dir, f"{chrom}.{pattern}.wigFix.gz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found ({len(positions)} positions need {feat_name} on {chrom})")
            max_pos = max(positions)
            scores, current_pos = {}, 0
            with gzip.open(path, "rt") as f:
                for line in f:
                    if line.startswith("fixedStep"):
                        for part in line.strip().split():
                            if part.startswith("start="):
                                current_pos = int(part.split("=")[1])
                        continue
                    if current_pos > max_pos + 1000:
                        break
                    if current_pos in positions:
                        try:
                            scores[current_pos] = float(line.strip())
                        except ValueError:
                            pass
                    current_pos += 1
            for i, (hc, hp) in hg38_map.items():
                if f"chr{hc}" == chrom and hp in scores:
                    results[feat_name][i] = scores[hp]
    return results


def compute_gnomad(variants, hg38_map, gnomad_dir):
    """gnomAD v4.1 joint AF and POPMAX AF, found variants only (v1.0.4). An annotation in v2.0."""
    af_scores, popmax_scores = {}, {}
    if not gnomad_dir:
        return af_scores, popmax_scores
    hg38_by_chr = defaultdict(list)
    for i, (hc, hp) in hg38_map.items():
        hg38_by_chr[hc].append((i, hp, variants[i][2], variants[i][3]))
    for chrom_num, vlist in hg38_by_chr.items():
        path = os.path.join(gnomad_dir, f"gnomad.joint.v4.1.sites.chr{chrom_num}.vcf.bgz.snp.tsv.gz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found ({len(vlist)} variants need gnomAD on chr{chrom_num}); "
                                    "a missing file must not read as allele frequency 0")
        lookup = defaultdict(list)
        for idx, pos, ref, alt in vlist:
            lookup[(pos, ref, alt)].append(idx)
        lo, hi = min(p for _, p, _, _ in vlist), max(p for _, p, _, _ in vlist)
        with gzip.open(path, "rt") as f:
            header = f.readline().strip().split("\t")
            af_idx, pm_idx = header.index("gnomAD_joint_AF"), header.index("gnomAD_joint_POPMAX_AF")
            for line in f:
                parts = line.split("\t")
                try:
                    pos = int(parts[1])
                except ValueError:
                    continue
                if pos < lo:
                    continue
                if pos > hi:
                    break
                for idx in lookup.pop((pos, parts[2], parts[3]), []):
                    try:
                        af_scores[idx] = float(parts[af_idx])
                    except ValueError:
                        pass
                    try:
                        popmax_scores[idx] = float(parts[pm_idx])
                    except ValueError:
                        pass
                if not lookup:
                    break
    return af_scores, popmax_scores


def gnomad_values(n, af_found, popmax_found, looked_up=None):
    """gnomAD AF and POPMAX AF for every variant. A looked-up variant that was not found gets 0.0;
    a variant that was not looked up (looked_up given and i not in it) gets NaN."""
    looked_up = range(n) if looked_up is None else looked_up
    val = lambda d, i: d.get(i, 0.0) if i in looked_up else np.nan
    return [val(af_found, i) for i in range(n)], [val(popmax_found, i) for i in range(n)]


def compute_gpn_msa(variants, hg38_map, gpn_msa_file):
    """GPN-MSA scores via tabix (v1.0.4)."""
    scores = {}
    if not hg38_map:
        return scores
    if not os.path.exists(gpn_msa_file):
        raise FileNotFoundError(f"{gpn_msa_file} not found")
    tbx = pysam.TabixFile(gpn_msa_file)
    absent = sorted({str(hc) for hc, _ in hg38_map.values()} - set(tbx.contigs))
    if absent:
        raise ValueError(f"GPN-MSA ({gpn_msa_file}) has no contig {absent}, which in-scope variants need")
    for i, (hc, hp) in hg38_map.items():
        ref, alt = variants[i][2], variants[i][3]
        for row in tbx.fetch(str(hc), hp - 1, hp):
            parts = row.split("\t")
            if len(parts) >= 5 and parts[2] == ref and parts[3] == alt:
                scores[i] = float(parts[4])
                break
    tbx.close()
    return scores


def compute_gene_level(genes, dbnsfp_path):
    """pLI, LOEUF and ClinGen HI from the dbNSFP gene file, looked up by Ensembl gene ID (decision O2).

    `genes` is {i: (gene_name, gene_id)}; the GENCODE version suffix is dropped. A row listing several
    IDs serves each of them; when an ID appears in more than one row the last wins, as v1 kept the last
    row per symbol. '.' is NaN for every feature, LOEUF included. An ID absent from the file gets no
    entry, so all three features stay NaN."""
    rows = {}
    with gzip.open(dbnsfp_path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        cols = {c: header.index(c) for c in GENE_LEVEL}
        ens = header.index("Ensembl_gene")
        for line in f:
            fs = line.rstrip("\n").split("\t")
            vals = {}
            for c, k in cols.items():
                v = fs[k].split(";")[0]
                try:
                    vals[c] = float(v) if v not in (".", "") else np.nan
                except ValueError:
                    vals[c] = np.nan
            for eid in fs[ens].split(";"):
                if eid not in (".", ""):
                    rows[eid] = vals
    return {i: dict(rows[gid.split(".")[0]]) for i, (_, gid) in genes.items()
            if gid and gid.split(".")[0] in rows}


def find_nearest_exon(pos, boundaries):
    """Nearest exon boundary among one gene's sorted boundaries (training rule,
    02_compute_features.find_nearest_exon): scan left first, then right; at equal
    distance the first exon found wins. Returns ((start, end, strand, gene), distance)."""
    if not boundaries:
        return None, float("inf")
    bnd_positions = [b[0] for b in boundaries]
    idx = bisect.bisect_left(bnd_positions, pos)
    best, best_dist, seen = None, float("inf"), set()
    for direction in (-1, 1):
        i = idx if direction == 1 else idx - 1
        while 0 <= i < len(boundaries):
            if abs(pos - boundaries[i][0]) > best_dist and best is not None:
                break
            _, es, ee, st, g = boundaries[i]
            key = (es, ee, st, g)
            if key not in seen:
                seen.add(key)
                d = min(abs(pos - es), abs(pos - ee))
                if d < best_dist:
                    best_dist, best = d, key
            i += direction
    return best, best_dist


def compute_exon_context(variants, gene_assignments, exons_by_chr):
    """Exon context by the training rule: the assigned gene's exons only (here its MANE exons),
    nearest boundary, no distance limit. gene_assignments is {i: gene_name}."""
    bnd = defaultdict(lambda: defaultdict(list))
    for c, exons in exons_by_chr.items():
        for s, e, st, g in exons:
            bnd[c][g].append((s, s, e, st, g))
            bnd[c][g].append((e, s, e, st, g))
    for c in bnd:
        for g in bnd[c]:
            bnd[c][g].sort()
    results = {}
    for i, v in enumerate(variants):
        gene = gene_assignments.get(i)
        if v is None or not gene:
            continue
        boundaries = bnd.get(v[0], {}).get(gene, [])
        if not boundaries:
            continue
        exon, dist = find_nearest_exon(v[1], boundaries)
        if exon:
            length = exon[1] - exon[0] + 1
            results[i] = {"dist_to_boundary": float(dist), "exon_length": float(length),
                          "exon_length_mod3": float(length % 3)}
    return results


def _key(chrom, pos, ref, alt):
    return (str(chrom).replace("chr", ""), int(pos), str(ref), str(alt))


def load_spt(path):
    """SpliceTransformer output (sptransformer.py --raw_score True): {chr:pos:ref:alt: (score, 15 tissues)}."""
    with open(path) as fh:
        first = fh.readline()
    sep = "," if first.count(",") > first.count("\t") else "\t"
    spt = pd.read_csv(path, sep=sep)
    chrom = "#CHROM" if "#CHROM" in spt.columns else "CHROM"
    missing = [t for t in TISSUES if t not in spt.columns]
    if missing:
        raise ValueError(f"{path} lacks SpliceTransformer tissue columns {missing}; run with --raw_score True")
    vals = spt[["score"] + TISSUES].astype(float).values
    return {_key(c, p, r, a): v for c, p, r, a, v in zip(spt[chrom], spt.POS, spt.REF, spt.ALT, vals)}


def load_pangolin(path):
    """Pangolin output: {chr:pos:ref:alt: (inc, dec)}; blank output is 0.0, as in training."""
    out = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split(",", 4)
            if len(parts) < 4:
                continue
            inc, dec = 0.0, 0.0
            if len(parts) == 5:
                for token in parts[4].replace(",", "|").split("|"):
                    if ":" in token:
                        try:
                            s = float(token.rsplit(":", 1)[-1])
                        except ValueError:
                            continue
                        inc, dec = max(inc, s), min(dec, s)
            out[_key(*parts[:4])] = (inc, dec)
    return out


SCORING_INPUTS = {   # argument: the features it provides; scoring needs all of them
    "dann_dir": "DANN", "reverse_liftover_dir": "DANN (hg38->hg19 chain)", "hg19_fasta": "DANN (GRCh37 FASTA)",
    "phylop_dir": "phyloP", "phastcons_dir": "phastCons", "gpn_msa": "GPN-MSA", "dbnsfp_gene": "gene level (dbNSFP)",
    "spt_scores": "SpliceTransformer", "pangolin_scores": "Pangolin",
}


def missing_scoring_inputs(args):
    """The feature sources a scoring run lacks. Without any of them the model would score
    variants with those features missing, which it never saw in training."""
    return [f"--{k.replace('_', '-')} ({what})" for k, what in SCORING_INPUTS.items() if not getattr(args, k, "")]


def load_model(model_dir=MODEL_DIR):
    """The v2.0 model and a check that it was trained on FEATURES, in this order."""
    import joblib
    feats = list(joblib.load(os.path.join(model_dir, "metasplice_v2_features.joblib")))
    if feats != FEATURES:
        raise ValueError("model/metasplice_v2_features.joblib does not list the scorer's 38 features in order")
    return joblib.load(os.path.join(model_dir, "metasplice_v2_full.joblib"))


# ============================================================
# Input parsing
# ============================================================
def parse_input(path, skipped=None):
    """SNVs (chrom with "chr", pos, REF, ALT) from a VCF (.vcf, .vcf.gz) or a table (.tsv, else comma-separated).

    Alleles are read case-insensitively, as VCF allows. Records that are not SNVs are skipped; with skipped (a dict),
    they are counted by reason. A VCF line with fewer than five columns, or a position that is not a whole number,
    raises ValueError naming the line (release 2.0.0)."""
    variants = []
    skipped = {} if skipped is None else skipped

    def take(chrom, pos, ref, alt, where):
        try:
            pos = int(pos)
        except (TypeError, ValueError):
            raise ValueError(f"{where}: position {pos!r} is not a whole number") from None
        chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
        ref, alt = ref.upper(), alt.upper()
        if ref in BASES and alt in BASES:
            variants.append((chrom, pos, ref, alt))
            return
        why = ("multi-allelic record" if "," in alt else "indel or other non-SNV" if len(ref) != 1 or len(alt) != 1
               else "REF or ALT not A, C, G or T")
        skipped[why] = skipped.get(why, 0) + 1

    if path.endswith(".vcf") or path.endswith(".vcf.gz"):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            for k, line in enumerate(f, 1):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    raise ValueError(f"{path}, line {k}: fewer than 5 tab-separated columns (CHROM POS ID REF ALT)")
                take(parts[0], parts[1], parts[3], parts[4], f"{path}, line {k}")
    else:
        sep = "\t" if path.endswith(".tsv") else ","
        with open(path) as f:
            for k, row in enumerate(csv.DictReader(f, delimiter=sep), 2):
                take(row.get("chr", row.get("CHROM", row.get("#CHROM", ""))) or "", row.get("pos", row.get("POS")),
                     row.get("ref", row.get("REF", "")) or "", row.get("alt", row.get("ALT", "")) or "",
                     f"{path}, line {k}")
    return variants


# ============================================================
# Main
# ============================================================
def env(name):
    return os.environ.get(name, "")


def main():
    ap = argparse.ArgumentParser(description=f"MetaSplice {VERSION}: splice-region variant pathogenicity (GRCh38)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--genome", default="hg38", choices=["hg38", "hg19"],
                    help="build of the input; hg19 input is lifted to GRCh38 (--liftover-dir)")
    ap.add_argument("--fasta", default=env("METASPLICE_FASTA"), help="GRCh38 FASTA")
    ap.add_argument("--hg19-fasta", default=env("METASPLICE_HG19_FASTA"), help="GRCh37 FASTA (DANN orientation check; the input REF check with --genome hg19)")
    ap.add_argument("--gtf", default=env("METASPLICE_GTF"), help="the full GENCODE v50 GTF (not its MANE subset)")
    ap.add_argument("--liftover-dir", default=env("METASPLICE_LIFTOVER"), help="hg19->hg38 Ensembl chain tables")
    ap.add_argument("--reverse-liftover-dir", default=env("METASPLICE_RLIFTOVER"), help="hg38->hg19 Ensembl chain tables")
    ap.add_argument("--dann-dir", default=env("METASPLICE_DANN"))
    ap.add_argument("--phylop-dir", default=env("METASPLICE_PHYLOP"))
    ap.add_argument("--phastcons-dir", default=env("METASPLICE_PHASTCONS"))
    ap.add_argument("--gnomad-dir", default=env("METASPLICE_GNOMAD"), help="gnomAD v4.1 joint (reported, not a model input)")
    ap.add_argument("--gpn-msa", default=env("METASPLICE_GPNMSA"))
    ap.add_argument("--dbnsfp-gene", default=env("METASPLICE_DBNSFP"), help="dbNSFP 5.4 gene file")
    ap.add_argument("--spt-scores", default=env("METASPLICE_SPT"), help="SpliceTransformer output for the GRCh38 variants")
    ap.add_argument("--pangolin-scores", default=env("METASPLICE_PANGOLIN"), help="Pangolin output for the GRCh38 variants")
    ap.add_argument("--features-out", default="", help="also write the 38-feature matrix (TSV)")
    ap.add_argument("--features-only", action="store_true",
                    help="write --features-out without scoring (no model needed; missing sources leave features missing)")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--include-out-of-scope", action="store_true", help="also report variants out of scope, with the reason")
    args = ap.parse_args()
    for name in ("fasta", "gtf"):
        if not getattr(args, name):
            ap.error(f"--{name} is required")
    if args.features_only:
        if not args.features_out:
            ap.error("--features-only needs --features-out")
    else:
        missing = missing_scoring_inputs(args)
        if missing:
            ap.error("scoring needs every source of the 38 features; missing: " + ", ".join(missing)
                     + ". To write features without scores, use --features-only --features-out.")
        for f in ("metasplice_v2_full.joblib", "metasplice_v2_features.joblib"):
            if not os.path.exists(os.path.join(args.model_dir, f)):
                ap.error(f"{os.path.join(args.model_dir, f)} not found; to write features without a model, "
                         "use --features-only --features-out")

    skipped = {}
    raw = parse_input(args.input, skipped)
    n = len(raw)
    print(f"MetaSplice {VERSION}: {n:,} SNVs ({args.genome} input)")
    if skipped:
        print(f"  skipped {sum(skipped.values()):,} records that are not SNVs: "
              + ", ".join(f"{k} {v:,}" for k, v in sorted(skipped.items())))
    status = [""] * n
    if args.genome == "hg19":
        if not args.liftover_dir:
            ap.error("--genome hg19 needs --liftover-dir (hg19->hg38 chain tables)")
        if not args.hg19_fasta:
            ap.error("--genome hg19 needs --hg19-fasta (the input REF is checked against GRCh37)")
        lifted, failed = lift_variants(raw, args.liftover_dir, args.fasta, source_fasta=args.hg19_fasta)
        variants = [lifted[i][:4] if i in lifted else None for i in range(n)]
        for i in range(n):
            status[i] = (f"lifted ({'forward' if lifted[i][4] == 1 else 'inverted'})" if i in lifted
                         else f"not lifted: {failed.get(i, 'no mapping')}")
        print(f"  lifted to GRCh38: {len(lifted):,}/{n:,}")
    else:
        fa = pysam.FastaFile(args.fasta)
        refs = set(fa.references)
        variants = list(raw)
        for i, (c, p, r, _) in enumerate(raw):
            if c not in refs:
                variants[i], status[i] = None, f"{c} not in the GRCh38 FASTA"
            elif fa.fetch(c, p - 1, p).upper() != r:
                variants[i], status[i] = None, "REF does not match GRCh38"
            else:
                status[i] = "GRCh38 input"
        fa.close()

    regions, genes, why = scope(variants, args.gtf)
    for i in range(n):
        if variants[i] is None:
            why[i] = status[i]
    print(f"  in scope: {len(regions):,}/{n:,}")
    # Features are computed for in-scope variants only; out-of-scope rows carry none.
    work = [variants[i] if i in regions else None for i in range(n)]
    names = {i: genes[i][0] for i in regions}
    in_genes = {i: genes[i] for i in regions}

    feat = [{f: np.nan for f in FEATURES} for _ in range(n)]
    for i in range(n):
        for r in REGION_ONEHOT:
            feat[i][f"r_{r}"] = 1.0 if regions.get(i) == r else 0.0
    for i, ctx in compute_exon_context(work, names, parse_mane_gtf(args.gtf)["exons_by_chr"]).items():
        feat[i].update(ctx)

    hg38_map = {i: (v[0].replace("chr", ""), v[1]) for i, v in enumerate(work) if v is not None}
    if args.dann_dir:
        if not (args.reverse_liftover_dir and args.hg19_fasta):
            ap.error("--dann-dir needs --reverse-liftover-dir and --hg19-fasta")
        print("DANN (GRCh37, via the hg38->hg19 chain)...")
        dann, dfail = compute_dann(work, args.dann_dir, args.reverse_liftover_dir, args.hg19_fasta)
        for i, v in dann.items():
            feat[i]["DANN_score"] = v
        print(f"  not lifted to GRCh37: {sum(1 for i in dfail if i in regions):,} in-scope variants (DANN missing)")
    if args.phylop_dir or args.phastcons_dir:
        print("Conservation...")
        for name, sc in compute_conservation(work, hg38_map, args.phylop_dir, args.phastcons_dir).items():
            for i, v in sc.items():
                feat[i][name] = v
    af_all = popmax_all = [np.nan] * n
    if args.gnomad_dir:
        print("gnomAD v4.1 (annotation)...")
        af_all, popmax_all = gnomad_values(n, *compute_gnomad(work, hg38_map, args.gnomad_dir), looked_up=set(regions))
    if args.gpn_msa:
        print("GPN-MSA...")
        for i, v in compute_gpn_msa(work, hg38_map, args.gpn_msa).items():
            feat[i]["gpn_msa_score"] = v
    if args.dbnsfp_gene:
        print("Gene level (by Ensembl gene ID)...")
        for i, v in compute_gene_level(in_genes, args.dbnsfp_gene).items():
            feat[i].update(v)
    if args.spt_scores:
        spt = load_spt(args.spt_scores)
        miss = [i for i in regions if _key(*variants[i]) not in spt]
        if miss:
            raise ValueError(f"{len(miss):,} in-scope variants have no SpliceTransformer row, e.g. {variants[miss[0]]}")
        for i, v in enumerate(work):
            if v is not None and _key(*v) in spt:
                vals = spt[_key(*v)]
                feat[i]["spt_orig"] = float(vals[0])
                for t, x in zip(TISSUES, vals[1:]):
                    feat[i][f"spt_o_{t}"] = float(x)
    if args.pangolin_scores:
        pg = load_pangolin(args.pangolin_scores)
        miss = [i for i in regions if _key(*variants[i]) not in pg]
        if miss:
            raise ValueError(f"{len(miss):,} in-scope variants have no Pangolin row, e.g. {variants[miss[0]]}")
        for i, v in enumerate(work):
            if v is not None and _key(*v) in pg:
                inc, dec = pg[_key(*v)]
                feat[i].update(pangolin_inc=inc, pangolin_dec=dec, pangolin_max=max(inc, abs(dec)))

    X = np.array([[feat[i][f] for f in FEATURES] for i in range(n)], dtype=np.float32).reshape(n, len(FEATURES))
    keep = [i for i in range(n) if args.include_out_of_scope or i in regions]
    cov = {f: np.isfinite(X[list(regions), k]).mean() if regions else 0.0 for k, f in enumerate(FEATURES)}
    print(f"  features present for >50% of in-scope variants: {sum(v > 0.5 for v in cov.values())}/{len(FEATURES)}")

    columns = (["chr", "pos", "ref", "alt"] + (["chr38", "pos38", "ref38", "alt38"] if args.genome == "hg19" else [])
               + ["region", "gene", "gene_id", "gnomad_af", "gnomad_popmax_af", "status"]
               + (["in_scope", "out_of_scope_reason"] if args.include_out_of_scope else []))

    def base_cols(i):
        v = variants[i]
        row = {"chr": raw[i][0], "pos": raw[i][1], "ref": raw[i][2], "alt": raw[i][3]}
        if args.genome == "hg19":
            row.update(chr38=v[0] if v else "", pos38=v[1] if v else "", ref38=v[2] if v else "", alt38=v[3] if v else "")
        row.update(region=regions.get(i, ""), gene=genes.get(i, ("", ""))[0], gene_id=genes.get(i, ("", ""))[1],
                   gnomad_af=af_all[i], gnomad_popmax_af=popmax_all[i], status=status[i])
        if args.include_out_of_scope:
            row.update(in_scope=i in regions, out_of_scope_reason=why.get(i, ""))
        return row

    if args.features_out:
        fo = pd.DataFrame([base_cols(i) for i in keep], columns=columns)
        fo = pd.concat([fo, pd.DataFrame(X[keep], columns=FEATURES)], axis=1)
        fo.to_csv(args.features_out, sep="\t", index=False)
        print(f"  wrote {args.features_out}")
    out = pd.DataFrame([base_cols(i) for i in keep], columns=columns)
    if not args.features_only:
        pred = load_model(args.model_dir).predict_proba(X)[:, 1] if n else np.zeros(0)
        # full precision, so that the exact PP3/BP4 thresholds give the same assignments as the model's scores
        out["splice_score"] = [float(pred[i]) if i in regions else np.nan for i in keep]
    out.to_csv(args.output, sep="\t", index=False)
    print(f"Saved: {args.output} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
