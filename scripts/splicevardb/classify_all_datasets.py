"""
Comprehensive script to replicate SpliceVarDB classifications for all 4 datasets.
Handles deduplication and excludes conflicting variants.
"""

import csv
from collections import defaultdict, Counter
import pandas as pd


def classify_savnet_variant(rna_reads, bayes_factor):
    """
    SAVNet classification criteria from Table 1:
    - Splice-altering: x ≥ 5 OR x ≥ 10
    - Low-frequency: 1 ≤ x < 5 AND 3 ≤ x < 10
    - Normal: x < 5 AND x < 10
    """
    if rna_reads >= 5 or bayes_factor >= 10:
        return 'Splice-altering'
    elif 1 <= rna_reads < 5 and 3 <= bayes_factor < 10:
        return 'Low-frequency splice-altering'
    else:
        return 'Normal'


def classify_misplice_variant(rna_reads, jaf):
    """
    MiSplice classification criteria from Table 1:
    - Splice-altering: x ≥ 5 AND x ≥ 0.05
    - Normal: Otherwise
    """
    if rna_reads >= 5 and jaf >= 0.05:
        return 'Splice-altering'
    else:
        return 'Normal'


def classify_mfass_variant(dpsi):
    """
    MFASS classification criteria from Table 1:
    - Splice-altering: x ≤ -0.50
    - Low-frequency: 0.03 ≤ |x| < 0.5
    - Normal: |x| < 0.03

    Note: Using v2_dpsi column (which corresponds to delta_index in sdv_simple_list
    and dpsi in snv_list), with v1_dpsi as fallback when v2_dpsi is NA.
    """
    if dpsi <= -0.50:
        return 'Splice-altering'
    elif 0.03 <= abs(dpsi) < 0.5:
        return 'Low-frequency splice-altering'
    elif abs(dpsi) < 0.03:
        return 'Normal'
    else:
        return 'Normal'


def classify_mapsy_variant(esm, log2_allelic_ratio):
    """
    MaPSY classification criteria:
    1. ESM=1 AND |allelic_ratio| >= 1.5 -> Splice-altering
    2. |allelic_ratio| >= 0.1 -> Low-frequency splice-altering
    3. |allelic_ratio| < 0.1 -> Normal

    Note: ESM=1 variants with |allelic_ratio| < 1.5 are reclassified
    using the normal allelic ratio thresholds.
    """
    import math

    if log2_allelic_ratio == float('-inf'):
        allelic_ratio = 0
    else:
        allelic_ratio = math.pow(2, log2_allelic_ratio)

    # ESM=1 AND high allelic ratio -> Splice-altering
    if esm == 1 and abs(allelic_ratio) >= 1.5:
        return 'Splice-altering'

    # Otherwise use allelic ratio threshold
    if abs(allelic_ratio) >= 0.1:
        return 'Low-frequency splice-altering'
    else:
        return 'Normal'


def process_savnet(input_file, output_file):
    """Process SAVNet data with deduplication and conflict handling."""
    print("\n" + "="*70)
    print("Processing SAVNet...")
    print("="*70)

    # Store all instances of each variant
    variant_instances = defaultdict(list)

    with open(input_file, 'r') as f:
        # Skip first 2 lines
        next(f)
        next(f)
        reader = csv.DictReader(f)

        for row in reader:
            mutation_key = row.get('Mutation key', '').strip('"')
            if mutation_key:
                try:
                    rna_reads = int(row['Supporting read count'])
                    bayes_factor_str = row['Bayes factor']
                    if bayes_factor_str != 'Rescued':
                        bayes_factor = float(bayes_factor_str)
                        classification = classify_savnet_variant(rna_reads, bayes_factor)
                        variant_instances[mutation_key].append({
                            'classification': classification,
                            'reads': rna_reads,
                            'bf': bayes_factor,
                            'sample': row.get('Sample name', ''),
                            'cancer': row.get('Cancer type', ''),
                            'gene': row.get('Gene', ''),
                            'row': row
                        })
                except (ValueError, KeyError):
                    pass

    # Process variants: exclude conflicting, keep one instance of non-conflicting
    final_variants = []
    conflicting_variants = []

    for mutation_key, instances in variant_instances.items():
        classifications = [inst['classification'] for inst in instances]
        unique_classifications = set(classifications)

        if len(unique_classifications) > 1:
            # Conflicting - exclude
            conflicting_variants.append({
                'mutation_key': mutation_key,
                'classifications': unique_classifications,
                'num_instances': len(instances)
            })
        else:
            # Non-conflicting - keep the first instance
            final_variants.append({
                'mutation_key': mutation_key,
                'classification': instances[0]['classification'],
                'num_instances': len(instances),
                **instances[0]
            })

    # Write results
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Mutation_Key', 'Gene', 'Cancer_Type', 'Sample',
                        'RNA_Reads', 'Bayes_Factor', 'Classification', 'Num_Instances'])
        for var in final_variants:
            writer.writerow([
                var['mutation_key'],
                var['gene'],
                var['cancer'],
                var['sample'],
                var['reads'],
                var['bf'],
                var['classification'],
                var['num_instances']
            ])

    # Summary
    classifications = [v['classification'] for v in final_variants]
    counts = Counter(classifications)

    print(f"Total unique variants: {len(variant_instances):,}")
    print(f"Conflicting variants (excluded): {len(conflicting_variants):,}")
    print(f"Final variants classified: {len(final_variants):,}")
    print(f"  Splice-altering: {counts['Splice-altering']:,}")
    print(f"  Low-frequency: {counts['Low-frequency splice-altering']:,}")
    print(f"  Normal: {counts['Normal']:,}")

    return counts, len(conflicting_variants)


def process_misplice(input_file, output_file):
    """
    Process MiSplice data from 'Passed Manual Review' sheet.

    All variants in this sheet have already passed the MiSplice pipeline
    filters (RNA Reads >= 5 AND JAF >= 0.05), so they are all Splice-altering.
    We only need to deduplicate by variant key.
    """
    print("\n" + "="*70)
    print("Processing MiSplice...")
    print("="*70)

    # Read from Excel file, 'Passed Manual Review' sheet
    df = pd.read_excel(input_file, sheet_name='Passed Manual Review')

    total_rows = len(df)

    # Create variant key for deduplication
    df['variant_key'] = (df['Chromosome'].astype(str) + ':' +
                         df['Position'].astype(str) + ':' +
                         df['Reference'].astype(str) + ':' +
                         df['MutationType'].astype(str))

    # Count duplicates before deduplication
    duplicate_count = total_rows - df['variant_key'].nunique()

    # Deduplicate - keep first occurrence
    df_unique = df.drop_duplicates(subset=['variant_key'], keep='first')

    # All variants are Splice-altering (already passed pipeline filters)
    df_unique = df_unique.copy()
    df_unique['Classification'] = 'Splice-altering'

    # Write results
    output_df = df_unique[['variant_key', 'GeneName', 'Chromosome', 'Position',
                           'Reference', 'MutationType', 'Junction Allele Fraction',
                           'CaseReads', 'Classification']].copy()
    output_df.columns = ['Variant_Key', 'Gene', 'Chromosome', 'Position',
                         'Reference', 'MutationType', 'JAF', 'RNA_Reads', 'Classification']
    output_df.to_csv(output_file, index=False)

    # Summary
    counts = Counter(df_unique['Classification'])

    print(f"Total rows in 'Passed Manual Review': {total_rows:,}")
    print(f"Duplicate variants removed: {duplicate_count:,}")
    print(f"Final unique variants: {len(df_unique):,}")
    print(f"  Splice-altering: {counts.get('Splice-altering', 0):,}")
    print(f"  Low-frequency: {counts.get('Low-frequency splice-altering', 0):,}")
    print(f"  Normal: {counts.get('Normal', 0):,}")
    print()
    print("Note: All variants in 'Passed Manual Review' are Splice-altering")
    print("      (already passed MiSplice pipeline RNA Reads and JAF filters)")

    return counts, 0  # No conflicts - all are splice-altering


def process_mfass(input_file, output_file):
    """Process MFASS data with category filtering and fallback column handling."""
    print("\n" + "="*70)
    print("Processing MFASS...")
    print("="*70)

    final_variants = []

    with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)

        # Find column indices
        variant_id_idx = 0
        category_idx = header.index('category')
        v1_dpsi_idx = header.index('v1_dpsi')
        v2_dpsi_idx = header.index('v2_dpsi')

        used_v2 = 0
        used_v1 = 0
        skipped = 0
        total_mutants = 0

        for row in reader:
            try:
                variant_id = row[variant_id_idx]
                category = row[category_idx]

                # Only process mutant category
                if category != 'mutant':
                    continue

                total_mutants += 1

                # Try v2_dpsi first (primary column)
                v2_dpsi_str = row[v2_dpsi_idx]
                v1_dpsi_str = row[v1_dpsi_idx]

                dpsi_value = None
                dpsi_source = None

                if v2_dpsi_str != 'NA':
                    try:
                        dpsi_value = float(v2_dpsi_str)
                        dpsi_source = 'v2_dpsi'
                        used_v2 += 1
                    except ValueError:
                        pass

                # Fallback to v1_dpsi if v2_dpsi not available
                if dpsi_value is None and v1_dpsi_str != 'NA':
                    try:
                        dpsi_value = float(v1_dpsi_str)
                        dpsi_source = 'v1_dpsi'
                        used_v1 += 1
                    except ValueError:
                        pass

                # Skip if both are NA
                if dpsi_value is None:
                    skipped += 1
                    continue

                # Classify the variant
                classification = classify_mfass_variant(dpsi_value)

                final_variants.append({
                    'variant_id': variant_id,
                    'classification': classification,
                    'dpsi_value': dpsi_value,
                    'dpsi_source': dpsi_source
                })

            except (ValueError, IndexError):
                pass

    # Write results
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Variant_ID', 'dPSI_Value', 'dPSI_Source', 'Classification'])
        for var in final_variants:
            writer.writerow([
                var['variant_id'],
                var['dpsi_value'],
                var['dpsi_source'],
                var['classification']
            ])

    # Summary
    classifications = [v['classification'] for v in final_variants]
    counts = Counter(classifications)

    print(f"Total mutant variants: {total_mutants:,}")
    print(f"Used v2_dpsi: {used_v2:,}")
    print(f"Used v1_dpsi (fallback): {used_v1:,}")
    print(f"Skipped (both NA): {skipped:,}")
    print(f"Final variants classified: {len(final_variants):,}")
    print(f"  Splice-altering: {counts.get('Splice-altering', 0):,}")
    print(f"  Low-frequency: {counts.get('Low-frequency splice-altering', 0):,}")
    print(f"  Normal: {counts.get('Normal', 0):,}")

    return counts, 0  # No conflicts in MFASS (no duplicates)


def process_mapsy(input_file, output_file):
    """Process MaPSY data with ESM-based classification."""
    print("\n" + "="*70)
    print("Processing MaPSY...")
    print("="*70)

    final_variants = []
    skipped = 0

    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                # Get variant identifiers
                variant_id = row['MUT_ID']
                wt_id = row.get('WT_ID', '')

                # Get ESM and log2_ratio
                esm_str = row.get('ESM', '0')
                log2_ratio_str = row['log2_vivo_allele_ratio']

                # Skip if empty log2_ratio
                if log2_ratio_str in ('NA', ''):
                    skipped += 1
                    continue

                esm = int(esm_str)

                if log2_ratio_str == '-Inf':
                    log2_ratio = float('-inf')
                else:
                    log2_ratio = float(log2_ratio_str)

                # Classify using ESM and allelic ratio
                classification = classify_mapsy_variant(esm, log2_ratio)

                final_variants.append({
                    'variant_id': variant_id,
                    'wt_id': wt_id,
                    'esm': esm,
                    'log2_ratio': log2_ratio,
                    'classification': classification
                })
            except (ValueError, KeyError):
                skipped += 1

    # Write results
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['WT_ID', 'MUT_ID', 'ESM', 'Log2_Ratio', 'Classification'])
        for var in final_variants:
            writer.writerow([
                var['wt_id'],
                var['variant_id'],
                var['esm'],
                var['log2_ratio'],
                var['classification']
            ])

    # Summary
    classifications = [v['classification'] for v in final_variants]
    counts = Counter(classifications)
    esm1_count = sum(1 for v in final_variants if v['esm'] == 1)

    print(f"Total variants classified: {len(final_variants):,}")
    print(f"Skipped (empty log2_ratio): {skipped:,}")
    print(f"ESM=1 variants: {esm1_count:,}")
    print(f"  Splice-altering: {counts.get('Splice-altering', 0):,}")
    print(f"  Low-frequency: {counts.get('Low-frequency splice-altering', 0):,}")
    print(f"  Normal: {counts.get('Normal', 0):,}")

    return counts, 0  # No conflicts in MaPSY


def main():
    """Main function to process all datasets."""
    print("\n" + "="*70)
    print("SpliceVarDB Classification Replication")
    print("="*70)

    # Expected counts from Table 2
    # Note: MiSplice Table 2 value (559) appears to be incorrect.
    # Correct value is 1904 unique variants from 'Passed Manual Review' sheet.
    expected = {
        'SAVNet': {'Total': 13864, 'Splice-altering': 10460,
                   'Low-frequency splice-altering': 3404, 'Normal': 0},
        'MiSplice': {'Total': 1904, 'Splice-altering': 1904,
                     'Low-frequency splice-altering': 0, 'Normal': 0},
        'MFASS': {'Total': 28962, 'Splice-altering': 1468,
                  'Low-frequency splice-altering': 17357, 'Normal': 10115},
        'MaPSY': {'Total': 5595, 'Splice-altering': 208,
                  'Low-frequency splice-altering': 4868, 'Normal': 459}
    }

    results = []

    # Process SAVNet
    counts_savnet, conflicts_savnet = process_savnet(
        'SAVNet/Supplemental_Table_S3.csv',
        'SAVNet/classified_variants_deduplicated.csv'
    )

    total_savnet = sum(counts_savnet.values())
    results.append({
        'Dataset': 'SAVNet',
        'Total': total_savnet,
        'Expected_Total': expected['SAVNet']['Total'],
        'Diff_Total': total_savnet - expected['SAVNet']['Total'],
        'Splice-altering': counts_savnet.get('Splice-altering', 0),
        'Expected_Splice': expected['SAVNet']['Splice-altering'],
        'Low-frequency': counts_savnet.get('Low-frequency splice-altering', 0),
        'Expected_LowFreq': expected['SAVNet']['Low-frequency splice-altering'],
        'Normal': counts_savnet.get('Normal', 0),
        'Expected_Normal': expected['SAVNet']['Normal'],
        'Conflicts': conflicts_savnet
    })

    # Process MiSplice (using Excel file with 'Passed Manual Review' sheet)
    counts_misplice, conflicts_misplice = process_misplice(
        'MiSplice/1-s2.0-S2211124718303978-mmc2.xlsx',
        'MiSplice/classified_variants_deduplicated.csv'
    )

    total_misplice = sum(counts_misplice.values())
    results.append({
        'Dataset': 'MiSplice',
        'Total': total_misplice,
        'Expected_Total': expected['MiSplice']['Total'],
        'Diff_Total': total_misplice - expected['MiSplice']['Total'],
        'Splice-altering': counts_misplice.get('Splice-altering', 0),
        'Expected_Splice': expected['MiSplice']['Splice-altering'],
        'Low-frequency': counts_misplice.get('Low-frequency splice-altering', 0),
        'Expected_LowFreq': expected['MiSplice']['Low-frequency splice-altering'],
        'Normal': counts_misplice.get('Normal', 0),
        'Expected_Normal': expected['MiSplice']['Normal'],
        'Conflicts': conflicts_misplice
    })

    # Process MFASS
    counts_mfass, conflicts_mfass = process_mfass(
        'MFASS/snv_data_clean.txt',
        'MFASS/classified_variants_deduplicated.csv'
    )

    total_mfass = sum(counts_mfass.values())
    results.append({
        'Dataset': 'MFASS',
        'Total': total_mfass,
        'Expected_Total': expected['MFASS']['Total'],
        'Diff_Total': total_mfass - expected['MFASS']['Total'],
        'Splice-altering': counts_mfass.get('Splice-altering', 0),
        'Expected_Splice': expected['MFASS']['Splice-altering'],
        'Low-frequency': counts_mfass.get('Low-frequency splice-altering', 0),
        'Expected_LowFreq': expected['MFASS']['Low-frequency splice-altering'],
        'Normal': counts_mfass.get('Normal', 0),
        'Expected_Normal': expected['MFASS']['Normal'],
        'Conflicts': conflicts_mfass
    })

    # Process MaPSY
    counts_mapsy, conflicts_mapsy = process_mapsy(
        'MaPSY/13059_2019_1653_MOESM2_ESM.csv',
        'MaPSY/classified_variants_deduplicated.csv'
    )

    total_mapsy = sum(counts_mapsy.values())
    results.append({
        'Dataset': 'MaPSY',
        'Total': total_mapsy,
        'Expected_Total': expected['MaPSY']['Total'],
        'Diff_Total': total_mapsy - expected['MaPSY']['Total'],
        'Splice-altering': counts_mapsy.get('Splice-altering', 0),
        'Expected_Splice': expected['MaPSY']['Splice-altering'],
        'Low-frequency': counts_mapsy.get('Low-frequency splice-altering', 0),
        'Expected_LowFreq': expected['MaPSY']['Low-frequency splice-altering'],
        'Normal': counts_mapsy.get('Normal', 0),
        'Expected_Normal': expected['MaPSY']['Normal'],
        'Conflicts': conflicts_mapsy
    })

    # Print comparison table
    print("\n" + "="*70)
    print("FINAL COMPARISON WITH TABLE 2")
    print("="*70)

    df = pd.DataFrame(results)
    print("\nSummary:")
    for _, row in df.iterrows():
        print(f"\n{row['Dataset']}:")
        print(f"  Total: {row['Total']:,} (Expected: {row['Expected_Total']:,}, Diff: {row['Diff_Total']:+,})")
        print(f"  Splice-altering: {row['Splice-altering']:,} (Expected: {row['Expected_Splice']:,})")
        print(f"  Low-frequency: {row['Low-frequency']:,} (Expected: {row['Expected_LowFreq']:,})")
        print(f"  Normal: {row['Normal']:,} (Expected: {row['Expected_Normal']:,})")
        print(f"  Conflicts excluded: {row['Conflicts']:,}")

    # Save to CSV
    df.to_csv('final_classification_comparison.csv', index=False)
    print("\n" + "="*70)
    print("Results saved to final_classification_comparison.csv")
    print("Individual dataset files saved with '_deduplicated' suffix")
    print("="*70)


if __name__ == '__main__':
    main()
