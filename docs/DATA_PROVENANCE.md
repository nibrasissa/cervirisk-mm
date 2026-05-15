# Data Provenance

This document is the integrity contract of CerviRisk-MM. It states exactly which data is real, which is sampled, and which is constructed. A reviewer should be able to read this page and know precisely what each prediction is grounded in.

## Provenance categories

| Tag | Meaning |
|---|---|
| `REAL_OBSERVED` | Directly observed in a real source (e.g. UCI age, smoking status). |
| `REAL_OUTCOME` | Real ground-truth outcome from a real biopsy. Never modified after ingestion. |
| `REAL_LOOKUP` | Deterministic lookup in a curated reference table (e.g. PaVE carcinogenicity score). |
| `REAL_COMPUTED` | Computed from real inputs via a published algorithm (e.g. polygenic risk score). |
| `SAMPLED_FROM_PRIOR` | Sampled from a published distribution (e.g. HPV strain prevalence). |
| `CONSTRUCTED` | A combination of unrelated real components into one synthetic record. |

## Per-feature provenance

| Feature | Provenance | Source | Citation |
|---|---|---|---|
| `age` | REAL_OBSERVED | UCI | Fernandes et al. 2017 |
| `smokes`, `smokes_years`, `smokes_packs_year` | REAL_OBSERVED | UCI | Fernandes et al. 2017 |
| `num_pregnancies` | REAL_OBSERVED | UCI | Fernandes et al. 2017 |
| `hormonal_contraceptives_years` | REAL_OBSERVED | UCI | Fernandes et al. 2017 |
| `iud_years` | REAL_OBSERVED | UCI | Fernandes et al. 2017 |
| `stds_*` (history flags) | REAL_OBSERVED | UCI | Fernandes et al. 2017 |
| `hpv_status` (binary) | REAL_OBSERVED | UCI | Fernandes et al. 2017 |
| `biopsy_outcome` | **REAL_OUTCOME** | UCI | Fernandes et al. 2017 |
| `assigned_hpv_strain` | SAMPLED_FROM_PRIOR | Global HPV prevalence | de Sanjosé et al. 2010; IARC monographs |
| `strain_carcinogenicity` | REAL_LOOKUP | PaVE / IARC | IARC Group 1 classification |
| `matched_genome_id` | REAL_OBSERVED | 1000 Genomes phase 3 | 1000 Genomes Project Consortium 2015 |
| `host_prs` | REAL_COMPUTED | 1000 Genomes × PGS Catalog | PGS Catalog methodology |
| `ancestry_match` | REAL_LOOKUP (with assumption) | 1000 Genomes super-population | UCI cohort presumed AMR (Caracas) |

## Integrity rules

1. **Biopsy outcomes are immutable.** Once set from UCI, the `biopsy_outcome` field cannot be overwritten. The `PatientRecord.set_feature()` method raises `ValueError` if called with `biopsy_outcome` as the name. Tested in `tests/test_ingestion_smoke.py::test_provenance_protects_biopsy_outcome`.

2. **Augmentation never overwrites observations.** New augmented features are added as new columns. UCI's original columns are never modified.

3. **Synthetic patients are flagged.** Any record containing a `SAMPLED_FROM_PRIOR` or `CONSTRUCTED` value is flagged `SYNTHETIC_ASSEMBLY`. The API response and any persisted prediction include this flag.

4. **Performance metrics specify their cohort.** Every reported metric (AUROC, AUPRC, calibration) names whether it was computed on UCI-only, UCI-augmented, or synthetic data.

5. **No record-level joins across sources.** There is no shared patient ID across UCI, NCBI, 1000 Genomes, and PGS Catalog because the linkage does not exist in the public data. Patient records are *assembled*, not joined. See `src/features/build_patient.py` (added on day 2) for the construction logic.

## What this pipeline does NOT capture

The augmentations cover the dominant axes of cervical cancer risk that the public data supports, but several known risk modifiers are not represented because the source data does not contain them:

- HPV sublineages within a type (e.g. HPV16 A1 vs D2)
- Multi-strain HPV co-infections
- Infection persistence over time (a major progression driver)
- HPV vaccination history (UCI predates widespread regional vaccination)
- Ancestries outside Admixed American (the UCI cohort is single-site, Caracas)
- Immune status beyond HIV (transplant, biologics, chronic inflammation)
- Environmental and healthcare-access factors
- Time dynamics in screening intervals

The pipeline architecture accepts these features without modification — they would be additional columns in `PatientRecord.features` with their own provenance tags. They are absent in this prototype because the corresponding data sources are not public.

## References

- Fernandes, K., Cardoso, J. S., & Fernandes, J. (2017). Transfer learning with partial observability applied to cervical cancer screening. *Pattern Recognition and Image Analysis*. UCI Machine Learning Repository, dataset ID 383.
- de Sanjosé, S. et al. (2010). Human papillomavirus genotype attribution in invasive cervical cancer: a retrospective cross-sectional worldwide study. *Lancet Oncology* 11(11): 1048–1056.
- IARC Working Group (2012). Biological agents. *IARC Monographs on the Evaluation of Carcinogenic Risks to Humans*, Volume 100B.
- 1000 Genomes Project Consortium (2015). A global reference for human genetic variation. *Nature* 526(7571): 68–74.
- Lambert, S. A. et al. (2021). The Polygenic Score Catalog as an open database for reproducibility and systematic evaluation. *Nature Genetics* 53(4): 420–425.
- Van Doorslaer, K. et al. (2017). The Papillomavirus Episteme: a major update to the papillomavirus sequence database. *Nucleic Acids Research* 45(D1): D499–D506.
