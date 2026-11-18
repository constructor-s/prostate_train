<!-- REFRESH GUIDE: Use the stable current results directories named beside each table/result/figure,
not timestamped before-refresh snapshots. In PI-CAI source notes, {SITE} means PCNN, RUMC, or ZGT. Model keys: nnunet274 = Multicenter;
rumcnotprostatexnoclinical236 = RUMC; clinical = clinical ellipsoid; gt = cohort reference.
For current membership use configs/picai_test_clinical_folds.json (testing_PCNN, testing_RUMC, testing_ZGT) and each analysis's run_metadata.json.
Postprocessing provenance: results/20261105_tile0.8_nomirroring/Dataset236_raw_3d_fullres_5fold/manual_postprocessing.json.
Historical before/after audit only: results/20261105_tile0.8_nomirroring/dataset236_analysis_refresh_20261117T002354663462Z/.
After changes, regenerate upstream analyses before volume_disagreement_quality_improvement.py.
The supplementary DSC cells in picai_test_clinical_volume.py can run on existing volume exports without repeating inference or resampling.
-->

# Methods

## Performance Evaluation

Performance was evaluated per scan, with each scan weighted equally within each cohort. Whole-gland segmentation overlap was measured using the Dice similarity coefficient (DSC). Volume agreement was assessed using mean absolute error (MAE) and root mean squared error (RMSE), in mL, and mean absolute percentage error (MA%E) and root mean squared percentage error (RMS%E). For predicted volume P and reference volume R, these were calculated as mean(|P − R|), √mean[(P − R)²], 100 × mean(|P − R|/R), and 100 × √mean[((P − R)/R)²], respectively. Higher DSC and lower volume errors indicate better performance. DSC was not assessed for scalar volume comparisons.

RUMC PI-CAI training and the three institution-specific PI-CAI testing reference masks were automated surrogate segmentations generated using Bosma22b. RUMC ProstateX references were manual whole-gland segmentations, and Toronto JDMI reference volumes were the median of manual-reader planimetry measurements. The Toronto JDMI cohort comprised 383 scans with finite, positive reference volumes after accession-level exclusions. Toronto JDMI clinical manual ellipsoid volumes used the reported vol_report_ref values, in mL.

Following review of external-cohort segmentation errors, the original 35-scan Multicenter test cohort was separated post hoc into 22 external-coil scans and 13 endorectal-coil (ERC) scans. Both models were evaluated separately on each group. This separation was prompted by an apparent acquisition mismatch: ERC imaging was not known to be represented in the RUMC Model’s training data. These exploratory subgroup results describe observed performance and do not establish that coil type caused the differences. The Multicenter training cohort is reported as mixed, with external-coil and ERC scans evaluated separately in the test cohort.

<!-- Latest data: notebooks/model_study_performance.py -> results/20261105_tile0.8_nomirroring/tables/run_metadata.json; manual_postprocessing.record and sha256. Selection evidence: work_dir_pooled2/nnunet_results/Dataset274_raw/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.json and work_dir_rumcnotprostatexnoclinical236/nnUNet_trained_models/Dataset236_raw/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.json. -->

The Multicenter Model used largest-connected-component filtering for whole-gland label 1, selected during cross-validation. Cross-validation did not select filtering for the RUMC Model. Following review of segmentation failures, the same nnUNet operation was manually imposed on its saved predictions in the Multicenter, RUMC training, ProstateX, and PI-CAI study folders. This was a post hoc processing change rather than a cross-validation-selected operation; all reported results for those folders use the processed masks. The separate Toronto prediction folders did not undergo this additional processing. Original predictions and a per-file processing audit were retained.

Multicenter and RUMC training performance was evaluated using in-sample five-fold ensembles rather than out-of-fold predictions. JDMI Model performance on Toronto JDMI data was also evaluated on training samples. Aggregate metrics were withheld for incomplete evaluations.

## Volume Agreement with Reference Measurements

Volume agreement in RUMC ProstateX was evaluated in the reviewed testing cohort of 157 patients, each contributing one scan with a manual whole-gland segmentation reference and clinical manual ellipsoid volume. Linkage and clinical-volume checks identified 190 otherwise eligible examinations; the reviewed manifest excluded 33 with superior or inferior manual-mask contact at the image boundary, including ProstateX-0031. The retained 157 masks had no outer-face contact in the field-of-view diagnostic. Boundary contact suggests possible incomplete coverage but does not establish truncation. The same reviewed testing manifest defined all ProstateX segmentation and volume comparisons. Clinical consistency review flags did not exclude additional cases. In Toronto JDMI, the analysis included 383 patients, each contributing one scan after accession-level exclusions and requiring finite, positive median manual-reader planimetry reference volumes. Clinical manual ellipsoid estimates, the Multicenter Model, and the RUMC Model were compared on the same complete cohort. No manual reference masks were available for Toronto JDMI, so DSC was not assessed. Signed differences were calculated as measured minus reference volume (P − R), in mL, and relative differences as 100 × (P − R)/R, in percent. Positive differences indicate overestimation. Bias was the mean signed difference, and 95% limits of agreement were calculated as bias ± 1.96 sample standard deviations. Differences were plotted against manual reference volume; limits of agreement are distinct from confidence intervals.

<!-- Latest data: notebooks/_picai_test_clinical.py -> configs/picai_test_clinical_folds.json; results/picai_test_clinical/selection_audit.csv and cohort_flow.csv, assignment=testing_PCNN/testing_RUMC/testing_ZGT. notebooks/picai_test_clinical_volume.py -> results/picai_test_clinical_volume/{SITE}/run_metadata.json: n_patients, training_overlap, prostatex_shared_patients; segmentation_run_metadata.json for DSC membership/geometry. -->

The PI-CAI testing datasets comprised PCNN (315 patients), RUMC (391), and ZGT (168), each contributing one examination. The 874 distinct testing patients were analyzed separately by institution; no pooled performance estimate was calculated. They were selected from the 1,500-examination PI-CAI dataset. The 236 Dataset236 training exams were preserved, and every other exam from those training patients was reserved outside testing. Six manually reviewed poor Bosma22b segmentations were excluded in a separate fold before visit selection. Among remaining patients, any superior or inferior outer-face contact by the Bosma22b reference mask triggered exclusion before visit selection. Clinical completeness required finite positive volume and PSA, valid lesion PI-RADS scores, and biopsy results; recorded PSAD was optional. Biopsy availability required biopsy histopathology, case ISUP 0–5, and valid lesion ISUP or Gleason results. Among remaining visits, complete clinicals were preferred, then the earliest MRI date, with original marksheet row order resolving ties. Incomplete and duplicate visits were retained in separate folds. Testing patients did not overlap either model’s PI-CAI training patients. Clinical review flags did not exclude additional cases. Bosma22b label 1 defined the automated whole-gland surrogate for DSC and volume comparisons. Every retained patient contributed to each model’s mean DSC within their site, including zero-overlap failures, after matching-grid and millimetre-unit checks with maximum voxel-corner displacement ≤0.5 mm. PI-CAI RUMC shared 116 patients with the reviewed ProstateX cohort; PCNN and ZGT shared none. The RUMC and ProstateX analyses therefore were not independent replications.

Paired post hoc comparisons in each cohort assessed RMSE, RMSPE, and the probability of being closer to the reference as separate endpoints. Equality of population RMSE was tested using a two-sided, null-centered bootstrap-t test of the mean paired squared-error difference, with 100,000 patient-level resamples. Equality of population RMSPE was tested using the same procedure applied to paired squared percentage errors, where percentage error was 100 × (P − R)/R. Each cohort used base random seed 20261102 and bootstrap-test seed 20261103. Two-sided exact binomial sign tests assessed whether the probability of a smaller absolute error differed from 0.5, excluding exact ties. Within each cohort, Holm correction jointly covered RMSE and sign tests across all three method pairs (six tests), while RMSPE tests formed a separate Holm family across the three method pairs (three tests). All reported p-values were Holm-adjusted, with significance assessed at p < 0.05. Resampling treated predictions as fixed and patients as independent observations; it did not capture model-training variability or remove limitations of the selected cohorts. These analyses assessed volume accuracy rather than spatial segmentation quality.

## Disagreement-Based Quality Improvement Analysis

Clinical manual ellipsoid volume (M) was compared with the Multicenter Model, nnUNet274 (A), without using reference volumes for screening. Signed disagreement was D = M − A in mL and Q = 100 × (M − A)/[(M + A)/2] in percent. Within each complete volume cohort, separate descriptive 95% limits of agreement (LoA) were estimated as the mean ±1.96 sample standard deviations of D and Q. The primary review flag required disagreement strictly outside both intervals; equality with a limit was unflagged. Individual-scale and union screens were descriptive secondary analyses. The pair-mean denominator for screening differs from the reference-volume denominator used for percentage errors. LoA describe observed disagreement rather than clinical acceptability.

The analysis retained 157 RUMC ProstateX patients with manual whole-gland segmentation references, 315 PI-CAI PCNN, 391 PI-CAI RUMC, and 168 PI-CAI ZGT patients with Bosma22b automated surrogate references, 383 Toronto JDMI patients with median manual-reader planimetry references, and 4,105 expanded Toronto JDMI patients with nnUNet702 automated surrogate references. Bosma22b used whole-gland label 1; nnUNet702 used the union of labels 1, 2, and 3. PI-CAI RUMC and ProstateX shared 116 patients and were analyzed separately; PCNN and ZGT had no ProstateX overlap. PI-CAI thresholds, tests, and correction families were calculated independently within each institution. Current expanded-cohort membership followed 11 accession exclusions from 4,116 examinations; all 383 manual-reference patients were included, so the Toronto analyses were overlapping. Existing notebook-selected PSA values, including Toronto report/alternative-source consolidation and its documented correction, were preserved. No model inference or cohort reselection was performed.

Within each flagged subset, manual ellipsoid and nnUNet274 volumes were evaluated using MAE, RMSE, MAPE, RMSPE, paired absolute-error reductions, and improved/worsened/tied counts. We reported review burden, improvement among all flagged patients, and exact 95% binomial intervals for closer probability among non-ties. Primary-intersection inference retained the null-centered bootstrap-t tests and paired BCa intervals with 100,000 resamples and seeds 20261102/20261103, the exact sign test, and the paired squared-error t-test sensitivity analysis. Degenerate BCa intervals used the existing percentile fallback. Within each cohort, RMSE and sign tests formed a two-test Holm family; RMSPE formed a separate one-comparison family. MAE and MAPE remained descriptive. Thresholds, selected patients, and predictions stayed fixed during resampling; intervals were pointwise and unadjusted and did not account for threshold estimation or selection uncertainty.

Volume flags remained fixed for downstream analyses. PSAD was PSA (ng/mL) divided by volume (mL), in ng/mL², using the same patient PSA for all methods. Unrounded PSAD categories were 0–<0.10, 0.10–<0.15, 0.15–<0.20, and ≥0.20 ng/mL². We counted category changes, corrected and newly incorrect assignments, and changes in absolute ordinal distance from the reference category. Exact McNemar tests compared paired category correctness within each primary subset, as a separate one-comparison family per cohort.

Biopsy recommendations were simulated at unrounded PSAD ≥0.15 ng/mL² among flagged patients with maximum recorded lesion PI-RADS 3 in ProstateX and each PI-CAI site or reported maximum PI-RADS 3 in Toronto. Agreement with the reference-volume decision included all patients with evaluable PSAD, regardless of cancer-outcome availability. Separate descriptive outcome analyses required valid case-level ISUP in ProstateX and each PI-CAI site (positive: grade group ≥2) or recorded Toronto CSCANCER Y/N. Unknown outcomes were excluded only from outcome metrics. Sensitivity used outcome-positive patients, specificity outcome-negative patients, and biopsy avoidance all outcome-evaluable patients, including false negatives. Empty groups and zero-denominator rates were reported explicitly. Full-cohort PSAD and biopsy methods and results are retained in Appendices D–E.

# Results

## Segmentation Performance and Volume Agreement

<!-- Latest data: notebooks/model_study_performance.py -> results/20261105_tile0.8_nomirroring/tables/dsc.csv; rows Dataset/Split, model columns Dataset274_raw_3d_fullres_5fold, Dataset236_raw_3d_fullres_5fold, nnunet-702, Bosma22b, Clinical manual ellipsoid; run_metadata.json. notebooks/picai_test_clinical_volume.py -> results/picai_test_clinical_volume/{SITE}/segmentation_summary.csv: method=nnunet274 or rumcnotprostatexnoclinical236, n and mean_dsc; segmentation_metrics_per_case.csv retains all testing IDs; segmentation_run_metadata.json. -->

**Table 1. Mean whole-gland DSC.**

| Dataset | Multicenter Model | RUMC Model | JDMI Model | Bosma22b | Clinical manual ellipsoid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multicenter Data (n = 239, mixed) | 0.953<br><small><em>(training)</em></small> | — | — | — | — |
| Multicenter Data (n = 22, external coil) | 0.905 | 0.904 | — | — | — |
| Multicenter Data (n = 13, ERC) | 0.911 | 0.593 | — | — | — |
| RUMC PI-CAI training (n = 236) | — | 0.981<br><small><em>(training)</em></small> | — | — | — |
| RUMC ProstateX (n = 157) | 0.906 | 0.916 | — | 0.956<br><small><em>(training)</em></small> | — |
| PI-CAI PCNN (n = 315) | 0.882 | 0.558 | — | — | — |
| PI-CAI RUMC (n = 391) | 0.917 | 0.950 | — | — | — |
| PI-CAI ZGT (n = 168) | 0.929 | 0.953 | — | — | — |
| Toronto JDMI (n = 383) | — | — | — | — | — |

<!-- Latest data: notebooks/model_study_performance.py -> results/20261105_tile0.8_nomirroring/tables/volume_mae_ml.csv and volume_rmse_ml.csv; rows Dataset/Split, model columns Dataset274_raw_3d_fullres_5fold, Dataset236_raw_3d_fullres_5fold, nnunet-702, Bosma22b, Clinical manual ellipsoid; run_metadata.json. notebooks/picai_test_clinical_volume.py -> results/picai_test_clinical_volume/{SITE}/comparator_summary.csv: method=nnunet274/rumcnotprostatexnoclinical236/clinical; columns mae_ml, rmse_ml; run_metadata.json. -->

**Table 2. Whole-gland volume error: MAE; RMSE (mL).**

| Dataset | Multicenter Model | RUMC Model | JDMI Model | Bosma22b | Clinical manual ellipsoid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multicenter Data (n = 239, mixed) | 0.79; 1.21<br><small><em>(training)</em></small> | — | — | — | — |
| Multicenter Data (n = 22, external coil) | 3.42; 4.07 | 2.72; 3.32 | — | — | — |
| Multicenter Data (n = 13, ERC) | 2.43; 3.17 | 6.33; 8.04 | — | — | — |
| RUMC PI-CAI training (n = 236) | — | 0.65; 0.96<br><small><em>(training)</em></small> | — | — | — |
| RUMC ProstateX (n = 157) | 5.09; 6.47 | 4.57; 5.85 | — | 1.12; 2.06<br><small><em>(training)</em></small> | 9.20; 12.06 |
| PI-CAI PCNN (n = 315) | 6.35; 8.48 | 12.68; 20.65 | — | — | 11.69; 16.78 |
| PI-CAI RUMC (n = 391) | 4.25; 5.29 | 2.21; 3.37 | — | — | 11.70; 15.07 |
| PI-CAI ZGT (n = 168) | 2.68; 3.86 | 1.81; 2.43 | — | — | 6.62; 8.92 |
| Toronto JDMI (n = 383) | 4.87; 6.76 | 5.28; 6.61 | 1.65; 2.33<br><small><em>(training)</em></small> | — | 7.18; 9.42 |

<!-- Latest data: notebooks/model_study_performance.py -> results/20261105_tile0.8_nomirroring/tables/volume_mape_pct.csv and volume_rmspe_pct.csv; rows Dataset/Split, model columns Dataset274_raw_3d_fullres_5fold, Dataset236_raw_3d_fullres_5fold, nnunet-702, Bosma22b, Clinical manual ellipsoid; run_metadata.json. notebooks/picai_test_clinical_volume.py -> results/picai_test_clinical_volume/{SITE}/comparator_summary.csv: method=nnunet274/rumcnotprostatexnoclinical236/clinical; columns mean_absolute_percentage_error, rmspe_percent; run_metadata.json. -->

**Table 3. Whole-gland percentage volume error: MAPE; RMSPE.**

| Dataset | Multicenter Model | RUMC Model | JDMI Model | Bosma22b | Clinical manual ellipsoid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multicenter Data (n = 239, mixed) | 1.56%; 2.06%<br><small><em>(training)</em></small> | — | — | — | — |
| Multicenter Data (n = 22, external coil) | 8.52%; 10.13% | 6.47%; 7.87% | — | — | — |
| Multicenter Data (n = 13, ERC) | 6.33%; 8.69% | 15.78%; 19.50% | — | — | — |
| RUMC PI-CAI training (n = 236) | — | 0.95%; 1.23%<br><small><em>(training)</em></small> | — | — | — |
| RUMC ProstateX (n = 157) | 8.85%; 10.93% | 8.10%; 10.47% | — | 2.09%; 4.53%<br><small><em>(training)</em></small> | 16.36%; 21.17% |
| PI-CAI PCNN (n = 315) | 13.09%; 18.82% | 35.83%; 65.90% | — | — | 22.23%; 29.57% |
| PI-CAI RUMC (n = 391) | 9.20%; 11.63% | 4.58%; 6.99% | — | — | 24.82%; 30.82% |
| PI-CAI ZGT (n = 168) | 5.87%; 7.90% | 4.30%; 6.56% | — | — | 14.35%; 19.14% |
| Toronto JDMI (n = 383) | 9.01%; 11.35% | 10.04%; 12.37% | 3.13%; 4.21%<br><small><em>(training)</em></small> | — | 13.89%; 18.07% |

ERC: endorectal coil; DSC: Dice similarity coefficient; MAE: mean absolute error; RMSE: root mean squared error; MAPE: mean absolute percentage error; RMSPE: root mean squared percentage error; training: the dataset was included as part of the model’s training cohort.

The Multicenter and RUMC Models achieved similar mean DSCs of 0.905 and 0.904, respectively, on the 22 held-out external-coil Multicenter test scans, and 0.906 and 0.916 on the 157 held-out RUMC ProstateX scans (Table 1). Similar segmentation overlap did not imply identical volume accuracy: on external-coil scans, MAPE was 8.52% for the Multicenter Model and 6.47% for the RUMC Model.

Both models had lower observed volume errors than clinical manual ellipsoid estimates in RUMC ProstateX and Toronto JDMI (Tables 2 and 3). RMSPEs for the Multicenter and RUMC Models were 10.93% and 10.47% in ProstateX, versus 21.17% for ellipsoid estimates, and 11.35% and 12.37% in Toronto, versus 18.07%. Corresponding RMSEs were 6.47 and 5.85 mL versus 12.06 mL in ProstateX, and 6.76 and 6.61 mL versus 9.42 mL in Toronto. Both models had significantly lower RMSE than ellipsoid estimates in both cohorts (all Holm-adjusted p < 0.001). RMSPE was also lower for both models: adjusted p = 0.004 and 0.003, respectively, in ProstateX, and both p < 0.001 in Toronto. The models were closer to the reference than ellipsoid estimates in 109/157 (69.4%) and 115/157 (73.2%) of ProstateX cases and 64.0% and 56.4% of Toronto cases, respectively (all Holm-adjusted exact sign-test p < 0.03).

In direct model comparisons, the RUMC Model had lower RMSE in ProstateX (5.85 versus 6.47 mL; Holm-adjusted p = 0.037), while the Toronto RMSE difference was not significant (adjusted p = 0.770). The Multicenter Model had lower RMSPE in Toronto (11.35% versus 12.37%; adjusted p = 0.047), but the between-model RMSPE difference in ProstateX was not significant (10.93% versus 10.47%; adjusted p = 0.616). The RUMC Model was closer to the reference in 94/157 ProstateX cases (59.9%; adjusted p = 0.033), while the Multicenter Model was closer in 231/382 non-tied Toronto cases (60.5%; adjusted p < 0.001; one tie excluded).

<!-- Latest data: notebooks/picai_test_clinical_volume.py -> results/picai_test_clinical_volume/{SITE}/comparator_summary.csv (method; bias_ml, relative_bias_percent, mae_ml, rmse_ml, mean_absolute_percentage_error, rmspe_percent); comparator_pairwise_statistics.csv (method_a/method_b; rmse_bootstrap_p_holm, rmspe_bootstrap_p_holm, closer_exact_p_holm, wins_a, losses_a); segmentation_summary.csv (mean_dsc); run_metadata.json and segmentation_run_metadata.json. -->

In PI-CAI PCNN (n = 315), mean DSC against Bosma22b was 0.882 for Multicenter and 0.558 for RUMC, including 14 and 131 zero-overlap cases, respectively (Table 1). Compared with clinical ellipsoid estimates, Multicenter had lower RMSE (8.48 versus 16.78 mL; Holm-adjusted p < 0.001) and lower RMSPE (18.82% versus 29.57%; adjusted p < 0.001). It was closer to the surrogate in 210/315 (66.7%) of cases (adjusted sign-test p < 0.001). Compared with clinical ellipsoid estimates, RUMC had higher RMSE (20.65 versus 16.78 mL; Holm-adjusted p = 0.074) and higher RMSPE (65.90% versus 29.57%; adjusted p < 0.001). It was closer to the surrogate in 177/315 (56.2%) of cases (adjusted sign-test p = 0.074). Between models, Multicenter had lower RMSE and RMSPE (both adjusted p < 0.001); RUMC was closer in 176/315 (55.9%) (adjusted sign-test p = 0.074). These endpoints assess surrogate agreement and can differ in their conclusions.

In PI-CAI RUMC (n = 391), mean DSC against Bosma22b was 0.917 for Multicenter and 0.950 for RUMC, with no zero-overlap cases for either model (Table 1). Compared with clinical ellipsoid estimates, Multicenter had lower RMSE (5.29 versus 15.07 mL; Holm-adjusted p < 0.001) and lower RMSPE (11.63% versus 30.82%; adjusted p < 0.001). It was closer to the surrogate in 319/391 (81.6%) of cases (adjusted sign-test p < 0.001). Compared with clinical ellipsoid estimates, RUMC had lower RMSE (3.37 versus 15.07 mL; Holm-adjusted p < 0.001) and lower RMSPE (6.99% versus 30.82%; adjusted p < 0.001). It was closer to the surrogate in 344/391 (88.0%) of cases (adjusted sign-test p < 0.001). Between models, RUMC had lower RMSE and RMSPE (both adjusted p < 0.001); RUMC was closer in 290/391 (74.2%) (adjusted sign-test p < 0.001). These endpoints assess surrogate agreement and can differ in their conclusions.

In PI-CAI ZGT (n = 168), mean DSC against Bosma22b was 0.929 for Multicenter and 0.953 for RUMC, with no zero-overlap cases for either model (Table 1). Compared with clinical ellipsoid estimates, Multicenter had lower RMSE (3.86 versus 8.92 mL; Holm-adjusted p = 0.005) and lower RMSPE (7.90% versus 19.14%; adjusted p = 0.002). It was closer to the surrogate in 130/168 (77.4%) of cases (adjusted sign-test p < 0.001). Compared with clinical ellipsoid estimates, RUMC had lower RMSE (2.43 versus 8.92 mL; Holm-adjusted p = 0.003) and lower RMSPE (6.56% versus 19.14%; adjusted p = 0.002). It was closer to the surrogate in 134/168 (79.8%) of cases (adjusted sign-test p < 0.001). Between models, the RMSE and RMSPE differences were not significant (adjusted p = 0.136 and p = 0.419, respectively); RUMC was closer in 109/168 (64.9%) (adjusted sign-test p < 0.001). These endpoints assess surrogate agreement and can differ in their conclusions.

Both models underestimated volume on average, as shown in the agreement plots against the manual reference volume (Figures 1–2). Mean relative biases for the Multicenter and RUMC Models were −2.8% and −2.0% in ProstateX (mean signed differences, −2.4 and −1.7 mL), and −5.2% and −8.3% in Toronto (−3.5 and −4.6 mL), respectively. Ellipsoid estimates overestimated volume in ProstateX (+7.0%; +3.3 mL) and underestimated it in Toronto (−2.5%; −1.5 mL). Both models had narrower observed absolute and relative limits of agreement than ellipsoid estimates in both cohorts.

![Whole-gland volume agreement against manual reference volume for 157 RUMC ProstateX patients.](results/prostatex_linked_volume/comparator_agreement.png)

**Figure 1. Whole-gland volume agreement in RUMC ProstateX (n = 157).** Columns show clinical manual ellipsoid estimates ("Manual ellipsoid"), the Multicenter Model ("Multicenter"), and the RUMC Model ("RUMC (same inst.)"). The top row plots measured against manual reference volume by planimetry. The middle and bottom rows show signed differences in mL and relative differences (100% × (measured − reference)/reference) against manual reference volume on a logarithmic x-axis. Histograms at right show counts of differences. Magenta points with black outlines identify the same two patients in every scatter panel, selected by manual ellipsoid–nnUNet274 disagreement outside both the mL and pair-mean percentage LoA. These screening limits are distinct from the method-versus-reference LoA shown as horizontal lines; all plotted statistics and histograms use the full cohort.

![Whole-gland volume agreement against median manual-reader planimetry volume for 383 Toronto JDMI patients.](results/toronto_jdmi_clinical_volume/comparator_agreement.png)

**Figure 2. Whole-gland volume agreement in Toronto JDMI (n = 383).** Columns show clinical manual ellipsoid estimates ("Manual ellipsoid"), the Multicenter Model ("Multicenter"), and the RUMC Model ("RUMC"). The top row plots measured against manual-reader planimetry reference volume. The middle and bottom rows show signed differences in mL and relative differences (100% × (measured − reference)/reference) against reference volume on a logarithmic x-axis. Histograms at right show counts of differences. Red points with black outlines identify the same seven patients in every scatter panel, selected for subsequent analysis by manual ellipsoid–nnUNet274 disagreement strictly outside both the mL and pair-mean percentage LoA (mean ±1.96 sample standard deviations). These screening limits are distinct from the method-versus-reference LoA shown as horizontal lines; all plotted statistics and histograms use the full cohort.

<!-- Latest data: notebooks/picai_test_clinical_volume.py -> results/picai_test_clinical_volume/{SITE}/comparator_agreement.png; comparator_summary.csv (method; bias_ml, relative_bias_percent, loa_lower_ml, loa_upper_ml, relative_loa_lower_percent, relative_loa_upper_percent); manual_automated_disagreement_flags.csv (outside_both); run_metadata.json. -->

In PI-CAI PCNN, mean signed volume differences were +5.57 mL (+12.2%) for Multicenter, +11.39 mL (+34.0%) for RUMC, +3.06 mL (+6.0%) for clinical ellipsoid (Figure 3a).

In PI-CAI RUMC, mean signed volume differences were +2.31 mL (+5.9%) for Multicenter, +0.34 mL (+1.2%) for RUMC, +9.09 mL (+20.2%) for clinical ellipsoid (Figure 3b).

In PI-CAI ZGT, mean signed volume differences were +0.77 mL (+2.9%) for Multicenter, +0.88 mL (+2.7%) for RUMC, -0.84 mL (-0.8%) for clinical ellipsoid (Figure 3c).

![Whole-gland volume agreement against Bosma22b in PI-CAI PCNN (n = 315).](results/picai_test_clinical_volume/PCNN/comparator_agreement.png)

**Figure 3a. Whole-gland volume agreement against Bosma22b in PI-CAI PCNN (n = 315).** Columns show clinical ellipsoid, Multicenter, and RUMC estimates. Rows show estimated versus reference volume, signed differences (mL), and reference-normalized percentage differences, with logarithmic reference axes and difference histograms. Horizontal lines show bias and descriptive 95% limits of agreement. Highlighted cases meet the within-site clinical–Multicenter strict-intersection disagreement screen; all summaries use the full site cohort. Bosma22b agreement is not independent manual-reference accuracy.

![Whole-gland volume agreement against Bosma22b in PI-CAI RUMC (n = 391).](results/picai_test_clinical_volume/RUMC/comparator_agreement.png)

**Figure 3b. Whole-gland volume agreement against Bosma22b in PI-CAI RUMC (n = 391).** Columns show clinical ellipsoid, Multicenter, and RUMC estimates. Rows show estimated versus reference volume, signed differences (mL), and reference-normalized percentage differences, with logarithmic reference axes and difference histograms. Horizontal lines show bias and descriptive 95% limits of agreement. Highlighted cases meet the within-site clinical–Multicenter strict-intersection disagreement screen; all summaries use the full site cohort. Bosma22b agreement is not independent manual-reference accuracy.

![Whole-gland volume agreement against Bosma22b in PI-CAI ZGT (n = 168).](results/picai_test_clinical_volume/ZGT/comparator_agreement.png)

**Figure 3c. Whole-gland volume agreement against Bosma22b in PI-CAI ZGT (n = 168).** Columns show clinical ellipsoid, Multicenter, and RUMC estimates. Rows show estimated versus reference volume, signed differences (mL), and reference-normalized percentage differences, with logarithmic reference axes and difference histograms. Horizontal lines show bias and descriptive 95% limits of agreement. Highlighted cases meet the within-site clinical–Multicenter strict-intersection disagreement screen; all summaries use the full site cohort. Bosma22b agreement is not independent manual-reference accuracy.

## Quality Improvement in Manual–Automated Disagreement Outliers

<!-- Latest data (PI-CAI: {SITE}=PCNN/RUMC/ZGT; other cohorts unchanged): notebooks/volume_disagreement_quality_improvement.py -> results/volume_disagreement_quality_improvement/{SITE}/disagreement_thresholds.csv (cohort, scale; loa_lower, loa_upper); case_audit.csv (cohort=picai; flag_absolute, flag_relative, flag_both, flag_either); paired_volume_summary.csv (screen=both; n, n_cohort, review_fraction); run_metadata.json. -->

The strict intersection screen flagged 2/157 ProstateX patients (1.3%), 7/383 Toronto manual-reference patients (1.8%), and 70/4,105 expanded Toronto patients (1.7%). Absolute disagreement LoA were −14.28 to +25.74 mL, −16.75 to +20.57 mL, and −21.04 to +25.37 mL, respectively. Corresponding symmetric percentage LoA were −22.70% to +39.60%, −31.12% to +34.43%, and −34.31% to +37.61%. The flagged ProstateX and Toronto manual-reference cases are highlighted in Figures 1 and 2, respectively. Individual absolute/relative screens flagged 10/4, 17/29, and 216/193 patients, respectively; their unions flagged 12, 39, and 339. Requiring both scales therefore produced a substantially smaller review group.

In PI-CAI PCNN, the strict-intersection screen flagged 7/315 (2.2%). Absolute disagreement LoA were -35.10 to +30.08 mL, and symmetric percentage LoA were -60.93% to +44.09%. The absolute and relative screens flagged 19 and 16 patients, respectively; their union flagged 28. The strict-intersection cases are highlighted in Figure 3a.

In PI-CAI RUMC, the strict-intersection screen flagged 3/391 (0.8%). Absolute disagreement LoA were -14.41 to +27.97 mL, and symmetric percentage LoA were -26.83% to +48.94%. The absolute and relative screens flagged 22 and 12 patients, respectively; their union flagged 31. The strict-intersection cases are highlighted in Figure 3b.

In PI-CAI ZGT, the strict-intersection screen flagged 4/168 (2.4%). Absolute disagreement LoA were -18.60 to +15.38 mL, and symmetric percentage LoA were -40.02% to +29.70%. The absolute and relative screens flagged 7 and 9 patients, respectively; their union flagged 12. The strict-intersection cases are highlighted in Figure 3c.

<!-- Latest data (PI-CAI: {SITE}=PCNN/RUMC/ZGT; other cohorts unchanged): notebooks/volume_disagreement_quality_improvement.py -> results/volume_disagreement_quality_improvement/{SITE}/volume_summary.csv (cohort, screen=both, method=clinical/nnunet274; mae_ml, rmse_ml, mape_percent, rmspe_percent); paired_volume_summary.csv (n, n_cohort, improved, n_non_ties); run_metadata.json. -->

**Table 4. Volume errors in strict-intersection outliers. Values show manual ellipsoid → nnUNet274.**

| Cohort and reference | Flagged, n/N (%) | MAE (mL) | RMSE (mL) | MAPE (%) | RMSPE (%) | Automated closer, n/N (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RUMC ProstateX (manual reference) | 2/157 (1.3%) | 33.08 → 4.13 | 33.34 → 4.71 | 65.15 → 7.60 | 66.73 → 8.41 | 2/2 (100.0%) |
| PI-CAI PCNN (Bosma22b automated surrogate) | 7/315 (2.2%) | 39.14 → 20.50 | 47.92 → 24.98 | 57.43 → 39.81 | 66.64 → 53.86 | 5/7 (71.4%) |
| PI-CAI RUMC (Bosma22b automated surrogate) | 3/391 (0.8%) | 23.43 → 6.06 | 24.13 → 6.64 | 59.98 → 16.85 | 66.26 → 22.14 | 3/3 (100.0%) |
| PI-CAI ZGT (Bosma22b automated surrogate) | 4/168 (2.4%) | 26.02 → 1.50 | 28.03 → 2.00 | 58.06 → 4.09 | 61.13 → 6.13 | 4/4 (100.0%) |
| Toronto JDMI (manual reference) | 7/383 (1.8%) | 25.71 → 9.43 | 26.54 → 19.50 | 41.27 → 7.65 | 44.20 → 11.17 | 6/7 (85.7%) |
| Expanded Toronto JDMI (automated surrogate) | 70/4,105 (1.7%) | 34.69 → 7.22 | 38.96 → 11.51 | 52.98 → 12.47 | 64.56 → 26.67 | 65/70 (92.9%) |

There were no volume-error ties. For ProstateX, Toronto manual-reference, and expanded Toronto, exact 95% intervals for the probability that automation was closer were 15.8–100.0%, 42.1–99.6%, and 84.1–97.6%, respectively. Despite lower observed errors, ProstateX RMSE and RMSPE differences were not significant under the primary bootstrap tests (adjusted p = 0.999 and 0.499). Toronto manual-reference RMSPE was lower with automation (p = 0.025), while its RMSE difference was not significant (adjusted p = 0.500). Expanded Toronto RMSE and RMSPE differences favored automation (adjusted p < 0.001 and p = 0.010). Exact sign-test adjusted p-values were 0.999, 0.250, and <0.001, respectively. These tests describe selected subsets with fixed thresholds; the two-patient ProstateX subset provides particularly limited inference. Small-sample sensitivity results and confidence intervals are provided in the analysis outputs.

<!-- Latest data (PI-CAI: {SITE}=PCNN/RUMC/ZGT; other cohorts unchanged): notebooks/volume_disagreement_quality_improvement.py -> results/volume_disagreement_quality_improvement/{SITE}/paired_volume_summary.csv (cohort=picai, screen=both; improved, worsened, tied, closer_ci_lower/upper, worsening_median_ml, worsening_max_ml); primary_volume_statistics.csv (cohort=picai; rmse_bootstrap_p_holm, rmspe_bootstrap_p_holm, closer_exact_p_holm); run_metadata.json. -->

Within the 7 flagged PI-CAI PCNN patients, automation reduced absolute volume error in 5, worsened it in 2, and tied in 0 (Figure 4). The exact 95% interval for closer probability was 29.0–96.3%. Holm-adjusted p-values were 0.502 for RMSE, 0.604 for RMSPE, and 0.502 for the sign test. Among worsened cases, the median absolute-error increase was 34.78 mL and the maximum was 38.57 mL.

Within the 3 flagged PI-CAI RUMC patients, automation reduced absolute volume error in 3, worsened it in 0, and tied in 0 (Figure 4). The exact 95% interval for closer probability was 29.2–100.0%. Holm-adjusted p-values were 0.500 for RMSE, 0.334 for RMSPE, and 0.500 for the sign test.

Within the 4 flagged PI-CAI ZGT patients, automation reduced absolute volume error in 4, worsened it in 0, and tied in 0 (Figure 4). The exact 95% interval for closer probability was 39.8–100.0%. Holm-adjusted p-values were 0.335 for RMSE, 0.126 for RMSPE, and 0.250 for the sign test.

Automation worsened absolute volume error in one Toronto manual-reference case, from 26.73 to 51.11 mL, a 24.39 mL increase. In the expanded surrogate-reference analysis, five cases worsened, with a median increase of 22.55 mL and maximum increase of 33.73 mL. Thus, even marked disagreement did not guarantee that the automated estimate was better (Figure 4).

<!-- Latest data (PI-CAI: {SITE}=PCNN/RUMC/ZGT; other cohorts unchanged): notebooks/volume_disagreement_quality_improvement.py -> results/volume_disagreement_quality_improvement/{SITE}/paired_volume_summary.csv (screen=both; n) and case_audit.csv (flag_both, clinical_error_ml, nnunet274_error_ml); notebooks/volume_disagreement_quality_improvement.py (optional all-site figure section) -> results/paper_picai_sites/paired_volume_errors.png and run_metadata.json (six panels). -->

![Paired signed volume errors in six separately evaluated datasets.](results/paper_picai_sites/paired_volume_errors.png)

**Figure 4. Paired signed volume errors in disagreement outliers.** Panels show ProstateX (n = 2), Toronto manual-reference (n = 7), expanded Toronto surrogate-reference (n = 70), PI-CAI PCNN (n = 7), PI-CAI RUMC (n = 3), and PI-CAI ZGT (n = 4). Each line connects clinical ellipsoid error (open circle) to Multicenter error (triangle) at the same reference volume. Green indicates lower absolute error, orange higher error, and gray a tie. Expanded Toronto uses an nnUNet702 surrogate; the three PI-CAI panels use Bosma22b surrogates. Their panels assess surrogate agreement, not independent manual-reference accuracy. PI-CAI thresholds were estimated independently within each institution.

### PSAD Category Changes

One of the two flagged ProstateX patients lacked PSA. In the remaining patient, automation corrected the PSAD category, increasing reference-category agreement from 0/1 to 1/1 (exact McNemar p = 1.000), with reduced ordinal category distance. In the seven Toronto manual-reference outliers, four categories changed and all four became correct; agreement increased from 3/7 to 7/7 (p = 0.125), with four ordinal improvements and three unchanged distances.

Among 70 expanded Toronto outliers, 48 changed category. Automation corrected 37 assignments and made four previously correct assignments incorrect; reference-category agreement increased from 26/70 (37.1%) to 59/70 (84.3%; p < 0.001). Ordinal distance decreased in 40 patients, increased in four, and was unchanged in 26. These expanded results describe agreement with surrogate-derived PSAD categories.

<!-- Latest data (PI-CAI: {SITE}=PCNN/RUMC/ZGT; other cohorts unchanged): notebooks/volume_disagreement_quality_improvement.py -> results/volume_disagreement_quality_improvement/{SITE}/psad_category_summary.csv (cohort=picai, screen=both; n, category_changed, clinical_correct, automated_correct, corrected, newly_incorrect, ordinal_closer/farther/equal, mcnemar_exact_p_holm); run_metadata.json. -->

In PI-CAI PCNN, all 7 flagged patients had evaluable PSAD. Categories changed in 5 cases; one incorrect assignment became correct and 2 correct assignments became incorrect. Exact Bosma22b-derived category agreement changed from 4/7 (57.1%) to 3/7 (42.9%) (Holm-adjusted McNemar p = 1.000). Ordinal distance decreased in 2, increased in 3, and was unchanged in 2.

In PI-CAI RUMC, all 3 flagged patients had evaluable PSAD. The category changed in one case; one incorrect assignment became correct and 0 correct assignments became incorrect. Exact Bosma22b-derived category agreement changed from 2/3 (66.7%) to 3/3 (100.0%) (Holm-adjusted McNemar p = 1.000). Ordinal distance decreased in 1, increased in 0, and was unchanged in 2.

In PI-CAI ZGT, all 4 flagged patients had evaluable PSAD. The category changed in one case; one incorrect assignment became correct and 0 correct assignments became incorrect. Exact Bosma22b-derived category agreement changed from 2/4 (50.0%) to 3/4 (75.0%) (Holm-adjusted McNemar p = 1.000). Ordinal distance decreased in 1, increased in 0, and was unchanged in 3.

### PI-RADS 3 Decisions and Recorded Outcomes

No flagged ProstateX patients were eligible for the PI-RADS 3 decision analysis. One Toronto manual-reference outlier was eligible; manual, automated, and reference-volume methods all recommended biopsy. In expanded Toronto, 13 outliers had PI-RADS 3 and evaluable PSAD, including nine with unknown cancer outcomes. Five decisions changed: automation corrected four reference-decision disagreements and introduced one, increasing reference-decision agreement from 8/13 to 11/13.

Known-outcome analyses included the single Toronto manual-reference outlier and four expanded Toronto outliers; all had recorded negative outcomes. In the manual-reference case, all methods recommended biopsy (FP = 1, TN = 0; specificity 0/1). In expanded Toronto, manual ellipsoid estimates would avoid biopsy in 2/4 patients (FP = 2, TN = 2; specificity 50.0%), compared with 3/4 for nnUNet274 and the surrogate (FP = 1, TN = 3; specificity 75.0%). These Toronto groups had zero true positives and false negatives because there were no outcome-positive patients. Sensitivity was undefined, and these very small subgroups cannot establish improved cancer detection.

<!-- Latest data (PI-CAI: {SITE}=PCNN/RUMC/ZGT; other cohorts unchanged): notebooks/volume_disagreement_quality_improvement.py -> results/volume_disagreement_quality_improvement/{SITE}/biopsy_eligibility.csv (cohort=picai, screen=both; n_reference_decisions, n_known_outcomes); biopsy_reference_decision_summary.csv (clinical_reference_agreement, automated_reference_agreement, corrected, newly_incorrect); biopsy_outcome_summary.csv (method=gt/clinical/nnunet274; tp, fn, fp, tn, biopsies_avoided); run_metadata.json. -->

PI-CAI PCNN had 1 flagged PI-RADS 3 patient with evaluable PSAD and 1 with known outcome. Reference-decision agreement changed from 0/1 to 0/1; no disagreements were corrected or introduced. Bosma22b reference had TP = 0, FN = 0, FP = 1, TN = 0, sensitivity undefined (zero denominator), specificity 0/1 (0.0%), and simulated biopsy avoidance 0/1 (0.0%). Clinical ellipsoid had TP = 0, FN = 0, FP = 0, TN = 1, sensitivity undefined (zero denominator), specificity 1/1 (100.0%), and simulated biopsy avoidance 1/1 (100.0%). Multicenter had TP = 0, FN = 0, FP = 0, TN = 1, sensitivity undefined (zero denominator), specificity 1/1 (100.0%), and simulated biopsy avoidance 1/1 (100.0%). These single-patient results cannot establish improved cancer detection.

PI-CAI RUMC had 1 flagged PI-RADS 3 patient with evaluable PSAD and 1 with known outcome. Reference-decision agreement changed from 0/1 to 1/1; one disagreement was corrected and none introduced. Bosma22b reference had TP = 1, FN = 0, FP = 0, TN = 0, sensitivity 1/1 (100.0%), specificity undefined (zero denominator), and simulated biopsy avoidance 0/1 (0.0%). Clinical ellipsoid had TP = 0, FN = 1, FP = 0, TN = 0, sensitivity 0/1 (0.0%), specificity undefined (zero denominator), and simulated biopsy avoidance 1/1 (100.0%). Multicenter had TP = 1, FN = 0, FP = 0, TN = 0, sensitivity 1/1 (100.0%), specificity undefined (zero denominator), and simulated biopsy avoidance 0/1 (0.0%). These single-patient results cannot establish improved cancer detection.

No flagged PI-CAI ZGT patients met the PI-RADS 3 decision or known-outcome criteria. Decision-agreement, sensitivity, specificity, and biopsy-avoidance rates were undefined for this empty subgroup.

# Discussion

The proposed review trigger identifies unusually discordant clinical and automated volume estimates using information available without a manual reference. Automation reduced observed volume error in most flagged cases, but worsened some measurements and PSAD categories. These findings support potential measurement review rather than automatic correction or proof of a particular manual error. Adjudication is needed to distinguish reporting or calculation mistakes, limitations of ellipsoid approximation, segmentation failures, and reference variability.

The strict intersection yields a small review workload but misses discrepancies detected on only one scale. Empirical LoA depend on cohort composition and can widen when extreme errors inflate the standard deviation; they are not clinical acceptability thresholds. Threshold estimation and evaluation used the same patients, and the reported resampling procedures condition on fixed selection. The small manual-reference groups, differing small-sample inference procedures, and sparse PI-RADS 3 outcomes limit conclusions. Prospective calibration and review are needed before claiming operational error mitigation. ProstateX evaluation additionally excludes otherwise eligible scans with manual-mask boundary contact; this post hoc coverage-based selection may favor easier cases and limits generalization to scans with incomplete or uncertain coverage.

The expanded Toronto analysis adds scale but uses an automated surrogate that may share errors with nnUNet274. Its 383-patient manual-reference subset is overlapping rather than an independent replication. Manual planimetry also has variability; the present comparisons do not establish why reference volumes differ between cohorts.

<!-- Historical diagnostic outputs (generation notebooks retired; retained output provenance describes the original runs): results/picai_rumc_error_review/{SITE}/reviewed_mask_diagnostics.csv (dsc/reference coverage in the eight highest-error cases), errors_with_image_metadata.csv and fov_summary.csv; run_metadata.json. notebooks/picai_test_clinical_volume.py -> results/picai_test_clinical_volume/{SITE}/run_metadata.json (prostatex_shared_patients). -->

The three PI-CAI analyses use automated surrogate references; PI-CAI RUMC overlaps ProstateX by 116 patients, while PCNN and ZGT have no ProstateX overlap. Bosma22b training independence is not established here. Failures were strongly site-dependent: PCNN contained all 14 Multicenter and 131 RUMC zero-overlap cases, whereas neither model had zero overlap in RUMC or ZGT. The PCNN image review found zero Bosma22b overlap in the eight largest RUMC volume-error cases. The association with wider fields of view is post hoc and does not identify the cause. Manually imposed largest-component filtering does not guarantee anatomical correctness; a retained non-prostate component may have a plausible volume. Volume agreement alone therefore does not demonstrate segmentation accuracy. The post hoc processing choice, six manually reviewed surrogate-mask exclusions, and reference-mask boundary exclusions limit generalization and require independent validation.

# Appendix

## Appendix A. PSAD Signed Differences

In the two manual-reference cohorts, both models overestimated PSAD on average. In ProstateX, mean signed differences for the Multicenter and RUMC Models were +0.0072 and +0.0053 ng/mL² (mean relative differences, +4.1% and +3.1%), respectively, compared with −0.0148 ng/mL² (−3.1%) for ellipsoid estimates. In Toronto, corresponding mean signed differences were +0.0104 and +0.0175 ng/mL² (+6.6% and +10.1%), compared with +0.0092 ng/mL² (+6.0%) for ellipsoid estimates. Both models had narrower observed absolute and relative limits of agreement than ellipsoid estimates in each cohort (Figures A1–A2).

![PSAD agreement and signed differences against manual-mask reference PSAD for 156 RUMC ProstateX patients.](results/prostatex_linked_volume/psa_density_agreement.png)

**Figure A1. PSAD agreement and signed differences in RUMC ProstateX (n = 156).** Columns show clinical manual ellipsoid estimates ("Manual ellipsoid"), the Multicenter Model ("Multicenter"), and the RUMC Model ("RUMC (same inst.)"). All densities use the same patient-level PSA. The top row plots estimated against reference PSAD derived from manual whole-gland segmentation volume, with both axes on logarithmic scales and the dashed identity line indicating equality. The middle row shows estimated minus reference PSAD (ng/mL²); the bottom row shows relative differences, 100 × (estimated − reference)/reference (%). Both difference rows use reference PSAD on a logarithmic x-axis. Solid horizontal lines indicate mean signed differences, and dashed horizontal lines indicate 95% limits of agreement (mean ±1.96 sample standard deviations), distinct from confidence intervals. Histograms at right show counts of differences across all included patients. Positive differences indicate PSAD overestimation.

![PSAD agreement and signed differences against manual-reader planimetry reference PSAD for 383 Toronto JDMI patients.](results/toronto_jdmi_clinical_volume/psa_density_agreement.png)

**Figure A2. PSAD agreement and signed differences in Toronto JDMI (n = 383).** Columns show clinical manual ellipsoid estimates ("Manual ellipsoid"), the Multicenter Model ("Multicenter"), and the RUMC Model ("RUMC"). All densities use the same patient-level PSA. Reference PSAD is derived from median manual-reader planimetry volume. The top row plots estimated against reference PSAD on logarithmic axes, with a dashed identity line. The middle and bottom rows show signed differences (estimated − reference, ng/mL²) and relative differences (100 × (estimated − reference)/reference, %), respectively, against reference PSAD on a logarithmic x-axis. Solid horizontal lines indicate mean signed differences, and dashed horizontal lines indicate 95% limits of agreement (mean ±1.96 sample standard deviations), distinct from confidence intervals. Histograms at right show counts of differences across all included patients. Positive differences indicate PSAD overestimation.

<!-- Latest data: notebooks/picai_test_clinical_volume.py -> results/picai_test_clinical_volume/{SITE}/psa_density_summary.csv (method; bias_psad, relative_bias_percent); psa_density_agreement.png; run_metadata.json. -->

In PI-CAI PCNN (n = 315), mean signed PSAD differences were -0.0269 ng/mL² (-9.7%) for Multicenter, -0.0436 ng/mL² (-16.4%) for RUMC, +0.0153 ng/mL² (+1.3%) for clinical ellipsoid (Figure A3a). Each method used the same patient PSA and Bosma22b volume defined reference PSAD.

In PI-CAI RUMC (n = 391), mean signed PSAD differences were -0.0164 ng/mL² (-4.7%) for Multicenter, -0.0040 ng/mL² (-0.7%) for RUMC, -0.0253 ng/mL² (-12.4%) for clinical ellipsoid (Figure A3b). Each method used the same patient PSA and Bosma22b volume defined reference PSAD.

In PI-CAI ZGT (n = 168), mean signed PSAD differences were -0.0136 ng/mL² (-2.3%) for Multicenter, -0.0098 ng/mL² (-2.3%) for RUMC, +0.0100 ng/mL² (+4.3%) for clinical ellipsoid (Figure A3c). Each method used the same patient PSA and Bosma22b volume defined reference PSAD.

![PSAD agreement against Bosma22b-derived reference PSAD in PI-CAI PCNN (n = 315).](results/picai_test_clinical_volume/PCNN/psa_density_agreement.png)

**Figure A3a. PSAD agreement against Bosma22b-derived reference PSAD in PI-CAI PCNN (n = 315).** Columns show clinical ellipsoid, Multicenter, and RUMC estimates. Rows show estimated versus reference PSAD, signed differences (ng/mL²), and reference-normalized percentage differences. Reference axes are logarithmic; horizontal lines show bias and descriptive 95% limits of agreement, and histograms summarize each site.

![PSAD agreement against Bosma22b-derived reference PSAD in PI-CAI RUMC (n = 391).](results/picai_test_clinical_volume/RUMC/psa_density_agreement.png)

**Figure A3b. PSAD agreement against Bosma22b-derived reference PSAD in PI-CAI RUMC (n = 391).** Columns show clinical ellipsoid, Multicenter, and RUMC estimates. Rows show estimated versus reference PSAD, signed differences (ng/mL²), and reference-normalized percentage differences. Reference axes are logarithmic; horizontal lines show bias and descriptive 95% limits of agreement, and histograms summarize each site.

![PSAD agreement against Bosma22b-derived reference PSAD in PI-CAI ZGT (n = 168).](results/picai_test_clinical_volume/ZGT/psa_density_agreement.png)

**Figure A3c. PSAD agreement against Bosma22b-derived reference PSAD in PI-CAI ZGT (n = 168).** Columns show clinical ellipsoid, Multicenter, and RUMC estimates. Rows show estimated versus reference PSAD, signed differences (ng/mL²), and reference-normalized percentage differences. Reference axes are logarithmic; horizontal lines show bias and descriptive 95% limits of agreement, and histograms summarize each site.

## Appendix B. Expanded Toronto JDMI Cohort: Volume and PSAD Agreement

Supplementary agreement analyses included all 4,105 patients in the expanded Toronto JDMI cohort, using whole-gland volumes from the same-institution nnUNet702 model as an automated surrogate reference. Figure B1 shows volume agreement and signed differences for clinical manual ellipsoid estimates, the Multicenter Model, and the RUMC Model. No retained surrogate-reference volumes were below 1 mL; the smallest was 11.43 mL. These results describe agreement with an automated surrogate reference.

![Whole-gland volume agreement and signed differences against nnUNet702 surrogate-reference volume for 4,105 Toronto JDMI patients.](results/toronto_jdmi_4111_clinical_volume/comparator_agreement.png)

**Figure B1. Whole-gland volume agreement and signed differences in the expanded Toronto JDMI cohort (n = 4,105).** Columns show clinical manual ellipsoid estimates ("Manual ellipsoid"), the Multicenter Model ("Multicenter"), and the RUMC Model ("RUMC"). The top row plots measured against nnUNet702 surrogate-reference volume on logarithmic axes, with the dashed identity line indicating equality. The middle row shows measured minus reference volume in mL; the bottom row shows relative differences, 100 × (measured − reference)/reference (%). Both difference rows use reference volume on a logarithmic x-axis, with positive differences indicating overestimation. Solid horizontal lines denote mean signed differences and dashed horizontal lines denote descriptive 95% limits of agreement (mean ±1.96 sample standard deviations), distinct from confidence intervals. Histograms at right show counts of differences. The relative-difference display is limited to −100.0% to +100.0%; observations outside this range remain in the numerical summaries.

Exact agreement with the surrogate-derived PSAD stratum was 3,631/4,105 (88.5%) for the Multicenter Model, 3,534/4,105 (86.1%) for the RUMC Model, and 3,218/4,105 (78.4%) for clinical manual ellipsoid estimates (Figure B2).

![Four-stratum PSAD categorization against the nnUNet702 surrogate reference for 4,105 Toronto JDMI patients.](results/toronto_jdmi_4111_clinical_volume/psa_density_confusion.png)

**Figure B2. PSAD 4×4 categorization matrices in the expanded Toronto JDMI cohort (n = 4,105).** Panels show clinical manual ellipsoid estimates, the Multicenter Model, and the RUMC Model. Rows indicate PSAD strata derived from the same-institution nnUNet702 automated surrogate-reference volume; columns indicate strata derived from each estimated volume. All methods use the same patient-level PSA. Strata are 0–<0.10, 0.10–<0.15, 0.15–<0.20, and ≥0.20 ng/mL², with values equal to a cutoff assigned to the higher stratum. Cells display patient counts and percentages within each reference row. These matrices assess agreement with surrogate-derived PSAD categories.

## Appendix C. Internal Consistency Analysis: Clinical Manual Ellipsoid Calculations

This retained supplementary analysis uses its saved 4,109-examination snapshot after seven accession exclusions, rather than the current 4,105-examination expanded-cohort snapshot after 11 exclusions. We screened reported clinical prostate volumes for internal consistency with their recorded dimensions to identify potential manual transcription or calculation errors in the n = 4,109 Toronto JDMI cohort. For recorded dimensions $d_1$, $d_2$, and $d_3$ in cm, the nominal ellipsoid volume in mL was recalculated as

$$
V_{\mathrm{nominal}}=\frac{\pi}{6}d_1d_2d_3.
$$

The screen allowed bounds of ±0.1 cm per dimension (for rounding) and coefficients from 0.5 (downward rounding from π/6) to 5π/24 (allowing use of the alternative bullet-volume formula). These deliberately generous assumptions gave inclusive lower and upper bounds of

$$
V_{\mathrm{lower}}=0.5(d_1-0.1)(d_2-0.1)(d_3-0.1)
$$

and

$$
V_{\mathrm{upper}}=\frac{5\pi}{24}(d_1+0.1)(d_2+0.1)(d_3+0.1).
$$

Reported volumes outside these bounds were classified as internally inconsistent. Above-bound reports falling within the interval from $(d_1-0.1)(d_2-0.1)(d_3-0.1)$ to $(d_1+0.1)(d_2+0.1)(d_3+0.1)$ were specially assigned to the missing-coefficient class, i.e., consistent with use of $V=d_1d_2d_3$ without a volume coefficient. Remaining above-bound reports were assigned to the above-range class, and below-bound reports to the below-range class, such that these three inconsistency classes were mutually exclusive (Table C1). The screen identifies discrepancies but does not establish their cause, with possible explanations including transcription errors, incorrect decimal placement, or use of a wrong coefficient.

**Table C1. Consistency of reported clinical volumes with recorded dimensions (n = 4,109).**

| Classification | Examinations, n (%) | Interpretation |
| --- | ---: | --- |
| Within range | 4,044 (98.42%) | Reported volume within the inclusive tolerance bounds |
| Missing coefficient | 13 (0.32%) | Reported volume approximately equal to $d_1d_2d_3$, consistent with accidental omission of the volume coefficient |
| Above range | 17 (0.41%) | Reported volume above the upper bound, excluding the missing-coefficient class |
| Below range | 35 (0.85%) | Reported volume below the lower bound |

Overall, 65/4,109 examinations (1.58%) were flagged. Median nominal ellipsoid minus reported volume was +14.86 mL in the below-range class, −32.27 mL in the above-range class, and −34.56 mL in the missing-coefficient class (Figure C1).

![Consistency of reported clinical volumes with dimension-derived ellipsoid volumes in 4,109 Toronto JDMI examinations.](results/toronto_jdmi_ellipsoid_tolerance/report_volume_quality_full_range_inset.png)

**Figure C1. Bland-Altman plot illustrating internal consistency of clinical ellipsoid volume calculations.** The horizontal axis shows the mean of nominal ellipsoid and reported volume (mL; logarithmic scale), and the vertical axis shows nominal ellipsoid minus reported volume (mL). The main panel includes all 4,109 eligible examinations, including the 65 flagged inconsistencies. The inset shows mean volumes of 10–300 mL and differences of −60 to +60 mL.

**Table C2. Volume agreement with the nnU-Net automated surrogate in flagged examinations (n = 65).**

| Method | Bias (mL) | MAE (mL) | RMSE (mL) | MAPE (%) | RMSPE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multicenter Model | −2.99 | 4.24 | 5.48 | 7.33 | 9.31 |
| Reported clinical volume | +8.46 | 22.73 | 31.97 | 41.84 | 61.18 |
| Recalculated ellipsoid | +51.06 | 65.64 | 308.82 | 70.80 | 233.19 |

The Multicenter Model, reported clinical volume, and nominal ellipsoid volume recalculated from the recorded dimensions were evaluated in all 65 flagged examinations against whole-gland volumes from a same-institution nnU-Net model as an automated surrogate reference (Table C2). The Multicenter Model, representing an automated nnU-Net approach, had lower MAE and RMSE than either clinical estimate. Additionally, in a subset of seven flagged examinations with positive finite median manual-reader planimetry references, RMSE was 9.39 mL for the Multicenter Model, 10.09 mL for reported volume, and 909.16 mL for recalculated ellipsoid volume; corresponding RMSPEs were 12.54%, 15.99%, and 631.53%. The extreme recalculated-volume discrepancies are compatible with errors in recorded dimensions; the screen alone does not establish their cause.

Figure C2 further shows that automation through nnU-Net reduced absolute volume error in 54 examinations, from a median of 20.09 mL for reported volume to 3.19 mL for the Multicenter Model. Many of these examinations had substantial reporting errors. In the remaining 11 examinations, reported volumes generally already agreed closely with the surrogate: median absolute error was 1.37 mL, compared with 6.84 mL for the model, with a median paired increase of 3.80 mL. These smaller discrepancies are compatible with inter-method variability. Automated segmentation may therefore support review of potential recording or calculation errors by reducing large volume discrepancies.

![Paired changes from reported-volume error to Multicenter Model error against the nnUNet702 surrogate in flagged Toronto JDMI examinations.](results/toronto_jdmi_ellipsoid_tolerance/paired_signed_error_reported_vs_nnunet702.png)

**Figure C2. Paired signed volume errors in internally inconsistent reports (n = 65).** The horizontal axis shows nnUNet702 automated surrogate-reference volume (mL; logarithmic scale), and the vertical axis shows estimated minus reference volume (mL; linear scale). Each line connects a reported-volume error (circle) to the Multicenter Model (nnUNet274) error (triangle) for the same examination. Solid green lines indicate reduced absolute error, and dashed orange lines indicate increased absolute error.

## Appendix D. Full-Cohort PSAD Categorization and Biopsy Simulation

### Full-cohort PSAD methods

PSA density (PSAD) was recalculated as PSA (ng/mL) divided by whole-gland volume (mL), in ng/mL², using the same patient-level PSA for every volume method. Reference PSAD used manual whole-gland segmentation volume in RUMC ProstateX, Bosma22b automated whole-gland surrogate volume within each PI-CAI site, and median manual-reader planimetry volume in Toronto JDMI. These were derived references rather than independent PSAD measurements; reported clinical PSAD values were not substituted. The ProstateX analysis included 156 of 157 selected patients, excluding ProstateX-0202 because PSA was missing, the Toronto analysis included all 383 volume-analysis patients with valid PSA and volume inputs, and the PI-CAI analyses included all 315 PCNN, 391 RUMC, and 168 ZGT testing patients separately.

Unrounded PSAD values were categorized into four strata: 0–<0.10, 0.10–<0.15, 0.15–<0.20, and ≥0.20 ng/mL², with values equal to a cutoff assigned to the higher stratum. For each method, 4×4 confusion matrices compared reference strata (rows) with estimated strata (columns), showing counts and percentages within each reference stratum. Exact stratum agreement was compared between methods using two-sided exact McNemar tests of paired correct/incorrect assignments, with Holm correction across the three method pairs separately within each cohort. These tests assessed PSAD categorization agreement rather than cancer detection. Continuous signed differences were estimated minus reference PSAD, with relative differences calculated as 100 × (estimated − reference)/reference. Bias and 95% limits of agreement were calculated as the mean signed difference and bias ± 1.96 sample standard deviations, respectively; the plots are presented in Appendix A.

### Full-cohort biopsy methods

Biopsy recommendations were simulated at unrounded PSAD ≥0.15 ng/mL² using reference-volume-derived PSAD, clinical manual ellipsoid estimates, and both models. In ProstateX and each PI-CAI site, eligibility required a maximum recorded lesion PI-RADS score of 3, evaluable PSAD, and a valid case-level International Society of Urological Pathology (ISUP) grade group; clinically significant prostate cancer (csPCa) was defined as ISUP grade group ≥2. In Toronto, eligibility required a reported maximum PI-RADS score of 3, evaluable PSAD, and a known recorded CSCANCER outcome (Y positive, N negative). No ISUP grades were inferred for Toronto, and unknown CSCANCER outcomes were excluded only from this biopsy analysis. Each cohort's methods were evaluated on the same eligible patients using 2×2 confusion matrices with recorded outcomes as rows and simulated avoid/recommend biopsy decisions as columns. Sensitivity used outcome-positive patients as its denominator, specificity used outcome-negative patients, and the fraction of biopsies avoided used all eligible patients, including false negatives. This retrospective simulation assessed a threshold rule rather than observed clinical management; the cohorts' outcome definitions were retained separately.

An exploratory expanded Toronto JDMI analysis included 4,105 patients, each contributing one examination after the current 11 accession-level exclusions. Whole-gland volume from the union of labels 1, 2, and 3 in nnUNet702 segmentations, generated by a model from the same institution, served as an automated surrogate reference. Reference and comparator PSAD used the same patient-level PSA. The expanded biopsy simulation retained patients with reported maximum PI-RADS 3, evaluable PSAD, and known recorded CSCANCER outcomes (Y positive, N negative), applying the same unrounded PSAD ≥0.15 ng/mL² threshold. Unknown outcomes were excluded only from the biopsy analysis. Supplementary volume and PSAD agreement analyses used all 4,105 patients.

### Full-cohort results

For each volumetry method, PSAD was calculated as PSA divided by the estimated volume and assigned to one of four clinical strata. Categorization accuracy was assessed by comparing these assignments with those obtained using the reference volume and the same patient-level PSA. In RUMC ProstateX, the Multicenter and RUMC Models correctly assigned 139/156 patients (89.1%) and 138/156 (88.5%), respectively, to the reference PSAD stratum, compared with 130/156 (83.3%) for clinical manual ellipsoid estimates (Figure D1). Differences in exact stratum agreement were not significant after Holm correction (exact McNemar p = 0.279 for either model versus ellipsoid estimates; p = 1.000 between models). Among discordant assignments, the models more often placed patients in higher PSAD strata than in lower strata relative to the reference: 11 versus 6 for the Multicenter Model and 10 versus 8 for the RUMC Model. Ellipsoid estimates placed eight patients in higher strata and 18 in lower strata.

In Toronto JDMI, exact stratum agreement was 321/383 (83.8%) for the Multicenter Model, 317/383 (82.8%) for the RUMC Model, and 298/383 (77.8%) for ellipsoid estimates (Figure D2). Pairwise differences did not reach statistical significance after Holm correction (exact McNemar p = 0.080 and 0.123 for ellipsoid estimates versus the Multicenter and RUMC Models, respectively; p = 0.644 between models). Among discordant assignments, both models and ellipsoid estimates favored higher PSAD strata than the reference. This upward tendency was most pronounced for the RUMC Model (62 higher versus 4 lower assignments), compared with 49 versus 13 for the Multicenter Model and 54 versus 31 for ellipsoid estimates. For all three methods, this tendency toward higher PSAD strata is consistent with their lower mean volumes relative to the Toronto planimetry reference (Figure 2), because a smaller volume yields a higher PSAD for the same PSA. Continuous PSAD agreement and signed differences are shown in Appendix A (Figures A1–A2).

![PSAD categorization in four strata for 156 RUMC ProstateX patients.](results/prostatex_linked_volume/psa_density_confusion.png)

**Figure D1. PSAD 4×4 categorization matrices in RUMC ProstateX (n = 156).** Panels show clinical manual ellipsoid estimates, the Multicenter Model, and the RUMC Model. Rows indicate PSAD strata derived from manual whole-gland segmentation volume; columns indicate strata derived from each estimated volume. Cells display patient counts and percentages within the reference row.

![PSAD categorization in four strata for 383 Toronto JDMI patients.](results/toronto_jdmi_clinical_volume/psa_density_confusion.png)

**Figure D2. PSAD 4×4 categorization matrices in Toronto JDMI (n = 383).** Panels show clinical manual ellipsoid estimates, the Multicenter Model, and the RUMC Model.

The ProstateX PI-RADS 3 subgroup comprised 23 patients, including five with csPCa and 18 without csPCa. At PSAD ≥0.15 ng/mL², all four volume methods produced the same counts (Figure D3). Each method would recommend biopsy in 16/23 patients (69.6%) and avoid biopsy in 7/23 (30.4%), with sensitivity of 100% (5/5) and specificity of 38.9% (7/18). No csPCa cases would be missed in this selected subgroup.

Of 62 Toronto patients with reported PI-RADS 3, 39 had unknown CSCANCER outcomes, leaving 23 patients (4 positive and 19 negative; Figure D4). Reference planimetry and both models would avoid biopsy in 18/23 patients (78.3%), versus 16/23 (69.6%) for ellipsoid estimates. All methods had sensitivity of 50.0% (2/4), with two recorded positive cases below the threshold. Specificity was 84.2% (16/19) for reference planimetry and both models and 73.7% (14/19) for ellipsoid estimates.

![Simulated biopsy decisions versus case-level ISUP outcomes in 23 RUMC ProstateX PI-RADS 3 patients.](results/prostatex_linked_volume/psa_density_biopsy_confusion.png)

**Figure D3. Biopsy 2×2 confusion matrices in RUMC ProstateX patients with maximum lesion PI-RADS 3 (n = 23).** Panels show PSAD derived from reference manual whole-gland segmentation, clinical manual ellipsoid estimates, the Multicenter Model, and the RUMC Model. Rows indicate no csPCa (ISUP grade groups 0–1) or csPCa (ISUP grade groups 2–5); columns indicate simulated avoidance of biopsy (PSAD <0.15 ng/mL²) or recommendation of biopsy (PSAD ≥0.15 ng/mL²). Cells display patient counts and percentages within each outcome row.

![Simulated biopsy decisions versus recorded CSCANCER outcomes in 23 Toronto JDMI PI-RADS 3 patients.](results/toronto_jdmi_clinical_volume/psa_density_biopsy_confusion.png)

**Figure D4. Biopsy 2×2 confusion matrices in Toronto JDMI patients with reported PI-RADS 3 and known CSCANCER outcomes (n = 23).** Panels show PSAD derived from reference manual whole-gland segmentation, clinical manual ellipsoid estimates, the Multicenter Model, and the RUMC Model. Rows indicate recorded CSCANCER outcomes (N negative, Y positive); columns indicate simulated avoid/recommend biopsy decisions.

<!-- Latest data: notebooks/picai_test_clinical_volume.py -> results/picai_test_clinical_volume/{SITE}/psa_density_classification_summary.csv (method; n, equal_stratum_n, lower_stratum_n, higher_stratum_n); psa_density_stratum_pairwise_statistics.csv (method_a/method_b; mcnemar_exact_p_holm); psa_density_confusion.png; run_metadata.json. -->

### PI-CAI site-specific full-cohort results

In PI-CAI PCNN, exact agreement with Bosma22b-derived PSAD strata was 245/315 (77.8%) for Multicenter, 224/315 (71.1%) for RUMC, 209/315 (66.3%) for clinical ellipsoid (Figure D5a). Multicenter placed 68 patients below and 2 above the reference stratum. RUMC placed 88 patients below and 3 above the reference stratum. Clinical ellipsoid placed 59 patients below and 47 above the reference stratum. For clinical ellipsoid versus Multicenter, Holm-adjusted McNemar p = 0.002. For clinical ellipsoid versus RUMC, Holm-adjusted McNemar p = 0.172. For Multicenter versus RUMC, Holm-adjusted McNemar p = 0.048. These are surrogate-category comparisons rather than cancer-detection results.

In PI-CAI RUMC, exact agreement with Bosma22b-derived PSAD strata was 350/391 (89.5%) for Multicenter, 374/391 (95.7%) for RUMC, 279/391 (71.4%) for clinical ellipsoid (Figure D5b). Multicenter placed 35 patients below and 6 above the reference stratum. RUMC placed 11 patients below and 6 above the reference stratum. Clinical ellipsoid placed 107 patients below and 5 above the reference stratum. For clinical ellipsoid versus Multicenter, Holm-adjusted McNemar p < 0.001. For clinical ellipsoid versus RUMC, Holm-adjusted McNemar p < 0.001. For Multicenter versus RUMC, Holm-adjusted McNemar p < 0.001. These are surrogate-category comparisons rather than cancer-detection results.

In PI-CAI ZGT, exact agreement with Bosma22b-derived PSAD strata was 154/168 (91.7%) for Multicenter, 158/168 (94.0%) for RUMC, 130/168 (77.4%) for clinical ellipsoid (Figure D5c). Multicenter placed 11 patients below and 3 above the reference stratum. RUMC placed 9 patients below and 1 above the reference stratum. Clinical ellipsoid placed 16 patients below and 22 above the reference stratum. For clinical ellipsoid versus Multicenter, Holm-adjusted McNemar p < 0.001. For clinical ellipsoid versus RUMC, Holm-adjusted McNemar p < 0.001. For Multicenter versus RUMC, Holm-adjusted McNemar p = 0.388. These are surrogate-category comparisons rather than cancer-detection results.

![PSAD four-stratum categorization in PI-CAI PCNN (n = 315).](results/picai_test_clinical_volume/PCNN/psa_density_confusion.png)

**Figure D5a. PSAD four-stratum categorization in PI-CAI PCNN (n = 315).** Columns are estimated strata; rows are Bosma22b-derived strata. Panels show clinical ellipsoid, Multicenter, and RUMC estimates using the same patient PSA. Strata are 0–<0.10, 0.10–<0.15, 0.15–<0.20, and ≥0.20 ng/mL², assigning cutoff equality to the higher stratum. Cells show counts and percentages within reference rows.

![PSAD four-stratum categorization in PI-CAI RUMC (n = 391).](results/picai_test_clinical_volume/RUMC/psa_density_confusion.png)

**Figure D5b. PSAD four-stratum categorization in PI-CAI RUMC (n = 391).** Columns are estimated strata; rows are Bosma22b-derived strata. Panels show clinical ellipsoid, Multicenter, and RUMC estimates using the same patient PSA. Strata are 0–<0.10, 0.10–<0.15, 0.15–<0.20, and ≥0.20 ng/mL², assigning cutoff equality to the higher stratum. Cells show counts and percentages within reference rows.

![PSAD four-stratum categorization in PI-CAI ZGT (n = 168).](results/picai_test_clinical_volume/ZGT/psa_density_confusion.png)

**Figure D5c. PSAD four-stratum categorization in PI-CAI ZGT (n = 168).** Columns are estimated strata; rows are Bosma22b-derived strata. Panels show clinical ellipsoid, Multicenter, and RUMC estimates using the same patient PSA. Strata are 0–<0.10, 0.10–<0.15, 0.15–<0.20, and ≥0.20 ng/mL², assigning cutoff equality to the higher stratum. Cells show counts and percentages within reference rows.

<!-- Latest data: notebooks/picai_test_clinical_volume.py -> results/picai_test_clinical_volume/{SITE}/psa_density_biopsy_summary.csv (method=gt/clinical/nnunet274/rumcnotprostatexnoclinical236; n, n_cspca, n_non_cspca, tp, fn, fp, tn, biopsies_avoided_n); psa_density_biopsy_audit.csv (biopsy_analysis_status); psa_density_biopsy_confusion.png; run_metadata.json. -->

The PI-CAI PCNN PI-RADS 3 subgroup comprised 42 patients with evaluable outcomes, including 2 with csPCa and 40 without (Figure D6a; Table D1).

The PI-CAI RUMC PI-RADS 3 subgroup comprised 72 patients with evaluable outcomes, including 12 with csPCa and 60 without (Figure D6b; Table D1).

The PI-CAI ZGT PI-RADS 3 subgroup comprised 15 patients with evaluable outcomes, including 1 with csPCa and 14 without (Figure D6c; Table D1).

**Table D1. Simulated biopsy outcomes within each PI-CAI site at PSAD ≥0.15 ng/mL².**

| Dataset | Volume method | Biopsies avoided, n/N (%) | Sensitivity, n/N (%) | Specificity, n/N (%) | csPCa below threshold, n |
| --- | --- | ---: | ---: | ---: | ---: |
| PI-CAI PCNN | Bosma22b reference | 18/42 (42.9%) | 2/2 (100.0%) | 18/40 (45.0%) | 0 |
| PI-CAI PCNN | Clinical ellipsoid | 16/42 (38.1%) | 2/2 (100.0%) | 16/40 (40.0%) | 0 |
| PI-CAI PCNN | Multicenter | 21/42 (50.0%) | 2/2 (100.0%) | 21/40 (52.5%) | 0 |
| PI-CAI PCNN | RUMC | 21/42 (50.0%) | 2/2 (100.0%) | 21/40 (52.5%) | 0 |
| PI-CAI RUMC | Bosma22b reference | 29/72 (40.3%) | 10/12 (83.3%) | 27/60 (45.0%) | 2 |
| PI-CAI RUMC | Clinical ellipsoid | 36/72 (50.0%) | 7/12 (58.3%) | 31/60 (51.7%) | 5 |
| PI-CAI RUMC | Multicenter | 32/72 (44.4%) | 9/12 (75.0%) | 29/60 (48.3%) | 3 |
| PI-CAI RUMC | RUMC | 30/72 (41.7%) | 10/12 (83.3%) | 28/60 (46.7%) | 2 |
| PI-CAI ZGT | Bosma22b reference | 5/15 (33.3%) | 0/1 (0.0%) | 4/14 (28.6%) | 1 |
| PI-CAI ZGT | Clinical ellipsoid | 6/15 (40.0%) | 0/1 (0.0%) | 5/14 (35.7%) | 1 |
| PI-CAI ZGT | Multicenter | 5/15 (33.3%) | 0/1 (0.0%) | 4/14 (28.6%) | 1 |
| PI-CAI ZGT | RUMC | 6/15 (40.0%) | 0/1 (0.0%) | 5/14 (35.7%) | 1 |

Biopsy avoidance includes outcome-positive patients below the threshold. These retrospective simulated decisions do not establish optimal management; small site-specific outcome denominators limit interpretation.

![Simulated biopsy decisions versus case-level ISUP outcomes in PI-CAI PCNN (n = 42).](results/picai_test_clinical_volume/PCNN/psa_density_biopsy_confusion.png)

**Figure D6a. Simulated biopsy decisions versus case-level ISUP outcomes in PI-CAI PCNN (n = 42).** Panels show Bosma22b reference-volume PSAD, clinical ellipsoid, Multicenter, and RUMC estimates. Rows indicate no csPCa (ISUP 0–1) or csPCa (ISUP 2–5); columns indicate simulated biopsy avoidance (PSAD <0.15 ng/mL²) or recommendation (PSAD ≥0.15 ng/mL²). Cells show counts and percentages within outcome rows; all methods use the same patient PSA.

![Simulated biopsy decisions versus case-level ISUP outcomes in PI-CAI RUMC (n = 72).](results/picai_test_clinical_volume/RUMC/psa_density_biopsy_confusion.png)

**Figure D6b. Simulated biopsy decisions versus case-level ISUP outcomes in PI-CAI RUMC (n = 72).** Panels show Bosma22b reference-volume PSAD, clinical ellipsoid, Multicenter, and RUMC estimates. Rows indicate no csPCa (ISUP 0–1) or csPCa (ISUP 2–5); columns indicate simulated biopsy avoidance (PSAD <0.15 ng/mL²) or recommendation (PSAD ≥0.15 ng/mL²). Cells show counts and percentages within outcome rows; all methods use the same patient PSA.

![Simulated biopsy decisions versus case-level ISUP outcomes in PI-CAI ZGT (n = 15).](results/picai_test_clinical_volume/ZGT/psa_density_biopsy_confusion.png)

**Figure D6c. Simulated biopsy decisions versus case-level ISUP outcomes in PI-CAI ZGT (n = 15).** Panels show Bosma22b reference-volume PSAD, clinical ellipsoid, Multicenter, and RUMC estimates. Rows indicate no csPCa (ISUP 0–1) or csPCa (ISUP 2–5); columns indicate simulated biopsy avoidance (PSAD <0.15 ng/mL²) or recommendation (PSAD ≥0.15 ng/mL²). Cells show counts and percentages within outcome rows; all methods use the same patient PSA.

## Appendix E. Expanded Toronto JDMI Full-Cohort Biopsy Simulation

The manual-reference cohort provided only 23 evaluable PI-RADS 3 patients, motivating an exploratory extension to the 4,105-patient Toronto JDMI cohort using the same-institution nnUNet702 model as an automated surrogate reference. This surrogate-reference analysis is presented separately from the manual-planimetry comparisons and the JDMI Model results in Tables 1–3. Of 763 patients with reported PI-RADS 3, 592 (77.6%) had unknown CSCANCER outcomes and were excluded from biopsy evaluation, leaving 171 patients with known outcomes: 24 recorded positives and 147 negatives. All 23 evaluable patients from the manual-reference subgroup remained included. Volume agreement and four-stratum PSAD categorization in the expanded cohort are presented in Appendix B (Figures B1–B2).

At PSAD ≥0.15 ng/mL², both the Multicenter and RUMC Models would avoid biopsy in 109/171 patients (63.7%), compared with 107/171 (62.6%) for clinical manual ellipsoid estimates and 111/171 (64.9%) for the same-institution surrogate (Table E1; Figure E1). Nine recorded positive cases fell below the threshold with ellipsoid estimates, versus ten with the surrogate and each model. The Multicenter and RUMC Models had identical aggregate confusion counts, although their biopsy recommendations differed for four patients. These results describe retrospective simulated decisions among patients with recorded outcomes; missing outcomes and the use of an automated surrogate reference limit their interpretation.

**Table E1. Simulated biopsy avoidance, sensitivity, and specificity at PSAD ≥0.15 ng/mL² in the expanded Toronto JDMI PI-RADS 3 subgroup (n = 171).**

| Volume method | Simulated biopsies avoided, n/N (%) | Sensitivity, n/N (%) | Specificity, n/N (%) |
| --- | ---: | ---: | ---: |
| Same-institution surrogate (nnUNet702) | 111/171 (64.9%) | 14/24 (58.3%) | 101/147 (68.7%) |
| Clinical manual ellipsoid | 107/171 (62.6%) | 15/24 (62.5%) | 98/147 (66.7%) |
| Multicenter Model | 109/171 (63.7%) | 14/24 (58.3%) | 99/147 (67.3%) |
| RUMC Model | 109/171 (63.7%) | 14/24 (58.3%) | 99/147 (67.3%) |

Biopsy avoidance uses all 171 eligible patients as its denominator and includes recorded positive cases below the threshold. Sensitivity uses the 24 CSCANCER-positive patients and specificity the 147 CSCANCER-negative patients.

![Simulated biopsy decisions versus recorded CSCANCER outcomes in 171 expanded-cohort Toronto JDMI PI-RADS 3 patients.](results/toronto_jdmi_4111_clinical_volume/psa_density_biopsy_confusion.png)

**Figure E1. Biopsy 2×2 confusion matrices in the expanded Toronto JDMI cohort with reported PI-RADS 3 and known CSCANCER outcomes (n = 171).** Panels show PSAD derived from the same-institution nnUNet702 automated surrogate reference, clinical manual ellipsoid estimates, the Multicenter Model, and the RUMC Model. Rows indicate recorded CSCANCER outcomes (N negative, Y positive); columns indicate simulated avoidance of biopsy (unrounded PSAD <0.15 ng/mL²) or recommendation of biopsy (PSAD ≥0.15 ng/mL²). All methods use the same patient-level PSA. Cells show patient counts and percentages within each outcome row. The matrices describe a retrospective threshold simulation rather than observed biopsy management.

