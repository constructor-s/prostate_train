# Methods

## Study datasets and reference standards

We compared automated prostate whole-gland segmentation and volumetry with clinical ellipsoid volume estimates and cohort-specific reference measurements. Eight testing cohorts were evaluated: Multicenter Data testing examinations acquired with an external coil (22 examinations), Multicenter Data endorectal-coil examinations (13 examinations), ProstateX RUMC (157 patients), Toronto JDMI (383 patients), PI-CAI RUMC excluding ProstateX RUMC (275 patients), Toronto JDMI Additional (3,722 patients), PI-CAI ZGT (168 patients), and PI-CAI PCNN with wide-field imaging (315 patients). The original 35-examination Multicenter Data test set was separated into external-coil and endorectal-coil subgroups after review of segmentation failures; this was an exploratory, post hoc subgroup analysis.

Manual whole-gland masks supplied the reference for the Multicenter Data and ProstateX cohorts. Toronto JDMI used the median of manual-reader planimetry volumes; spatial reference masks were unavailable for its segmentation evaluation. PI-CAI used Bosma22b automated whole-gland masks, and Toronto JDMI Additional used an internal nnU-Net model, with whole gland defined as the union of labels 1, 2, and 3. Comparisons with these automated surrogate references assess agreement with their labeling methods rather than accuracy against independent manual truth. Clinical ellipsoid volumes were taken from the saved clinical measurements, including reported volumes for Toronto JDMI.

The ProstateX cohort required a unique PI-CAI examination linkage, a finite positive clinical volume, and a manual whole-gland mask. Of 190 otherwise eligible examinations, 33 were excluded because the manual mask contacted a superior or inferior outer image face. PI-CAI testing excluded single-center training patients and their repeat examinations, six manually reviewed poor Bosma22b masks, and examinations with superior or inferior reference-mask boundary contact. Among eligible visits, clinically complete examinations were preferred, followed by the earliest MRI date, with original marksheet order resolving ties. Clinical completeness for PI-CAI testing required finite positive volume and PSA, valid lesion PI-RADS scores, and biopsy results; recorded PSAD was optional. Boundary contact was an operational selection criterion and did not establish anatomical truncation.

For the six clinical comparison cohorts, each patient contributed one examination and cohort membership was made disjoint. The PI-CAI RUMC cohort retained 275 of 391 patients after removing 116 represented in ProstateX RUMC, using PI-CAI patient identifiers. Toronto JDMI Additional retained 3,722 of an archived 4,105-patient cohort after removing the 383 Toronto JDMI manual-reference patients by medical record identifier. The archived Toronto measurements and accession exclusions were preserved rather than substituting the current exclusion file, whose membership differs. Patient identifiers were compared within the Dutch and Toronto identifier systems. The 236 single-center training patients had no patient-ID overlap with either evaluated PI-CAI RUMC or ZGT cohort.

## Model development

The multicenter model was trained on 239 T2-weighted MRI examinations: 119 from Prostate158, 67 from PROMISE12, 27 from Prostate Diagnosis, and 26 from QIN Prostate Repeatability. These are examination counts rather than asserted counts of unique patients. The single-center model was trained on 236 examinations from 236 RUMC patients in PI-CAI. Patients with any clinically complete RUMC examination or any ProstateX mapping were excluded, and the earliest examination was selected for each remaining patient. Training-selection completeness required finite positive volume and PSA, finite nonnegative recorded PSAD, valid PI-RADS scores, and biopsy results on the same examination. Single-center training masks were Bosma22b automated whole-gland annotations. Their shared provenance with the PI-CAI evaluation references was considered when interpreting performance.

Both models used nnU-Net's three-dimensional full-resolution residual-encoder U-Net with one T2-weighted input channel and whole-gland segmentation as the target. Model-specific plans used z-score intensity normalization and resampling to approximately 3.00 × 0.402 × 0.402 mm for the multicenter model and 3.00 × 0.300 × 0.300 mm for the single-center model, in the plans' array-axis order. Patch sizes in the same order were 28 × 384 × 384 and 16 × 512 × 512 voxels, respectively; both used a batch size of two. The multicenter and single-center networks had seven and eight resolution stages, respectively, with instance normalization and leaky rectified linear unit activations.

Five folds were trained for each model, with 100 epochs per fold, 250 training iterations and 50 validation iterations per epoch. Saved trainer records specify stochastic gradient descent with initial learning rate 0.01, Nesterov momentum 0.99, weight decay 3 × 10⁻⁵, and polynomial learning-rate decay. Training used combined soft Dice and cross-entropy loss with deep supervision. The installed nnU-Net training implementation includes rotation, scaling, mirroring, Gaussian noise and blur, brightness and contrast changes, gamma transformations, and simulated low-resolution augmentation. Saved training records report PyTorch 2.12.0+cu130. The environment inspected for this analysis contains nnU-Net 2.7.0 and dynamic-network-architectures 0.4.4; these installed versions are not independently timestamped records of the original training environment.

## Inference and segmentation processing

Predictions were generated using the exported models' inference entrypoints and ensembles of folds 0–4. Recorded inference settings used a sliding-window tile step of 0.8 and disabled mirroring at inference. Whole-gland foreground was label 1 for both evaluated models. Largest-connected-component filtering for the multicenter model was selected during cross-validation. Filtering was not selected during single-center cross-validation; following review of failures, the same operation was imposed post hoc on its saved Multicenter Data, ProstateX, PI-CAI, and Toronto predictions. Original predictions and processing audits were retained.

Spatial overlap was evaluated only after checking mask geometry and millimetre units; PI-CAI mask comparisons required matching grids with maximum voxel-corner displacement no greater than 0.5 mm. Segmentation-derived volume was calculated as foreground voxel count multiplied by voxel volume in mm³ and divided by 1,000 to obtain mL. Spatial comparison used masks in the same geometry. Scalar Toronto reference volumes were used only for volume comparisons.

## Segmentation and volume evaluation

Each examination was weighted equally within its cohort. Whole-gland overlap was measured using the Dice similarity coefficient, DSC = 2|P ∩ R|/(|P| + |R|), where P and R denote predicted and reference foreground masks. Segmentation failures, including zero-overlap cases, remained in the evaluated cohort. DSC was not calculated for Toronto cohorts with scalar reference volumes.

For measured volume V and positive reference volume R, volume errors were summarized as mean absolute error, MAE = mean(|V − R|), and root mean squared error, RMSE = √mean[(V − R)²], in mL. Percentage errors were MAPE = 100 × mean(|V − R|/R) and RMSPE = 100 × √mean[((V − R)/R)²]. Higher DSC and lower volume errors indicate better agreement with the reference. Manual-reference agreement plots used signed differences V − R and relative differences 100 × (V − R)/R. Descriptive 95% limits of agreement were the mean signed difference ±1.96 sample standard deviations; these limits are not confidence intervals.

## Disagreement screening and downstream analyses

Following comparison of both models, subsequent disagreement analyses used the multicenter model as a representative automated approach because of its broader generalization across the evaluated imaging settings. The analysis examined whether an automated measurement available to a radiologist could identify discordant ellipsoid estimates and potentially support volume assessment and downstream decisions; no reader intervention or deployed correction workflow was tested.

Screening used clinical ellipsoid volume M and automated volume A without consulting each patient's reference volume. Signed disagreement was D = M − A in mL, and relative disagreement was Q = 100 × (M − A)/[(M + A)/2] in percent. The pair-mean denominator differs from the reference-volume denominator used for percentage errors. ProstateX RUMC supplied the calibration distribution, from which mean ±1.96 sample standard deviations gave fixed, rounded limits of −14.28 to +25.74 mL and −22.70 to +39.60%. A case was selected only if strictly outside both intervals, irrespective of tail; equality at a boundary was retained. The rounded intervals were applied unchanged to all six clinical cohorts, without local recentering. Cohort-specific limits were descriptive. These empirically calibrated criteria were exploratory review thresholds, not established clinical tolerances, and ProstateX was not an independent validation of their transportability.

Within selected cases, automated and ellipsoid measurements were compared by absolute reference-volume error and by MAE, RMSE, MAPE, and RMSPE. Automated wins, losses, and ties indicated smaller, larger, or exactly equal absolute error relative to ellipsoid measurements. PSAD was calculated using the same patient-level PSA for each volume method, as PSA in ng/mL divided by volume in mL, yielding ng/mL². Finite positive PSA and finite PSAD for all three methods were required for category comparisons. Unrounded PSAD values were assigned to categories <0.10, 0.10–<0.15, 0.15–<0.20, and ≥0.20 ng/mL², assigning cutoff equality to the higher category. Closeness was absolute ordinal distance from the reference-volume-derived category. Equal distance could include different categories equally far from the reference; closer categories were not necessarily identical to the reference category.

Among selected cases with maximum recorded PI-RADS 3, biopsy recommendations were simulated using unrounded PSAD ≥0.15 ng/mL² for the reference, ellipsoid, and automated volumes. Changes from ellipsoid to automated recommendations were classified as moving into or away from agreement with the reference recommendation. Recorded outcome-positive status was ISUP grade group ≥2 for PI-CAI and CSCANCER Y for Toronto; these outcome definitions were retained separately. Unknown outcomes were retained for recommendation-agreement comparisons but excluded from outcome-based metrics. Sensitivity was the proportion of outcome-positive cases recommended for biopsy, and specificity the proportion of outcome-negative cases assigned to biopsy avoidance. Overall biopsy avoidance included both true negatives and false negatives. Outcome analyses were descriptive because positive cases were sparse and outcomes were missing for some patients.

## Statistical analysis and reproducibility

Two-sided exact binomial tests compared the proportion of automated wins among non-ties with 50%, separately for absolute volume error and PSAD category distance. Ties were retained in descriptive denominators but excluded from the tests. Tables 4 and 5 report Holm-adjusted p-values across six cohorts, with separate correction families for volume and PSAD comparisons. Undefined tests were assigned p = 1 for correction and retained as undefined in reporting. Pooled comparisons summed wins and losses first across the two manual-reference cohorts and then across all six clinical cohorts; their p-values were unadjusted. These pooled comparisons assume independent patients and may be dominated by the larger cohorts. Predictions, calibration thresholds, and selected cases were treated as fixed; tests did not incorporate model-training variability or uncertainty from threshold estimation and case selection.

The analysis was assembled from saved measurements and segmentation evaluations using [the synthesis notebook](notebooks/paper_new_results.py). Membership audits, patient exclusions, reference provenance, input SHA-256 hashes, and numerical outputs were retained under results/paper_new_results/. The synthesis checked cohort membership and reproduced calculations without new model training, inference, or mask processing. This retrospective analysis of recorded measurements and simulated recommendations does not establish prospective effects on radiologist performance, cancer detection, or clinical management.

# Results

## Segmentation performance and volume accuracy

Eight testing cohorts were evaluated using the multicenter model, the single-center model trained only on RUMC data, and clinical manual ellipsoid estimates (Tables 1–3). Whole-gland volume errors were calculated relative to manual expert whole-gland segmentations where available (Testing fold of Multicenter Data, ProstateX, and Toronto JDMI), and a surrogate high-quality automated reference trained within the same institutions otherwise (Bosma22b for PI-CAI and internal nnU-Net model output for Toronto JDMI). Note that the endorectal coil data and wide-field PI-CAI PCNN data were particularly challenging as the former was not represented in the single-center model and the latter was not encountered in either model's training data, so they serve as a stress test of model generalization.

On the Multicenter Data testing fold, mean whole-gland DSC was 0.905 for the multicenter model and 0.904 for the single-center model. On ProstateX RUMC, the corresponding values were 0.906 and 0.916, respectively. The single-center model, having never seen endorectal coil or wide-field images, performed poorly in those cohorts (DSC 0.593 and 0.558, respectively). The multicenter model was able to generalize better to the wide-field PCNN data that it also did not encounter in training, achieving DSC 0.882. Upon inspection, the successes/failures were binary rather than due to poor overlap. On the endorectal coil data, the single-center model often failed to segment the distorted prostate and opt to segment the more circular endorectal coil. On the PCNN data, both models sometimes failed catastrophically and segmented the femoral heads instead of the prostate.

**Table 1. Mean whole-gland Dice similarity coefficient (DSC), testing cohorts only.**

<!-- TABLE_1_START -->

| Dataset | Multicenter | Single-center |
| --- | --- | --- |
| Multicenter Data Testing (n = 22) | 0.905 | 0.904 | — |
| ProstateX RUMC (n = 157) | 0.906 | 0.916 | — |
| Toronto JDMI (n = 383) | — | — | — |
| PI-CAI RUMC excluding ProstateX RUMC (n = 275) | 0.919 | 0.961 | — |
| Toronto JDMI Additional (n = 3,722) | — | — | — |
| PI-CAI ZGT (n = 168) | 0.929 | 0.953 | — |
| Multicenter Data Testing (ERC) (n = 13) | 0.911 | 0.593 | — |
| PI-CAI PCNN (wide field) (n = 315) | 0.882 | 0.558 | — |

<!-- TABLE_1_END -->

Both models had lower MAE, RMSE, MAPE, and RMSPE than clinical ellipsoid estimates in all datasets other than the PCNN stress test. Multicenter model MAE was 5.09 mL in ProstateX and 4.87 mL in Toronto JDMI, compared with 9.20 and 7.18 mL for ellipsoid estimates. Interestingly, the single-center model trained on RUMC data had stringly good performance in PI-CAI RUMC excluding ProstateX RUMC and PI-CAI ZGT, but not ProstateX RUMC. This suggest bias in label provenance, as it was trained on the Bosma22b automated whole-gland labels and used similar nnU-Net architecture. Automated labels are less noisy than manual segmentation and may have overfitted to idosyncratic features. In PCNN, the multicenter model still reduced all four errors overall relative to ellipsoid estimates despite failing to segment some cases, whereas the single-center model had higher errors, again showing its poorer generalization abilities. Figures 1–2 show whole-gland volume agreement against the independent manual references in ProstateX RUMC and Toronto JDMI.

**Table 2. Whole-gland volume error: MAE; RMSE (mL).**

<!-- TABLE_2_START -->

| Dataset | Clinical ellipsoid | Multicenter | Single-center |
| --- | --- | --- | --- |
| Multicenter Data Testing (n = 22) | — | 3.42; 4.07 | 2.72; 3.32 |
| ProstateX RUMC (n = 157) | 9.20; 12.06 | 5.09; 6.47 | 4.57; 5.85 |
| Toronto JDMI (n = 383) | 7.18; 9.42 | 4.87; 6.76 | 5.28; 6.63 |
| PI-CAI RUMC excluding ProstateX RUMC (n = 275) | 12.87; 16.32 | 4.25; 5.26 | 1.42; 2.20 |
| Toronto JDMI Additional (n = 3,722) | 8.12; 11.70 | 4.03; 5.74 | 4.71; 6.49 |
| PI-CAI ZGT (n = 168) | 6.62; 8.92 | 2.68; 3.86 | 1.81; 2.43 |
| Multicenter Data Testing (ERC) (n = 13) | — | 2.43; 3.17 | 6.33; 8.04 |
| PI-CAI PCNN (wide field) (n = 315) | 11.69; 16.78 | 6.35; 8.48 | 12.68; 20.65 |

<!-- TABLE_2_END -->

**Table 3. Whole-gland percentage volume error: MAPE; RMSPE (%).** Percentage errors use the positive reference volume as their denominator.

<!-- TABLE_3_START -->

| Dataset | Clinical ellipsoid | Multicenter | Single-center |
| --- | --- | --- | --- |
| Multicenter Data Testing (n = 22) | — | 8.52; 10.13 | 6.47; 7.87 |
| ProstateX RUMC (n = 157) | 16.36; 21.17 | 8.85; 10.93 | 8.10; 10.47 |
| Toronto JDMI (n = 383) | 13.89; 18.07 | 9.01; 11.35 | 10.03; 12.36 |
| PI-CAI RUMC excluding ProstateX RUMC (n = 275) | 28.14; 33.98 | 9.65; 12.18 | 3.20; 4.90 |
| Toronto JDMI Additional (n = 3,722) | 13.05; 18.14 | 6.39; 9.00 | 7.76; 12.82 |
| PI-CAI ZGT (n = 168) | 14.35; 19.14 | 5.87; 7.90 | 4.30; 6.56 |
| Multicenter Data Testing (ERC) (n = 13) | — | 6.33; 8.69 | 15.78; 19.50 |
| PI-CAI PCNN (wide field) (n = 315) | 22.23; 29.57 | 13.09; 18.82 | 35.83; 65.90 |

<!-- TABLE_3_END -->

![Whole-gland volume agreement against manual whole-gland segmentation planimetry for 157 ProstateX RUMC patients.](results/paper_new_results/prostatex_comparator_agreement.png)

**Figure 1. Whole-gland volume agreement in ProstateX RUMC (n = 157).** Columns show clinical manual ellipsoid estimates (“Manual ellipsoid”), the multicenter model (“Multicenter”), and the single-center model (“Single-center”). The top row plots measured against reference volume from manual whole-gland segmentation planimetry, with logarithmic axes and a dashed identity line. The middle and bottom rows show signed differences (measured − reference, mL) and relative differences (100 × (measured − reference)/reference, %) against reference volume on a logarithmic x-axis. Solid horizontal lines show mean signed differences and dashed lines show descriptive 95% limits of agreement (mean ±1.96 sample SD). Histograms at right show counts of differences; colors, markers, and line styles distinguish methods. Orange points with white outlines identify the same 2 patients across all scatter panels, selected strictly outside both fixed common screening intervals (−14.28 to +25.74 mL and −22.70 to +39.60%); screening percentages use the clinical–multicenter model pair mean. These screening criteria differ from the method-versus-reference limits shown horizontally. All statistics and histogram bins use the full cohort. Nonzero counts outside displayed limits are annotated; the percentage display range is ±90%.

![Whole-gland volume agreement against median manual-reader planimetry for 383 Toronto JDMI patients.](results/paper_new_results/toronto_manual_comparator_agreement.png)

**Figure 2. Whole-gland volume agreement in Toronto JDMI (n = 383).** Columns show clinical manual ellipsoid estimates (“Manual ellipsoid”), the multicenter model (“Multicenter”), and the single-center model (“Single-center”). The top row plots measured against reference volume from median manual-reader planimetry, with logarithmic axes and a dashed identity line. The middle and bottom rows show signed differences (measured − reference, mL) and relative differences (100 × (measured − reference)/reference, %) against reference volume on a logarithmic x-axis. Solid horizontal lines show mean signed differences and dashed lines show descriptive 95% limits of agreement (mean ±1.96 sample SD). Histograms at right show counts of differences; colors, markers, and line styles distinguish methods. Orange points with white outlines identify the same 10 patients across all scatter panels, selected strictly outside both fixed common screening intervals (−14.28 to +25.74 mL and −22.70 to +39.60%); screening percentages use the clinical–multicenter model pair mean. These screening criteria differ from the method-versus-reference limits shown horizontally. All statistics and histogram bins use the full cohort. Nonzero counts outside displayed limits are annotated; the percentage display range is ±100%.

## When volumes between automated model and ellipsoid volumes, which is closer to the reference?

Here we examine whether, when automated and ellipsoid volumes disagree, an automated measurement available to the radiologist could help identify measurement errors and improve volume assessment and downstream clinical decisions. Given its consistent performance and broader generalization, subsequent analyses use the multicenter model to represent automated volumetry.

Figure 3 shows agreement between automated multicenter model volumes and clinical manual ellipsoid measurements in the six cohorts. Note that different from Figures 1 and 2 above, these illustrate agreement between two estimates, rather than either method's error against the reference.

We used ProstateX RUMC as the calibration cohort to establish a common disagreement criteria because we were able to confidently assess the performance of automated segmentation performance against manual reference in the previous section, and a common criteria allow more consistent interpretation and pooling across cohorts. Its manual vs. automated 95% limits of agreement were: −14.28 to +25.74 mL for absolute-scale differences and −22.70 to +39.60% for relative differences. Cases were defined as outliers using a conservative criteria requiring being strictly outside both intervals, such that there is no bias against very small (which would result in large relative error) and very large (which would result in large absolute error) prostate volumes. The other cohorts' agreement limits were broadly comparable (except for the wide-field PCNN cohort which illustrates a stress test case study), supporting the use of this common screen across cohorts.

![Absolute and relative Bland–Altman disagreement between automated multicenter model and clinical manual ellipsoid volumes.](results/paper_new_results/bland_altman_clinical_nnunet274.png)

**Figure 3. Automated–clinical manual volume disagreement and common screening criteria.** Each row represents one cohort. Left panels show clinical ellipsoid minus multicenter model volume in mL; right panels show 100 × difference / pair mean in percent. Pair mean volume is displayed on logarithmic x-axes with ticks at 10, 30, 100, and 300 mL; calculations use the original measurements. Gray solid lines indicate cohort means and gray dotted lines cohort-specific 95% limits of agreement. Black dashed lines show the fixed common criteria, −14.28 to +25.74 mL and −22.70 to +39.60%. Orange points with white outlines exceed both intervals and identify the same selected cases in both columns; muted gray points are all other cases. Each legend reports the full-cohort selected count. Display cropping changes neither selection nor calculations; nonzero counts outside the displayed axes are annotated. Titles give full cohort sizes.

**Table 4. Reference volume errors among cases exceeding both common disagreement criteria: manual → automated.** Flagged n/N (%) uses each complete cohort as denominator. MAE/RMSE are in mL and MAPE/RMSPE in percent. Improved/all reports lower absolute reference error with the multicenter model. Adjusted p-values use a six-cohort Holm family.

<!-- TABLE_4_START -->

| Dataset | Flagged n/N (%) | MAE (mL) | RMSE (mL) | MAPE (%) | RMSPE (%) | Improved / all | Adjusted p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ProstateX RUMC | 2/157 (1.3%) | 33.08 → 4.13 | 33.34 → 4.71 | 65.15 → 7.60 | 66.73 → 8.41 | 2/2 | 0.500 |
| Toronto JDMI | 10/383 (2.6%) | 21.75 → 8.61 | 23.14 → 16.87 | 35.25 → 8.15 | 38.77 → 10.79 | 8/10 | 0.219 |
| PI-CAI RUMC excluding ProstateX RUMC | 11/275 (4.0%) | 33.38 → 5.85 | 36.32 → 6.23 | 65.77 → 11.92 | 69.93 → 14.51 | 11/11 | 0.004 |
| Toronto JDMI Additional | 143/3,722 (3.8%) | 28.13 → 5.37 | 32.12 → 7.87 | 41.14 → 9.56 | 50.29 → 20.33 | 135/143 | <0.001 |
| PI-CAI ZGT | 7/168 (4.2%) | 21.70 → 1.43 | 23.71 → 1.90 | 33.24 → 2.25 | 37.03 → 3.01 | 7/7 | 0.047 |
| PI-CAI PCNN (wide field) | 60/315 (19.0%) | 19.15 → 10.07 | 24.39 → 13.26 | 32.62 → 19.25 | 40.29 → 30.09 | 47/60 | <0.001 |

<!-- TABLE_4_END -->

Table 4 compares automated and clinical ellipsoid volume errors in cases exceeding both disagreement criteria. Among the first two cohorts with manual segmentation references, automated volumes were closer to the reference in 10/12 cases (83.3%; p = 0.04); overall across all six cohorts, automated volumes were closer to the reference in 210/233 selected cases (90.1%; p < 0.001). All error measures (MAE, RMSE, MAPE, RMSPE) decreased in each cohort.

## Impact on PSAD category and simulated biopsy recommendation


PSAD categories calculated using automated volumetry were closer to the reference category in 73/143 Toronto JDMI Additional outliers, equally distant in 63, and farther in seven (Table 5). Across the first two reference cohorts with manual segmentation references, automated PSAD categories were closer in eight cases, equally distant in three, and farther in none (p = 0.008). Across all six cohorts, automated categories were closer in 116 cases, equally distant in 95, and farther in 21. (p < 0.001).

**Table 5. Ordinal PSAD category closeness among common-criteria outliers.** Equal-distance counts are retained because categorical ties are meaningful. Adjusted p-values test automated versus manual wins among non-ties against 50%, with Holm correction across six cohorts. Undefined tests, if encountered, conservatively occupy a family slot as p = 1 when adjusting the other cohorts and remain undefined in the output.

<!-- TABLE_5_START -->

| Dataset | Automated closer | Equal distance | Manual closer | Adjusted p |
| --- | --- | --- | --- | --- |
| ProstateX RUMC (n = 1, another missing PSA value) | 1 | 0 | 0 | 1.000 |
| Toronto JDMI (n = 10) | 7 | 3 | 0 | 0.078 |
| PI-CAI RUMC excluding ProstateX RUMC (n = 11) | 6 | 4 | 1 | 0.375 |
| Toronto JDMI Additional (n = 143) | 73 | 63 | 7 | <0.001 |
| PI-CAI ZGT (n = 7) | 3 | 4 | 0 | 0.500 |
| PI-CAI PCNN (wide field) (n = 60) | 26 | 21 | 13 | 0.213 |

<!-- TABLE_5_END -->

Forty-seven out of 233 disagreement cases had PI-RADS 3: none in ProstateX, three in Toronto JDMI, one in PI-CAI RUMC excluding ProstateX RUMC, 30 in Toronto JDMI Additional, one in ZGT, and 12 in PCNN. The individual cases are shown in Table 6. Twenty cases had unknown outcomes, comprising one Toronto JDMI and 19 Toronto Additional cases. 


In Toronto JDMI, biopsy recommendation did not change using any of reference, manual ellipsoid, or automated volumetry, one recommendation and negative biopsy, one avoidance and negative biopsy, one recommendation and unknown outcome.

In PI-CAI RUMC, the identified patient had ISUP grade group 2. Reference and automated PSAD exceeded the biopsy threshold and would correctly recommend biopsy, whereas manual PSAD would not, i.e. the biopsy recommendation would differ.

In Toronto JDMI Additional, eight of 30 recommendations changed between ellipsoid and automated: six from recommend to avoid and two from avoid to recommend. Outcome was available for 11 cases. In the single outcome-positive case, only manual ellipsoid measurements would recommend biopsy, i.e. both reference and automated volumetry would incorrectly avoid biopsy. Among the ten outcome-negative cases, reference, manual ellipsoid, and automated measurements would avoid biopsy in seven, five, and seven, respectively (specificity: 70%, 50%, and 70%). All three methods had concordant recommendations and avoidance in two and five cases, respectively. In the remaining three outcome-negative cases, the recommendataions by reference, manual, and automated measurements were: (recommend, recommend, avoid), (avoid, recommend, recommend), and (avoid, recommend, avoid).

The one ZGT case had ISUP grade group 0 and would recommend biopsy by all three measurements. PCNN, which was challenging for the automated model, four of 12 recommendations changed between ellipsoid and automated, all from recommend to avoid. One patient had ISUP grade group 2, and 11 had grade group 0 or 1. In the single outcome-positive case (ISUP grade group 2), all three measurements would recommend biopsy. Among the 11 outcome-negative cases (ISUP grade group 0 or 1), reference, manual ellipsoid, and automated measurements would avoid biopsy in four, three, and seven, respectively (specificity: 36.4%, 27.3%, and 63.6%). All three methods agreed on recommending biopsy in four outcome-negative cases and avoiding biopsy in two. In the remaining five outcome-negative cases, recommendations by reference, manual, and automated measurements were: (recommend, avoid, avoid) in one, (recommend, recommend, avoid) in two, and (avoid, recommend, avoid) in two.

<!-- 
Unverified table:
| Dataset | Cases | Changed | Toward reference | Away from reference |
|---|---:|---:|---:|---:|
| ProstateX RUMC | 0 | 0 | 0 | 0 |
| Toronto JDMI | 3 | 0 | 0 | 0 |
| PI-CAI RUMC excluding ProstateX | 1 | 1 | 1 | 0 |
| Toronto JDMI Additional | 30 | 8 | 6 | 2 |
| PI-CAI ZGT | 1 | 0 | 0 | 0 |
| PI-CAI PCNN | 12 | 4 | 2 | 2 |
| **Total** | **47** | **13** | **9** | **4** |

| Dataset | Specificity improved¹ | Specificity worsened² | Sensitivity improved³ | Sensitivity worsened⁴ | Changed, outcome unknown |
|---|---:|---:|---:|---:|---:|
| Toronto JDMI | 0 | 0 | 0 | 0 | 0 |
| PI-CAI RUMC excluding ProstateX | 0 | 0 | 1 | 0 | 0 |
| Toronto JDMI Additional | 2 | 0 | 0 | 1 | 5 |
| PI-CAI ZGT | 0 | 0 | 0 | 0 | 0 |
| PI-CAI PCNN | 4 | 0 | 0 | 0 | 0 |
| **Total** | **6** | **0** | **1** | **1** | **5** |

¹ False positive → true negative. ² True negative → false positive.
³ False negative → true positive. ⁴ True positive → false negative.
 -->

Overall, 13 of 47 (27.7%) PI-RADS 3 cases had changed biopsy recommendations between manual ellipsoid and automated volumetry, with nine (69.2%) moving toward the reference recommendation and four (30.8%) moving away. Among the changed cases, six had known negative outcomes and automated volumetry correctedly changed to avoiding biopsy in all six; two had known positive outcomes and automated volumetry correctly changed to recommendation in one and incorrectly changed to avoidance in the other. Five cases had unknown outcomes.

**Table 6. Every PI-RADS 3 common-criteria outlier, including unknown outcomes.** Method cells give volume in mL; PSAD in ng/mL²; and the simulated biopsy recommendation. Displayed PSAD is rounded to three decimals; decisions use unrounded values. Reference is Bosma22b for PI-CAI, median manual-reader planimetry for Toronto JDMI, and nnUNet702 for Toronto JDMI Additional. Rows are sorted globally by outcome (ISUP ≥2, CSCANCER Y, ISUP 0–1, CSCANCER N, then Unknown), then reference, manual, and automated biopsy recommendation (Recommend before Avoid, missing last), then the stated cohort order and case identifier. Identifiers are retained for local audit.

<!-- TABLE_6_START -->

| Dataset | Case | Reference: mL; PSAD; biopsy | Manual: mL; PSAD; biopsy | Automated: mL; PSAD; biopsy | Recorded outcome |
| --- | --- | --- | --- | --- | --- |
| PI-CAI PCNN (wide field) | picai_10304_1000310 | 51.08; 0.190; Recommend | 37.00; 0.262; Recommend | 55.16; 0.176; Recommend | ISUP 2 |
| PI-CAI RUMC excluding ProstateX RUMC | picai_10658_1000674 | 41.53; 0.185; Recommend | 73.00; 0.105; Avoid | 44.06; 0.175; Recommend | ISUP 2 |
| Toronto JDMI Additional | AIPR_20011751_20210312_333abfe653dd472 | 45.30; 0.130; Avoid | 29.60; 0.199; Recommend | 44.85; 0.132; Avoid | CSCANCER Y |
| PI-CAI ZGT | picai_10179_1000182 | 94.95; 0.484; Recommend | 73.00; 0.630; Recommend | 94.77; 0.485; Recommend | ISUP 0 |
| PI-CAI PCNN (wide field) | picai_10015_1000015 | 61.24; 0.376; Recommend | 50.00; 0.460; Recommend | 67.38; 0.341; Recommend | ISUP 0 |
| PI-CAI PCNN (wide field) | picai_10692_1000708 | 113.51; 0.247; Recommend | 97.00; 0.289; Recommend | 124.09; 0.226; Recommend | ISUP 0 |
| PI-CAI PCNN (wide field) | picai_10714_1000730 | 70.16; 0.185; Recommend | 56.00; 0.232; Recommend | 71.27; 0.182; Recommend | ISUP 0 |
| PI-CAI PCNN (wide field) | picai_10967_1000986 | 37.77; 0.344; Recommend | 25.00; 0.520; Recommend | 50.88; 0.256; Recommend | ISUP 0 |
| PI-CAI PCNN (wide field) | picai_10678_1000694 | 56.77; 0.157; Recommend | 45.00; 0.198; Recommend | 62.00; 0.144; Avoid | ISUP 1 |
| PI-CAI PCNN (wide field) | picai_11089_1001112 | 53.02; 0.162; Recommend | 44.00; 0.195; Recommend | 58.38; 0.147; Avoid | ISUP 0 |
| PI-CAI PCNN (wide field) | picai_11018_1001038 | 66.00; 0.167; Recommend | 128.00; 0.086; Avoid | 79.54; 0.138; Avoid | ISUP 0 |
| PI-CAI PCNN (wide field) | picai_10147_1000149 | 52.77; 0.121; Avoid | 40.00; 0.160; Recommend | 55.22; 0.116; Avoid | ISUP 1 |
| PI-CAI PCNN (wide field) | picai_10387_1000393 | 42.59; 0.129; Avoid | 30.00; 0.183; Recommend | 46.03; 0.119; Avoid | ISUP 1 |
| PI-CAI PCNN (wide field) | picai_10276_1000281 | 92.50; 0.084; Avoid | 77.00; 0.101; Avoid | 102.20; 0.076; Avoid | ISUP 1 |
| PI-CAI PCNN (wide field) | picai_10509_1000518 | 73.30; 0.115; Avoid | 59.00; 0.142; Avoid | 76.91; 0.109; Avoid | ISUP 1 |
| Toronto JDMI | AIPR_20002786_20180330_2ec95099fbd2198 | 51.63; 0.159; Recommend | 30.00; 0.273; Recommend | 47.63; 0.172; Recommend | CSCANCER N |
| Toronto JDMI Additional | AIPR_20010619_20190131_2ecfd97f7b94dae | 42.47; 0.370; Recommend | 25.90; 0.607; Recommend | 40.54; 0.387; Recommend | CSCANCER N |
| Toronto JDMI Additional | AIPR_20013606_20190503_2ecfd97f7c0ea1f | 73.06; 0.168; Recommend | 47.00; 0.262; Recommend | 70.33; 0.175; Recommend | CSCANCER N |
| Toronto JDMI Additional | AIPR_20011722_20210130_333abfe653c8fee | 74.37; 0.155; Recommend | 64.00; 0.181; Recommend | 81.38; 0.142; Avoid | CSCANCER N |
| Toronto JDMI Additional | AIPR_20010980_20181206_2ecfd97f7c03398 | 49.08; 0.143; Avoid | 30.00; 0.233; Recommend | 44.53; 0.157; Recommend | CSCANCER N |
| Toronto JDMI Additional | AIPR_20008731_20210106_333abfe653db533 | 105.71; 0.132; Avoid | 62.00; 0.226; Recommend | 100.89; 0.139; Avoid | CSCANCER N |
| Toronto JDMI | AIPR_20000108_20160714_2ec95031808142e | 57.07; 0.107; Avoid | 43.00; 0.142; Avoid | 59.54; 0.102; Avoid | CSCANCER N |
| Toronto JDMI Additional | AIPR_20002447_20160610_2ec9508216c2f1d | 80.28; 0.064; Avoid | 116.00; 0.044; Avoid | 69.96; 0.073; Avoid | CSCANCER N |
| Toronto JDMI Additional | AIPR_20004279_20160326_2ec950caaac76d3 | 53.84; 0.040; Avoid | 105.00; 0.020; Avoid | 50.76; 0.042; Avoid | CSCANCER N |
| Toronto JDMI Additional | AIPR_20009071_20161209_2ec95184ab13535 | 60.53; 0.076; Avoid | 43.00; 0.107; Avoid | 57.87; 0.079; Avoid | CSCANCER N |
| Toronto JDMI Additional | AIPR_20010610_20220509_333abfe656a1fa8 | 87.02; 0.103; Avoid | 68.00; 0.132; Avoid | 86.79; 0.103; Avoid | CSCANCER N |
| Toronto JDMI Additional | AIPR_20011234_20200402_333abfe65299d3b | 136.01; 0.069; Avoid | 99.60; 0.094; Avoid | 131.56; 0.071; Avoid | CSCANCER N |
| Toronto JDMI | AIPR_20007451_20170809_2ec9515013cc7c1 | 46.28; 0.171; Recommend | 32.00; 0.247; Recommend | 47.46; 0.166; Recommend | Unknown |
| Toronto JDMI Additional | AIPR_20001488_20201231_333abfe653da97a | 88.77; 0.193; Recommend | 61.00; 0.280; Recommend | 79.11; 0.216; Recommend | Unknown |
| Toronto JDMI Additional | AIPR_20008595_20140607_2ec95179733ac8a | 168.21; 0.212; Recommend | 121.00; 0.295; Recommend | 158.26; 0.226; Recommend | Unknown |
| Toronto JDMI Additional | AIPR_20010357_20180922_2ecfd97f7b4588c | 79.34; 0.543; Recommend | 53.00; 0.813; Recommend | 72.08; 0.598; Recommend | Unknown |
| Toronto JDMI Additional | AIPR_20004741_20151226_2ec950e34dbfc35 | 66.36; 0.157; Recommend | 89.00; 0.117; Avoid | 59.27; 0.175; Recommend | Unknown |
| Toronto JDMI Additional | AIPR_20003974_20141222_2ec950c41d574a9 | 84.91; 0.141; Avoid | 46.20; 0.260; Recommend | 75.32; 0.159; Recommend | Unknown |
| Toronto JDMI Additional | AIPR_20004206_20150314_2ec950c8e2a49c8 | 87.04; 0.121; Avoid | 54.00; 0.194; Recommend | 80.99; 0.130; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20006611_20130923_2ec9512f4649001 | 85.62; 0.142; Avoid | 67.00; 0.182; Recommend | 84.99; 0.144; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20012592_20220503_333abfe655ff2af | 84.63; 0.106; Avoid | 57.00; 0.158; Recommend | 83.16; 0.108; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20003211_20170208_2ec950a51ad4110 | 39.95; 0.145; Avoid | 65.00; 0.089; Avoid | 37.98; 0.153; Recommend | Unknown |
| Toronto JDMI Additional | AIPR_20001168_20171102_2ec95056c4c4e79 | 85.24; 0.133; Avoid | 132.00; 0.086; Avoid | 79.67; 0.142; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20005698_20171023_2ec95109549dd1d | 84.98; 0.090; Avoid | 56.00; 0.136; Avoid | 78.18; 0.097; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20006438_20161106_2ec9512acd059ba | 147.04; 0.066; Avoid | 92.00; 0.105; Avoid | 137.13; 0.071; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20006512_20160816_2ec9512ca04284a | 92.10; 0.109; Avoid | 70.00; 0.143; Avoid | 91.46; 0.109; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20006709_20131130_2ec951313a906c6 | 119.75; 0.067; Avoid | 89.00; 0.090; Avoid | 121.35; 0.066; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20006763_20170812_2ec95132ad649d3 | 55.93; 0.097; Avoid | 39.70; 0.137; Avoid | 59.18; 0.092; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20006855_20170827_2ec95134956d2ba | 84.16; 0.081; Avoid | 64.00; 0.106; Avoid | 82.10; 0.083; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20011095_20200221_333abfe6524c00e | 96.47; 0.065; Avoid | 72.90; 0.086; Avoid | 94.69; 0.066; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20011541_20201031_333abfe65362df3 | 58.30; 0.079; Avoid | 34.20; 0.135; Avoid | 59.63; 0.077; Avoid | Unknown |
| Toronto JDMI Additional | AIPR_20011745_20210331_333abfe653d99da | 103.05; 0.104; Avoid | 71.50; 0.150; Avoid | 98.02; 0.109; Avoid | Unknown |

<!-- TABLE_6_END -->

### Interpretation and cohort provenance

Common screening identified a higher review burden in wide-field PCNN than in the other cohorts. Automated volumes were more often closer to the reference in all six selected groups, but sometimes worsened individual volume errors or PSAD classification. The strongest statistical evidence arose in cohorts with automated surrogate references. Small manual-reference groups, data-derived calibration, and selection on disagreement make the findings exploratory. The analysis evaluates a fixed review criterion rather than a proven clinical tolerance or deployed correction workflow. ProstateX threshold calibration is in-sample, and a separate prospective cohort would be needed to evaluate transportability independently.

The six clinical analysis cohorts are disjoint at the patient level. PI-CAI RUMC excluding ProstateX RUMC retains 275 of 391 parent-cohort patients after removing the 116 patients represented in ProstateX RUMC, matched by patient_id. Toronto JDMI Additional retains 3,722 of 4,105 archived parent-cohort patients after removing all 383 Toronto JDMI manual-reference patients, matched by amrn. Patient identifiers were compared within their respective Dutch and Toronto identifier systems. Cohort membership and removals are exported for audit. Disjointness removes duplicate patients; differing populations and reference methods still require consideration before combining estimates. PSAD category agreement concerns reference-derived categories, not cancer detection. Missing outcomes and the small number of outcome-positive cases in these selected PI-RADS 3 outliers constrain biopsy interpretation.

**Toronto cohort version:** The additional cohort is derived from the archived 4,105-patient parent, preserving its saved measurements and exclusions. Its saved audit records 11 excluded accessions, including six with very small nnUNet702 surrogate volumes. The current exclusion file also contains 11 accessions, but four do not occur in the full saved listing and its membership differs from the archived exclusions. Applying it selects 4,109 patients: six additions and two removals relative to the archived 4,105. It is not substituted for the archived parent used here. The manual-reference cohort remains 383 with unchanged membership. Archived exclusions, current-file membership differences, and patient-subtraction audits are exported under results/paper_new_results/; parent configs and source exports were preserved.

**Reproducibility:** Run [the synthesis notebook](notebooks/paper_new_results.py) in cell order using the project environment. It recomputes requested metrics from saved measurements, reconciles repeated site exports, checks membership and provenance, derives disjoint analysis cohorts, and generates Tables 1–6 and Figures 1–3. Numeric tables, case audits, reviewed provenance differences, and input SHA-256 hashes are saved in [results/paper_new_results](results/paper_new_results/). Table markers permit regeneration without replacing this prose. This synthesis performs no new training, inference, or mask processing.
