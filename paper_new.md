---
title: "Deep Learning Prostate MRI Volumetry for Quality Assurance of Ellipsoid Measurements and PSA Density Assessment"
bibliography: paper_new.bib
csl: elsevier-vancouver.csl
link-citations: true
---

# Abstract

Previous studies have shown that automated deep learning prostate volumetry can improve measurement accuracy, but its practical utility, such as to support quality improvement by correcting erroneous clinical measurements, remains less explored. We retrospectively studied deep learning volumetry on T2-weighted MRI with clinical ellipsoid estimates in 5,016 examinations in public PI-CAI cohorts spanning three centers and one internal Toronto cohort. Automated volumetry reduced volume error across all cohorts. Substantial disagreement between automated volumetry and clinical ellipsoid estimates occurred in 1.3–4.2% of cases across three centers; automated volumes were closer to the reference in 163/172 discordant cases (94.8%, p < 0.001). In the remaining cohort acquired with a wide-field protocol not represented in model training, substantial disagreement occurred in 19.0% of cases; automated volumes were still closer to the reference in 47/60 discordant cases (78.3%, p < 0.001). Among these discordant cases, there were 27 patients with PI-RADS 3 findings and known biopsy outcomes. Automated measurements would have spared six additional outcome-negative patients from biopsy using a PSA density threshold of 0.15 ng/mL², increasing simulated specificity from 37.5% (9/24) to 62.5% (15/24, p = 0.03). Sensitivity remained 66.7% (2/3). These findings suggest a practical role for automated volumetry as a second measurement for quality assurance, with substantial discrepancies prompting radiologist review.

# Introduction

Reliable prostate volume estimation is important for interpreting prostate-specific antigen density (PSAD), calculated by dividing serum PSA by whole-gland volume. The Prostate Imaging Reporting and Data System (PI-RADS) version 2.1 recommends reporting prostate volume using segmentation or an ellipsoid approximation derived from three linear dimensions [@turkbey2019pirads]. PSAD complements MRI findings in risk assessment, particularly when findings are negative or equivocal, and has been incorporated into risk-adapted biopsy strategies [@schoots2021risk]. Because volume is the denominator of PSAD, errors in volume measurement propagate to PSAD error, with approximately proportional effects on its relative error. Inaccurate volume estimation may therefore alter risk categorization or threshold-based biopsy recommendations.

Although ellipsoid volumetry is reproducible overall, differences between readers can alter PSAD classification in individual patients. In a study of 40 patients assessed by seven radiologists, ellipsoid volume measurements showed excellent inter-reader reproducibility (ICC 0.988), yet reader differences resulted in discordant classification at a PSAD threshold of 0.15 ng/mL² in 7/40 patients (17.5%) using the traditional ellipsoid formula, compared with 2/40 (5%) using manual planimetry [@hamzaoui2022volume]. In a Swedish national quality-assurance survey of ten cases, involving 43 radiologists from 17 departments, agreement with expert-consensus PSAD threshold classification was generally high, but fell to 35/45 (78%) and 39/44 (89%) assessments for two small prostates of 24 and 25 mL, respectively. For another case, reader-estimated volumes ranged from 40 to 107 mL (compared with an expert-consensus volume of 52 mL), demonstrating that large errors can occur in practice [@wallstrom2025quality]. These findings motivate evaluating whether automated volumetry can improve individual-patient volume assessment and help identify discrepancies warranting review.

The manual reporting workflow introduces additional opportunities for errors when dimensions and calculated volumes are manually transcribed between image viewers, calculators, and text reporting interfaces. In a review of 253 ultrasound and dual-energy X-ray absorptiometry reports, Bauer et al. reported numerical transcription errors, defined as a recorded value differing by at least 10% from the source measurement, in 6–28% of reports depending on report type [@bauer2018transcription]. In a DXA reporting study, 15 of 100 finalized reports contained 25 numerical transcription errors despite attending review, including seven errors involving negative numbers and 18 involving other incorrect digits; the preliminary reports contained even more errors, involving 20 of 100 reports and 44 transcription errors [@wachsmann2018integration]. These studies support the need to reduce manual numerical transfer in reporting.

Automated segmentation offers reproducible voxel-based volumetry without human intervention and without assuming an ellipsoid gland shape. Since 2012, there are now many public annotated datasets provide resources for developing and evaluating prostate segmentation methods across differing imaging conditions [@litjens2014promise12; @adams2022prostate158; @fedorov2018testretest]. The openly available nnU-Net framework further facilitates reproducible development by automatically configuring preprocessing, network architecture, training, and postprocessing for a segmentation task [@isensee2021nnunet]. Isensee et al.'s recent reassessment highlighted that with appropriately configured convolutional U-Net baselines, including residual-encoder variants, U-Net maintains state-of-the-art performance without requiring complex architecture [@isensee2024revisited]. Together, these resources make automated volumetry accessible for radiologists and researchers.

Previous studies have evaluated both volumetric accuracy and the diagnostic performance of segmentation-derived PSAD, with differing results. AutoProstate, an automated prostate MRI analysis and reporting framework using a 2D nnU-Net ensemble for T2-weighted zonal segmentation, reduced mean absolute percentage volume error from 13% with radiologist ellipsoid estimates to 9% in 80 external-validation patients [@mehta2021autoprostate]. In 489 patients with PSA levels of 4–10 ng/mL, Han et al. reported smaller volume bias and narrower 95% limits of agreement with automated segmentation based on nnU-Net than with ellipsoid estimation (+1.4 mL, −9.9 to +12.6 mL versus +4.4 mL, −15.2 to +24.0 mL) compared to manual ground truth. Discrimination of clinically significant cancer improved using automated instead of ellipsoid-derived PSAD (AUC 0.754 versus 0.719; p=0.02). Among 113 PI-RADS 3 patients, automated PSAD at cutoff of 0.168 ng/mL² identified 14/21 clinically significant cancers, compared with 12/21 using ellipsoid and PSAD at 0.159 ng/mL², but would have recommended six additional biopsies overall, including four additional biopsies in patients without clinically significant cancer [@han2026volumetry]. 

In contrast, Colvin et al. found no significant improvement in PSAD discrimination for Gleason ≥3+4 cancer in 397 patients using semi-automated segmentation (automation with vendor software followed by manual correction) rather than ellipsoid measurements (AUC 0.64 versus 0.63; p=0.45). An improvement was observed for Gleason ≥4+4 disease (AUC 0.63 versus 0.61; p=0.02). Semi-automation had smaller mean volumes (51.0 versus 55.4 mL) and higher inter-reader reproducibility (ICC 0.98 versus 0.93) [@colvin2021measurement] Similarly, Bayraktaroğlu et al. reported broadly comparable PSAD discrimination using commercial AI volumetry and three readers’ ellipsoid measurements in 171 patients (AUC 0.788 versus 0.755–0.781). At a PSAD threshold of 0.15 ng/mL², AI and mean-reader ellipsoid measurements yielded similar sensitivity (70.9% versus 68.6%) and specificity (68.2% versus 69.4%); in the 17-patient PI-RADS 3 subgroup, both produced the same classifications with five patients above the threshold, and detecting one of the three clinically significant cancers [@bayraktaroglu2026validation].

Together, these studies establish that while automated volumetry can improve agreement with reference measurements, its downstream effect size on PSAD-based decisions remain questionable. We argue that, in current practice, automation may be better seen as a second quality-assurance measurement that can help specifically identify individual, large discrepancies warranting review. We used nnU-Net, an openly available, self-configuring framework with state-of-the-art segmentation performance, and publicly available prostate MRI training datasets to support reproducible model development and evaluation. We compared models trained on multicenter and single-center data with routine clinical ellipsoid measurements across heterogeneous imaging cohorts, including acquisition settings that could challenge automated segmentation. We assessed segmentation performance and volumetric agreement against manual reference measurements where available, and extended the evaluation to much larger PI-CAI and our institution's cohorts using validated automated surrogate references. We identified cases with suspected substantial volume errors and evaluated reference-derived PSAD category agreement and changes in simulated biopsy recommendations for patients with PI-RADS 3 findings. These analyses examine the potential of automated volumetry to support measurement review and PSAD assessment.

# Methods

## Study datasets and reference standards

We compared automated whole-gland prostate segmentation and volumetry with clinical ellipsoid estimates using public MRI datasets and a complementary Toronto JDMI cohort. Public sources comprised Prostate158, PROMISE12, Prostate Diagnosis, QIN Prostate Repeatability, and PI-CAI, including its ProstateX subset. Manual segmentation annotations were available for all except the non-Prostate-X PI-CAI  subset [@adams2022prostate158; @litjens2014promise12; @fedorov2018testretest]. Radboud university medical center's (RUMC) Bosma22b automated whole-gland masks (a five-fold uuU-Net model geared toward robust prostate segmentation) served as surrogate references where manual masks were unavailable. These masks were manually inspected for quality and six were subsequently excluded. As of July 2026, PI-CAI provided clinical prostate volume, PSA, histopathology, and PI-RADS information. 

Public PI-CAI images sometimes had restricted axial coverage, which would not become an issue had all readers examined the same images, but we found cases where it could underestimate segmentation-derived volume relative to clinical measurements of the full gland, presumably on the taller full clinical image. Therefore, we excluded PI-CAI examinations whose reference masks contacted the superior or inferior image face.

Given the abundance of RUMC data and the availability of both expert manual masks and clinical measurements in ProstateX, we selected ProstateX (all from RUMC) as the primary test cohort and trained two nnU-Net models. First, the multicenter, multiprotocol model used 239 T2-weighted MRI examinations: 119 from Prostate158, 67 from PROMISE12 after excluding RUMC examinations,^[RUMC origin was inferred from acquisition geometry rather than verified institution metadata: through-plane spacing ≥3.6 mm, dimensions other than 320 × 320 × 24, and an origin other than (0, 0, 0). This heuristic excluded 25 of 100 examinations before splitting the remaining 75 into 67 training and eight testing examinations.] 27 from Prostate Diagnosis, and 26 from QIN Prostate Repeatability. The 35 held-out examinations were separated into 22 external-coil examinations (20 Prostate158 and two PROMISE12) and 13 endorectal-coil examinations (six PROMISE12, three Prostate Diagnosis, and four QIN Prostate Repeatability). Note that QIN included two acquisitions per patient and were stratified during train/test splitting.

A second single-center model used 236 examinations from 236 other RUMC patients in PI-CAI, after excluding ProstateX patients as well as any patient with at least one clinically complete RUMC examination, reserving these patients for evaluation. A clinically complete exam was defined as requiring valid prostate volume, PSA, PSAD, PI-RADS, and biopsy results. Since manual segmentation masks were unavailable, the single-center model was trained on Bosma22b automated whole-gland masks.

To expand the analysis, we retrospectively collected an internal Toronto JDMI dataset to complement the public datasets, including axial T2-weighted images and clinical reports from which clinical measurements were extracted. Endorectal-coil examinations, previously treated patients, and failed examinations were excluded. The dataset was split into a subset with manual expert segmentation and another without manual segmentation and only automated segmentation from an internal, validated nnU-Net model. After reviewing the quality of internal automated segmentation, seven more cases were excluded (six due to extreme undersegmentation and one with a large heterogeneous mass replacing almost the entire prostate). This results in two final disjoint groups: 383 patients with manual reference (Toronto JDMI) and 3,718 with automated reference (Toronto JDMI Additional).

## Model parameters and validation

Both the multicenter and single-center models used nnU-Net's three-dimensional full-resolution residual-encoder U-Net (nnunetv2 version 2.7.0, monai 1.5.2) with one T2-weighted input channel and whole-gland segmentation as the target. To preserve reproducibility, we used recommended defaults from nnU-Net without customization. The model-specific plans used z-score intensity normalization and resampling to approximately 3.00 × 0.402 × 0.402 mm for the multicenter model and 3.00 × 0.300 × 0.300 mm for the single-center model. Patch sizes were 28 × 384 × 384 and 16 × 512 × 512 voxels, respectively. The networks had seven and eight resolution stages, respectively, with instance normalization and leaky rectified linear unit activations. Both used a batch size of two for a target GPU VRAM of 24 GB with Nvidia 4090.

Five folds were trained for each model, with 100 epochs per fold, 250 training iterations and 50 validation iterations per epoch. The default nnU-Net trainer used stochastic gradient descent with initial learning rate 0.01, Nesterov momentum 0.99, weight decay 3 × 10⁻⁵, polynomial learning-rate decay, combined soft Dice and cross-entropy loss with deep supervision. Images were augmented by rotation, scaling, mirroring, Gaussian noise and blur, brightness and contrast changes, gamma transformations, and simulated low-resolution.

Predictions were generated using five-fold ensembles, with a sliding-window tile step of 0.8 and mirroring disabled. Given that the prostate is a single connected organ, largest-connected-component filtering was performed for output of both models, resulting in slight improvement by removing small stray voxels.

The models' performance were validated by measuring whole-gland overlap using the mean Dice similarity coefficient (DSC) and mean volume errors of estimated volumes against the volumes of the manual reference segmentation in ProstateX and Toronto JDMI. DSC is defined as $\mathrm{DSC} = \frac{2\mathrm{TP}}{2\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}$, where TP, FP, and FN denote true-positive, false-positive, and false-negative voxel counts, respectively. DSC ranges from 0 (no overlap) to 1 (perfect overlap). Volume errors were summarized by: mean absolute error, $\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N}|V_i-R_i|$, root mean squared error, $\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(V_i-R_i)^2}$, in mL, mean absolute percentage error $\mathrm{MAPE} = \frac{100\%}{N}\sum_{i=1}^{N}\frac{|V_i-R_i|}{R_i}$ and root mean squared percentage error $\mathrm{RMSPE} = 100\%\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\frac{V_i-R_i}{R_i}\right)^2}$, where $N$ is the number of examinations, $V_i$ is the measured volume, and $R_i$ is the reference volume. Both volume error (in mL) and relative errors are important because a seemingly large volume error in a large prostate may actually be a small error on relative scale, and a seemingly large relative error in a small prostate may actually be a small volume error in units of milliliters. RMSEs are susceptible to outliers whereas MAEs are more robust.

## Ellipsoid vs automated volumetry disagreement identification and analyses

Subsequent analysis examined whether an automated measurement available to a radiologist, if discordant from the report's ellipsoid estimates, could potentially improve downstream PSAD categorization and biopsy decisions. For brevity, we presented results only using the multicenter model because of its broader generalization. Disagreement was assessed using Bland-Altman analysis of signed volume differences, $V_{\mathrm{ellipse}} - V_{\mathrm{auto}}$ in mL, and relative differences, $100\% \times \frac{V_{\mathrm{ellipse}} - V_{\mathrm{auto}}}{(V_{\mathrm{ellipse}} + V_{\mathrm{auto}})/2}$. Note that this screening did not use each patient's reference volume, which would not be available to radiologists at time of reporting. 

For standardization, we chose ProstateX RUMC as the calibration cohort: the 95% limits of agreement (mean difference $\pm 1.96$ sample standard deviations) yielded bounds of −14.28 to +25.74 mL and −22.70 to +39.60%. Most other cohorts had broadly similar limits of agreement, except for the wide-field PCNN cohort. An examination was classified as a disagreement outlier only when both its volume difference and relative difference lay outside the bounds. Conservatively requiring both criteria reduced inadvertent selection driven solely by large relative differences in small prostates or large volume differences in large prostates. Applying the same fixed bounds across all six clinical cohorts provided a common selection rule for pooled analyses.

Within this selected disagreement subset, automated and ellipsoid measurements were again compared by absolute reference-volume error and by MAE, RMSE, MAPE, and RMSPE to reference volume. Additionally, exact binomial tests compared the proportion of automated wins among non-ties with chance (50%), with Holm p-value adjustment.

Then, PSAD was calculated among selected subset as PSA in ng/mL divided by volume in mL, and then were assigned to four clinical categories: <0.10, 0.10–<0.15, 0.15–<0.20, and ≥0.20 ng/mL². Improvement in categorization were evaluated by closeness (absolute ordinal distance) to the reference-volume-derived category, with statistical testing using exact binomial tests in non-tie cases, with Holm p-value adjustment.

Lastly, within a smaller subset with maximum recorded PI-RADS 3, biopsy recommendations were simulated using PSAD ≥0.15 ng/mL² for the reference, ellipsoid, and automated volumes. Changes from ellipsoid to automated recommendations were classified as moving into or away from agreement with the reference recommendation, as well as considering positive or negative biopsy status when available. Outcome-positive status was defined as ISUP grade group ≥2. Changes in specificity and sensitivity within this subset was tested using exact McNemar tests.

# Results

## Segmentation performance validation

Both models were evaluated in eight testing cohorts; clinical ellipsoid estimates were available in the six clinical comparison cohorts (Tables 1–3). Whole-gland volume errors were calculated relative to manual whole-gland segmentation masks for Multicenter Data, ProstateX, and Toronto JDMI. Automated surrogate references from each institution respectively were used for PI-CAI and Toronto JDMI Additional. Note that the endorectal coil data and wide-field PI-CAI PCNN data were particularly challenging as the former was not represented in the single-center model and the latter was not encountered in either model's training data, so they serve as a stress test of model generalization, rather than expected performance if training and testing sets were matched.

On the external-coil subgroup of the Multicenter Data testing fold, mean whole-gland DSC was 0.905 for the multicenter model and 0.904 for the single-center model. On ProstateX RUMC, the corresponding values were 0.906 and 0.916, respectively. The single-center model, having never seen endorectal coil or wide-field images, performed poorly in those cohorts (DSC 0.593 and 0.558, respectively). The multicenter model was able to generalize better to the wide-field PCNN data that it also did not encounter in training, achieving DSC 0.882. Visual inspection identified gross localization failures rather than poor contour overlap. On the endorectal coil data, the single-center model often failed to segment the distorted prostate and instead segmented the more circular endorectal coil. On the PCNN data, both models sometimes failed catastrophically and segmented the femoral heads or gluteal muscles instead of the prostate.

Both models had lower MAE, RMSE, MAPE, and RMSPE than clinical ellipsoid estimates in the five clinical cohorts, other than the PCNN stress test. Multicenter model MAE was 5.09 mL in ProstateX and 4.87 mL in Toronto JDMI, compared with 9.20 and 7.18 mL for ellipsoid estimates. In PCNN, the multicenter model still reduced all four errors overall relative to ellipsoid estimates despite failing to segment some cases, whereas the single-center model had higher errors, suggesting more limited generalizing ability from sing-center training. The single-center model trained on RUMC data showed particularly strong agreement with PI-CAI RUMC (exclusive of ProstateX) and PI-CAI ZGT references. This may reflect shared label provenance, because the model was trained on Bosma22b automated whole-gland labels and evaluated against labels produced by the same model in those cohorts. As such, agreement with these automated references does not necessarily mean greater accuracy against independent manual annotations. Figures 1–2 show whole-gland volume agreement against the independent manual references in ProstateX RUMC and Toronto JDMI.

**Table 1. Mean whole-gland Dice similarity coefficient (DSC), testing cohorts only.** PI-CAI and Toronto JDMI Additional use automated surrogate references (Bosma22b and nnUNet702, respectively). Reference masks of Toronto JDMI cases need to be retrieved.

<!-- TABLE_1_START -->

| Dataset | Multicenter | Single-center |
| --- | --- | --- |
| Multicenter Data (external coil) (n = 22) | 0.905\* | 0.904 |
| Multicenter Data (ERC) (n = 13) | 0.911\* | 0.593 |
| ProstateX RUMC (n = 157) | 0.906 | 0.916\* |
| Toronto JDMI (n = 383) | — | — |
| PI-CAI RUMC, exclusive of ProstateX (n = 275) | 0.919 | 0.961\*† |
| Toronto JDMI Additional (n = 3,718) | 0.939 | 0.934 |
| PI-CAI ZGT (n = 168) | 0.929 | 0.953† |
| PI-CAI PCNN (wide field) (n = 315) | 0.882 | 0.558† |

\* *Data from institutions also represented in model training.*  
† *Reference segmentations were generated by the same internal RUMC model that provided surrogate training segmentations.*

<!-- TABLE_1_END -->

**Table 2. Whole-gland volume error: MAE; RMSE (mL).**

<!-- TABLE_2_START -->

| Dataset | Clinical ellipsoid | Multicenter | Single-center |
| --- | --- | --- | --- |
| Multicenter Data (external coil) (n = 22) | — | 3.42; 4.07\* | 2.72; 3.32 |
| Multicenter Data (ERC) (n = 13) | — | 2.43; 3.17\* | 6.33; 8.04 |
| ProstateX RUMC (n = 157) | 9.20; 12.06 | 5.09; 6.47 | 4.57; 5.85\* |
| Toronto JDMI (n = 383) | 7.18; 9.42 | 4.87; 6.76 | 5.28; 6.63 |
| PI-CAI RUMC, exclusive of ProstateX (n = 275) | 12.87; 16.32 | 4.25; 5.26 | 1.42; 2.20\*† |
| Toronto JDMI Additional (n = 3,718) | 8.12; 11.70 | 4.03; 5.72 | 4.70; 6.48 |
| PI-CAI ZGT (n = 168) | 6.62; 8.92 | 2.68; 3.86 | 1.81; 2.43† |
| PI-CAI PCNN (wide field) (n = 315) | 11.69; 16.78 | 6.35; 8.48 | 12.68; 20.65† |

\* *Data from institutions also represented in model training.*  
† *Reference segmentations were generated by the same internal RUMC model that provided surrogate training segmentations.*

<!-- TABLE_2_END -->

**Table 3. Whole-gland percentage volume error: MAPE; RMSPE (%).**

<!-- TABLE_3_START -->

| Dataset | Clinical ellipsoid | Multicenter | Single-center |
| --- | --- | --- | --- |
| Multicenter Data (external coil) (n = 22) | — | 8.52; 10.13\* | 6.47; 7.87 |
| Multicenter Data (ERC) (n = 13) | — | 6.33; 8.69\* | 15.78; 19.50 |
| ProstateX RUMC (n = 157) | 16.36; 21.17 | 8.85; 10.93 | 8.10; 10.47\* |
| Toronto JDMI (n = 383) | 13.89; 18.07 | 9.01; 11.35 | 10.03; 12.36 |
| PI-CAI RUMC, exclusive of ProstateX (n = 275) | 28.14; 33.98 | 9.65; 12.18 | 3.20; 4.90\*† |
| Toronto JDMI Additional (n = 3,718) | 13.06; 18.14 | 6.36; 8.87 | 7.75; 12.80 |
| PI-CAI ZGT (n = 168) | 14.35; 19.14 | 5.87; 7.90 | 4.30; 6.56† |
| PI-CAI PCNN (wide field) (n = 315) | 22.23; 29.57 | 13.09; 18.82 | 35.83; 65.90† |

\* *Data from institutions also represented in model training.*  
† *Reference segmentations were generated by the same internal RUMC model that provided surrogate training segmentations.*

<!-- TABLE_3_END -->

![**Figure 1. Whole-gland volume agreement in ProstateX RUMC (n = 157).** Columns show clinical manual ellipsoid estimates, the multicenter model, and the single-center model. The top row plots measured against reference volume from manual whole-gland segmentation planimetry, with logarithmic axes and identity line. The middle and bottom rows show signed differences (measured − reference, mL) and relative differences (100% × (measured − reference)/reference) against reference volume on a logarithmic x-axis. Histograms at right show counts of differences. Orange points identify the same 2 patients across all scatter panels analyzed in the next subsection due to large disagreement between manual ellipsoid and multicenter model.](results/paper_new_results/prostatex_comparator_agreement.png)

![**Figure 2. Whole-gland volume agreement in Toronto JDMI with manual expert segmentation (n = 383).** Columns show clinical manual ellipsoid estimates, the multicenter model, and the single-center model. The top row plots measured against reference volume from manual whole-gland segmentation planimetry, with logarithmic axes and identity line. The middle and bottom rows show signed differences (measured − reference, mL) and relative differences (100% × (measured − reference)/reference) against reference volume on a logarithmic x-axis. Histograms at right show counts of differences. Orange points identify the same 10 patients across all scatter panels analyzed in the next subsection due to large disagreement between manual ellipsoid and multicenter model.](results/paper_new_results/toronto_manual_comparator_agreement.png)

## When automated and ellipsoid volumes disagree, which is closer to the reference?

Here we examine whether, when automated and ellipsoid volumes disagree, an automated measurement available to the radiologist could help identify measurement errors and improve volume assessment and downstream clinical decisions. Given its consistent performance and broader generalization, subsequent analyses use the multicenter model to represent automated volumetry.

Figure 3 shows agreement between automated multicenter model volumes and clinical manual ellipsoid measurements in the six cohorts. Note that different from Figures 1 and 2 above, these illustrate Bland-Altman agreement between two estimates, rather than either method's error against the reference.

We used ProstateX RUMC as the calibration cohort to establish common disagreement criteria because manual whole-gland segmentation references were available to assess automated performance, and common criteria allow consistent interpretation across cohorts. Its manual vs. automated 95% limits of agreement were: −14.28 to +25.74 mL for absolute-scale differences and −22.70 to +39.60% for relative differences. Cases were defined as outliers only when strictly outside both intervals, limiting selection driven solely by large relative differences at small volumes or large absolute differences at large volumes. The other cohorts' agreement limits were broadly comparable (except for the wide-field PCNN cohort which illustrates a stress test case study), supporting the use of this common screen across cohorts.

![**Figure 3. Bland-Altman analyses showing automated–clinical manual volume disagreement and common screening criteria.** Each row represents one cohort. Left panels show clinical ellipsoid minus multicenter model volume in mL; right panels show 100 × difference / pair mean in percent. Pair mean volume is displayed on logarithmic x-axes. Black dashed lines show the fixed common criteria, −14.28 to +25.74 mL and −22.70 to +39.60%. Orange points with white outlines exceed both intervals and identify the same selected cases in both columns.](results/paper_new_results/bland_altman_clinical_nnunet274.png)

**Table 4. Reference volume errors among cases exceeding both common disagreement criteria: manual → automated.**

<!-- TABLE_4_START -->

| Dataset | Flagged n/N (%) | MAE (mL) | RMSE (mL) | MAPE (%) | RMSPE (%) | Improved / all | Adjusted p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ProstateX RUMC | 2/157 (1.3%) | 33.08 → 4.13 | 33.34 → 4.71 | 65.15 → 7.60 | 66.73 → 8.41 | 2/2 | 0.500 |
| Toronto JDMI | 10/383 (2.6%) | 21.75 → 8.61 | 23.14 → 16.87 | 35.25 → 8.15 | 38.77 → 10.79 | 8/10 | 0.219 |
| PI-CAI RUMC, exclusive of ProstateX | 11/275 (4.0%) | 33.38 → 5.85 | 36.32 → 6.23 | 65.77 → 11.92 | 69.93 → 14.51 | 11/11 | 0.004 |
| Toronto JDMI Additional | 142/3,718 (3.8%) | 28.29 → 5.23 | 32.23 → 7.59 | 41.38 → 9.37 | 50.46 → 20.16 | 135/142 | <0.001 |
| PI-CAI ZGT | 7/168 (4.2%) | 21.70 → 1.43 | 23.71 → 1.90 | 33.24 → 2.25 | 37.03 → 3.01 | 7/7 | 0.047 |
| PI-CAI PCNN (wide field) | 60/315 (19.0%) | 19.15 → 10.07 | 24.39 → 13.26 | 32.62 → 19.25 | 40.29 → 30.09 | 47/60 | <0.001 |

<!-- TABLE_4_END -->

Table 4 compares automated and clinical ellipsoid volume errors in cases exceeding both disagreement criteria. All error measures (MAE, RMSE, MAPE, RMSPE) decreased in each cohort with automated volumetry. Across the two manual-reference cohorts, automated volumes were closer to the reference in 10/12 cases (83.3%; unadjusted p = 0.039). Inspecting the two cases that did not improve, one had severe hyperplasia (192 mL) and the automated volumetry missed segmenting a portion of the lesion, and the other had slight under segmentation by the radiologist that favored the manual ellipsoid by 0.44 mL. Overall across all six cohorts, automated volumes were closer to the reference in 210/232 disagreement cases (90.5%; p < 0.001). Upon inspection, the seven Toronto JDMI Additional cases that did not improve showed undersegmentation or oversegmentation relative to the automated surrogate reference, in cases with motion artifact or large lesions. On visual review of the 13 PI-CAI PCNN cases that did not improve, seven showed segmentation failure involving the femoral head or gluteal muscle, and six showed oversegmentation by the multicenter model relative to the provided reference segmentation.

<!-- 
ID                                         Improvement (mL)
AIPR_20002988_20180720_2ec9509f8708c29          -24.39
AIPR_20002697_20180118_2ec950974243c5d           -0.44 # please check manual segmentation mask
AIPR_20000108_20160714_2ec95031808142e           11.59
AIPR_20007451_20170809_2ec9515013cc7c1           13.10
AIPR_20000643_20150421_2ec9504b95eff90           15.43
AIPR_20001158_20170404_2ec950561fb140b           16.88
AIPR_20002201_20180706_2ec9507c708ede2           16.91
AIPR_20002786_20180330_2ec95099fbd2198           17.63
AIPR_20008833_20170113_2ec9517f737d699           29.86
AIPR_20000257_20171019_2ec950342021cf7           34.85

ID                                         Improvement (mL)
AIPR_20011993_20210702_333abfe6548866c          -33.73
AIPR_20011773_20221117_333abfe657101b2          -22.55
AIPR_20012433_20220308_333abfe6559a798          -21.14 # exclude.json in next round of analysis
AIPR_20011374_20200726_333abfe652f05c4          -14.39
AIPR_20011887_20210722_333abfe65438c9e           -8.82
AIPR_20011935_20210820_333abfe65456fc6           -7.85
AIPR_20012839_20220916_333abfe65689005           -4.71
AIPR_20012134_20210903_333abfe654e6f77           -3.78
AIPR_20011840_20210627_333abfe6541a0dd            0.43
AIPR_20012751_20220808_333abfe65659313            0.60
AIPR_20011948_20210719_333abfe654639f3            2.06
AIPR_20011722_20210130_333abfe653c8fee            3.37
AIPR_20012146_20211027_333abfe654ef470            4.14
AIPR_20008586_20180414_2ec951795c59caa            4.85
AIPR_20011208_20200306_333abfe6528d314            7.07
AIPR_20011434_20221215_333abfe65717eb2            7.60
AIPR_20013659_20200109_333abfe6524028d            7.83
AIPR_20008077_20150518_2ec9515e38d7416            9.48
AIPR_20013165_20230102_333abfe6574bf77           11.19
AIPR_20012278_20220115_333abfe65547725           11.21
AIPR_20012838_20220912_333abfe65688c1c           11.42
AIPR_20011553_20201108_333abfe65368f9f           11.82
AIPR_20009968_20170820_2ec951a9791dd77           12.00
AIPR_20006763_20170812_2ec95132ad649d3           12.98
AIPR_20012736_20220703_333abfe656525b2           14.23
AIPR_20006383_20180308_2ec9511a0c0c556           14.34
AIPR_20002962_20180914_2ecfd97f7ae6738           14.45
AIPR_20003973_20180416_2ec950c416a61e1           14.50
AIPR_20010980_20181206_2ecfd97f7c03398           14.53
AIPR_20004904_20140307_2ec99991320939c           14.53
AIPR_20001474_20180416_2ec9505d477c4e8           14.58
AIPR_20009368_20160531_2ec9518b3c8ee15           14.59
AIPR_20010619_20190131_2ecfd97f7b94dae           14.64
AIPR_20008736_20131115_2ec9517d284ffab           14.79
AIPR_20002670_20141017_2ec950964375385           14.82
AIPR_20009071_20161209_2ec95184ab13535           14.87
AIPR_20007518_20200707_333abfe65358dcc           14.97
AIPR_20012142_20210912_333abfe654ebdbd           15.16
AIPR_20006629_20131210_2ec9512faaeb548           15.20
AIPR_20010212_20180924_2ecfd97f7b19724           15.22
AIPR_20011751_20210312_333abfe653dd472           15.25
AIPR_20010788_20190331_2ecfd97f7bc81ea           15.55
AIPR_20004741_20151226_2ec950e34dbfc35           15.55
AIPR_20013646_20190825_2ecfd97f7c1b46a           15.89
AIPR_20006301_20180224_2ec99990959abbe           16.51
AIPR_20011257_20200427_333abfe652a50ef           16.73
AIPR_20004147_20130624_2ec950c7d554ad8           16.89
AIPR_20001747_20161120_2ec95072c889377           17.56
AIPR_20011063_20191213_333abfe6523a891           17.68
AIPR_20011272_20200403_333abfe652aea32           17.69
AIPR_20010926_20180925_2ecfd97f7bf29c8           17.76
AIPR_20011558_20201126_333abfe6536bad6           17.82
AIPR_20001578_20170215_2ec9506013a85c6           17.90
AIPR_20006611_20130923_2ec9512f4649001           17.99
AIPR_20006855_20170827_2ec95134956d2ba           18.10
AIPR_20001488_20201231_333abfe653da97a           18.11
AIPR_20008309_20170911_2ec951643b32346           18.14
AIPR_20013044_20221227_333abfe6570a1e3           18.24
AIPR_20008725_20220312_333abfe65627015           18.30
AIPR_20012569_20220604_333abfe655efde7           18.43
AIPR_20001083_20180106_2ec950549589f9d           18.59
AIPR_20010610_20220509_333abfe656a1fa8           18.79
AIPR_20008740_20160809_2ec9517d403a8bd           18.99
AIPR_20009867_20180221_2ec951a6de04543           19.01
AIPR_20010357_20180922_2ecfd97f7b4588c           19.08
AIPR_20000840_20140520_2ec9504fcf17ed6           19.61
AIPR_20005801_20160322_2ec9510be0fccb7           19.65
AIPR_20012399_20220223_333abfe6558825e           20.07
AIPR_20012664_20220527_333abfe65627fb5           20.32
AIPR_20001034_20121228_2ec950533ca161a           20.42
AIPR_20012644_20220628_333abfe6561d753           20.44
AIPR_20010961_20190213_2ecfd97f7bfd51d           20.57
AIPR_20010438_20181008_2ecfd97f7b5de2d           20.69
AIPR_20007408_20170617_2ec95140572f3ca           20.95
AIPR_20002472_20170130_2ec999906c2adf8           21.30
AIPR_20013539_20200218_333abfe6523113e           21.30
AIPR_20006512_20160816_2ec9512ca04284a           21.46
AIPR_20013532_20220524_333abfe6563164e           21.49
AIPR_20004607_20130618_2ec9999095e0678           21.71
AIPR_20012696_20220712_333abfe6563a4b5           21.75
AIPR_20011095_20200221_333abfe6524c00e           21.79
AIPR_20005698_20171023_2ec95109549dd1d           22.18
AIPR_20011541_20201031_333abfe65362df3           22.76
AIPR_20011548_20201221_333abfe653660bd           22.92
AIPR_20002073_20141006_2ec95079d28f31a           23.01
AIPR_20012121_20211008_333abfe654df278           23.02
AIPR_20003211_20170208_2ec950a51ad4110           23.09
AIPR_20010479_20180829_2ecfd97f7b6a6e6           23.26
AIPR_20013606_20190503_2ecfd97f7c0ea1f           23.33
AIPR_20012401_20220326_333abfe65588a30           23.53
AIPR_20011132_20200315_333abfe6525f4b9           23.64
AIPR_20010360_20190802_2ecfd97f7b466c2           24.80
AIPR_20002447_20160610_2ec9508216c2f1d           25.40
AIPR_20009371_20171206_2ec9518b4830b64           25.97
AIPR_20012592_20220503_333abfe655ff2af           26.16
AIPR_20013157_20221123_333abfe65745600           26.27
AIPR_20010489_20190213_2ecfd97f7b6d9d2           26.38
AIPR_20011745_20210331_333abfe653d99da           26.52
AIPR_20011530_20201212_333abfe6535571b           26.58
AIPR_20004931_20150914_2ec950e8177e463           26.62
AIPR_20004206_20150314_2ec950c8e2a49c8           26.99
AIPR_20004867_20161224_2ec950e646a0472           27.32
AIPR_20002457_20130821_2ec95082607d825           27.69
AIPR_20012384_20220218_333abfe6557f57f           27.86
AIPR_20010726_20180824_2ecfd97f7bb52ce           28.07
AIPR_20011930_20210722_333abfe65453c30           28.08
AIPR_20009107_20180117_2ec99990959b573           29.02
AIPR_20003974_20141222_2ec950c41d574a9           29.12
AIPR_20006709_20131130_2ec951313a906c6           29.16
AIPR_20005329_20160531_2ec950f10be2ed4           29.69
AIPR_20011539_20201226_333abfe6535d804           30.03
AIPR_20010608_20180911_2ecfd97f7b91920           30.52
AIPR_20011447_20200928_333abfe6531d73d           30.72
AIPR_20006327_20140806_2ec951188d162c4           30.73
AIPR_20004061_20160804_2ec950c6889cf77           31.03
AIPR_20011234_20200402_333abfe65299d3b           31.96
AIPR_20011895_20210619_333abfe6543e67c           33.00
AIPR_20012158_20211031_333abfe654f86f3           33.49
AIPR_20012635_20220527_333abfe65619ac3           33.86
AIPR_20013624_20210118_333abfe6539a41e           34.59
AIPR_20010367_20190706_2ecfd97f7b48c03           34.83
AIPR_20009053_20160301_2ec951844ac5173           36.62
AIPR_20008595_20140607_2ec95179733ac8a           37.26
AIPR_20008731_20210106_333abfe653db533           38.89
AIPR_20001168_20171102_2ec95056c4c4e79           41.19
AIPR_20009448_20180217_2ec9519cbc063cc           41.56
AIPR_20004646_20160624_2ec950e0ff6ec83           43.41
AIPR_20006438_20161106_2ec9512acd059ba           45.13
AIPR_20013596_20190304_2ecfd97f7c0bb84           47.10
AIPR_20012550_20220519_333abfe655e2cda           47.17
AIPR_20007151_20180212_2ec9513b3243840           47.28
AIPR_20004279_20160326_2ec950caaac76d3           48.07
AIPR_20000492_20130524_2ec95039080d6a7           51.20
AIPR_20010798_20180910_2ecfd97f7bcb12b           51.39
AIPR_20012763_20220805_333abfe65660bd1           52.37
AIPR_20006527_20150905_2ec9512d008bc07           53.27
AIPR_20012490_20220216_333abfe655be9ea           53.37
AIPR_20012922_20221007_333abfe656b84e6           53.56
AIPR_20008220_20170428_2ec951620383c35           56.08
AIPR_20010493_20180927_2ecfd97f7b6ee6f           56.27
AIPR_20007948_20180108_2ec9515c0d018f4           65.65
AIPR_20003867_20150212_2ec950c17f80452           70.80
AIPR_20004443_20141103_2ec950cda96866b          102.61

ID                      Improvement (mL)
picai_11271_1001294      -38.57
picai_10901_1000917      -31.00
picai_10142_1000144      -22.70
picai_11316_1001339      -22.23
picai_10281_1000286      -19.01
picai_10222_1000226      -16.80 # Over segmented?
picai_10307_1000313      -11.54 # Over segmented?
picai_10342_1000348       -6.10 # Over segmented?
picai_10807_1000823       -4.92 # Over segmented?
picai_10425_1000433       -3.62 # Over segmented?
picai_10134_1000136       -1.33
picai_11108_1001131       -1.29 # Over segmented?
picai_10967_1000986       -0.34
picai_10676_1000692        1.10
picai_10841_1000857        1.69
picai_11032_1001052        2.21
picai_10453_1000461        2.37
picai_10286_1000292        3.12
picai_10756_1000772        3.16
picai_10527_1000537        3.39
picai_11089_1001112        3.66
picai_10626_1000640        4.13
picai_10015_1000015        5.11
picai_11078_1001100        5.72
picai_10276_1000281        5.81
picai_10692_1000708        5.92
picai_10678_1000694        6.54
picai_10263_1000267        7.19
picai_11039_1001059        7.27
picai_10615_1000629        7.72
picai_10541_1000552        8.36
picai_11282_1001305        8.47
picai_10387_1000393        9.15
picai_10304_1000310        9.99
picai_10147_1000149       10.31
picai_10509_1000518       10.70
picai_10277_1000282       11.66
picai_10878_1000894       12.69
picai_10714_1000730       13.05
picai_10022_1000022       13.54
picai_11244_1001267       14.18
picai_10902_1000918       14.19
picai_10888_1000904       15.11
picai_10278_1000283       15.42
picai_11442_1001466       15.79
picai_10221_1000225       16.36
picai_10777_1000793       17.58
picai_10905_1000921       18.19
picai_11406_1001430       18.37
picai_11380_1001403       19.41
picai_10859_1000875       21.79
picai_11288_1001311       22.98
picai_11006_1001026       27.19
picai_10642_1000658       28.09
picai_10808_1000824       28.15
picai_10647_1000663       36.64
picai_10202_1000206       42.43
picai_10921_1000938       42.99
picai_11018_1001038       48.46
picai_10349_1000355       77.20
 -->

## Impact on PSAD category and simulated biopsy recommendation

PSAD categories calculated using automated volumetry were closer to the reference category in 73/142 Toronto JDMI Additional outliers, equally distant in 62, and farther in seven (Table 5). Across the two manual-reference cohorts, automated PSAD categories were closer in eight of 11 evaluable cases, equally distant in three, and farther in none (p = 0.008). One ProstateX disagreement case lacked PSA and was excluded from PSAD comparisons. Across all six cohorts, automated categories were closer in 116 of 231 evaluable cases, equally distant in 94, and farther in 21 (p < 0.001).

**Table 5. Ordinal four PSAD category closeness among common-criteria outliers.**

<!-- TABLE_5_START -->

| Dataset | Automated closer | Equal distance | Manual closer | Adjusted p | PSAD ≥0.15 → <0.15, n | PSAD <0.15 → ≥0.15, n |
| --- | --- | --- | --- | --- | --- | --- |
| ProstateX RUMC (n = 1, another missing PSA value) | 1 | 0 | 0 | 1.000 | 0 | 1 |
| Toronto JDMI (n = 10) | 7 | 3 | 0 | 0.078 | 0 | 1 |
| PI-CAI RUMC, exclusive of ProstateX (n = 11) | 6 | 4 | 1 | 0.375 | 1 | 3 |
| Toronto JDMI Additional (n = 142) | 73 | 62 | 7 | <0.001 | 31 | 9 |
| PI-CAI ZGT (n = 7) | 3 | 4 | 0 | 0.500 | 1 | 0 |
| PI-CAI PCNN (wide field) (n = 60) | 26 | 21 | 13 | 0.213 | 23 | 4 |

<!-- TABLE_5_END -->

Forty-seven out of 232 disagreement cases had PI-RADS 3: none in ProstateX, three in Toronto JDMI, one in PI-CAI RUMC excluding ProstateX RUMC, 30 in Toronto JDMI Additional, one in ZGT, and 12 in PCNN. The individual cases are tabulated and discussed in Appendix A. Twenty cases had unknown outcomes, comprising one Toronto JDMI and 19 Toronto Additional cases.

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

Overall, 13 of 47 (27.7%) PI-RADS 3 cases had changed biopsy recommendations between manual ellipsoid and automated volumetry, with nine (69.2%) moving toward the reference recommendation and four (30.8%) moving away. Among the changed cases, six had known negative outcomes and automated volumetry correctly changed to avoiding biopsy in all six; two had known positive outcomes and automated volumetry correctly changed to recommendation in one and incorrectly changed to avoidance in the other. (Five other cases had unknown outcomes.) This results in specificity increased from 37.5% (9/24) with manual ellipsoid measurements to 62.5% (15/24; p = 0.03) with automated volumetry, while sensitivity remained 66.7% (2/3) for both methods.

# Discussion

In this retrospective study of open datasets from multiple centers as well as our large internal Toronto JDMI dataset, automated T2-weighted MRI volumetry reduced whole-gland volume error relative to clinical ellipsoid measurements in all centers. Its potential value was made apparent when the two measurements disagreed substantially. Automated volumes were closer to the reference in 210/232 selected cases (90.5%, p < 0.001), including 163/172 (94.8%, p < 0.001) excluding the challenging wide-field PCNN cohort and 47/60 (78.3%, p < 0.001) in PCNN; even just within the two smaller cohorts with manual references automated volumes was closer in 10/12 disagreement cases (83.3%, p = 0.039). These findings support automated volumetry as a quality-assurance tool, as a second measurement that can help identify discrepancies necessetating review. It also emphasized the importance of matching training and testing sets, even though a mismatched model still provided improvement.

Prior studies on automated prostate volume and PSAD estimation have demonstrated the feasibility of automated reporting and improved volume estimation, while findings on downstream PSAD discrimination in full clinical cohorts have varied, all with relatively small practical effect sizes [@mehta2021autoprostate; @han2026volumetry; @colvin2021measurement; @bayraktaroglu2026validation]. It is important to note that a more accurate volume need not change a patient's PSAD category or improve cancer discrimination, especially when the discrepancy is small. We suspect many cases with small improvement effect sizes diluted the overall effect size in these previous studies.

<!-- TODO: Read [@ghavami2019downstream] -->

Therefore, this study extends previous work by focusing on the discrepant cases in which automated and routine clinical measurements disagree, addressing a complementary question: whether two measurements available during reporting can identify cases in which further inspection, or use of automated volumetry, may be useful. Given our large cohort size, we were able to focus only on outliers. These represent cases that should trigger radiologist review, and we demonstrate that significant error reduction can be achieved. 

<!-- The PCNN dataset, which included wide-field images not seen by the model before, contained much more disagreement (60/315, 19.0%), illustrating the importance of having training data to be representative the testing data. Despite this train-test mismatch, automation still significantly reduced error among the identified disagreements. -->

<!-- The proposed screen used only the clinical and automated volumes, without requiring a reference measurement for the patient being assessed. Requiring disagreement beyond both absolute and relative limits reduced selection driven solely by gland size. The resulting review burden varied substantially across settings: 1.3–4.2% of cases were selected in the five other clinical cohorts, compared with 19.0% in PCNN. This variation is relevant to implementation because a threshold that produces an acceptable number of flags in one setting may generate substantially more in another. Moreover, agreement between the two methods does not establish that either is correct, and disagreement alone cannot identify which measurement is erroneous. The limits used here were empirically calibrated in ProstateX rather than defined by clinical consequences; they require independent validation before use as clinical tolerances. -->

When automated and ellipsoid volumes disagreed, automated volumetry frequently changed PSAD classification while generally improving or preserving agreement with the reference-derived category. Among 231 evaluable disagreement cases across all PI-RADS categories, automated volumetry changed the PSAD category in 144 (62.3%) and crossed the 0.15 ng/mL² threshold in 74 (32.0%), demonstrating potential for affecting downstream management. Automated PSAD categories were almost always closer to (116/231, 50.2%) or equidistant from (94/231, 40.7%) the reference-derived category.

In the PI-RADS 3 biopsy decision simulation, automated volumetry would have avoided biopsy in 6 additional outcome-negative patients out of 24, significantly increasing specificity from 37.5% to 62.5% (p = 0.03). Sensitivity was numerically unchanged at 2/3 (66.7%), with two opposing changes. Since PI-RADS 3 cases with biopsy results are rare across the board, most of these cases were from our own instutition, so these results may only be reflective of local practice.

Initial model validation highlighted the importance of acquisition diversity: the multicenter model was more robust across imaging settings, whereas the single-center RUMC model performed only slightly better on ProstateX RUMC than the multicenter model (DSC 0.916 vs 0.906) which left out RUMC data in training to fairly assess generalization ability. Therefore, the exceptional performance of the single-center model on PI-CAI RUMC may mostly reflect shared Bosma22b automated annotations for training and evaluation. Meanwhile, the single-center model failed to segment some unfamiliar endorectal coil and wide-field protocols. Given precense of these failures, we recommend still displaying segmentation masks and retaining reader review when measurements disagree to avoid silent catastrophic failures.

Aside from the limitations mentioned above, the main limitation is the study's reliance on automated surrogate references for most cases. These references used the same nnU-Net architecture as the evaluated models and may share systematic errors. Nonetheless, consistent findings in the smaller manual-reference cohorts give confidence in the overall results. Initial visual inspection helped identify obvious segmentation errors. While careful independent expert manual planimetry remains gold standard in such studies, this was impractical across thousands of examinations needed to study relatively uncommon outliers and PI-RADS 3 disagreement cases. In terms of practical implementation, most institutions today are still limited by the heavy computational requirements of deploying deep learning models. The U-Net architecture, even with our optimized parameters, takes roughly three seconds to process each case on a GPU and an order of magnitude longer on a CPU.

Overall, our findings support integrating deep learning automated volumetry into prostate MRI workflows to improve measurement quality and PSAD assessment, with the potential to increase biopsy specificity and reduce unnecessary biopsies.

# References

::: {#refs}
:::

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Appendix A: Table of PI-RADS 3 common-criteria outlier and outcomes

Table 6 tabulates the simulated biopsy decision and biopsy outcomes in 47 cases.

In Toronto JDMI, all three methods gave the same biopsy recommendation for each of the three cases: one recommendation with a negative outcome, one avoidance with a negative outcome, and one recommendation with an unknown outcome.

In PI-CAI RUMC, the identified patient had ISUP grade group 2. Reference and automated PSAD exceeded the biopsy threshold and would correctly recommend biopsy, whereas manual PSAD would not, i.e. the biopsy recommendation would differ.

In Toronto JDMI Additional, eight of 30 recommendations changed between ellipsoid and automated: six from recommend to avoid and two from avoid to recommend. Outcome was available for 11 cases. In the single outcome-positive case, only manual ellipsoid measurements would recommend biopsy, i.e. both reference and automated volumetry would incorrectly avoid biopsy. Among the ten outcome-negative cases, reference, manual ellipsoid, and automated measurements would avoid biopsy in seven, five, and seven, respectively (specificity: 70%, 50%, and 70%). All three methods had concordant recommendations and avoidance in two and five cases, respectively. In the remaining three outcome-negative cases, the recommendations by reference, manual, and automated measurements were: (recommend, recommend, avoid), (avoid, recommend, recommend), and (avoid, recommend, avoid).

The one ZGT case had ISUP grade group 0, and all three measurements would recommend biopsy. In PCNN, which was challenging for the automated model, four of 12 recommendations changed between ellipsoid and automated, all from recommend to avoid. One patient had ISUP grade group 2, and 11 had grade group 0 or 1. In the single outcome-positive case (ISUP grade group 2), all three measurements would recommend biopsy. Among the 11 outcome-negative cases (ISUP grade group 0 or 1), reference, manual ellipsoid, and automated measurements would avoid biopsy in four, three, and seven, respectively (specificity: 36.4%, 27.3%, and 63.6%). All three methods agreed on recommending biopsy in four outcome-negative cases and avoiding biopsy in two. In the remaining five outcome-negative cases, recommendations by reference, manual, and automated measurements were: (recommend, avoid, avoid) in one, (recommend, recommend, avoid) in two, and (avoid, recommend, avoid) in two.

**Table 6. Every PI-RADS 3 common-criteria outlier, including unknown outcomes.**

<!-- TABLE_6_START -->

| Dataset | Case | Reference: mL; PSAD; biopsy | Manual: mL; PSAD; biopsy | Automated: mL; PSAD; biopsy | Recorded outcome |
| --- | --- | --- | --- | --- | --- |
| PI-CAI PCNN (wide field) | picai_10304_1000310 | 51.08; 0.190; Recommend | 37.00; 0.262; Recommend | 55.16; 0.176; Recommend | ISUP 2 |
| PI-CAI RUMC | picai_10658_1000674 | 41.53; 0.185; Recommend | 73.00; 0.105; Avoid | 44.06; 0.175; Recommend | ISUP 2 |
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
