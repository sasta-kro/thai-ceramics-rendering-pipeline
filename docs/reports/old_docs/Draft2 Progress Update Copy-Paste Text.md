# Draft 2 Progress Update: Copy-Paste Text

This file contains revised text for the project paper. It records the second pot, the Thai-Chinese cultural justification, and the completed Python video-frame extraction work without directly editing the Word document.

## Important Factual Positioning

- The two ordered objects should no longer be described as both unpainted or matte.
- The first object is an unpainted red earthenware pot with carved surface details.
- The second object is marketed in the supplied product image as a 7-inch dragon jar (`โอ่งมังกร 7 นิ้ว`). It has a glossy brown glaze and a raised dragon motif.
- The second object's exact workshop and production region have not yet been independently verified. The paper should therefore describe it as a candidate associated with the Thai dragon-jar tradition, not automatically claim that it was made in Ratchaburi.
- The dragon motif has Chinese origins, but dragon jars developed into a recognized Thai-Chinese craft tradition in Ratchaburi and became widely used in Thai households. The project can present this object as evidence of cultural adaptation and exchange rather than as a purely indigenous Thai form.
- The specific Ratchaburi dragon-jar industry began in the twentieth century. Earlier Chinese influence on ceramics in Thailand can be discussed separately, but the paper should not claim that this exact modern jar type existed in ancient Thailand without stronger evidence.

## 1. Replacement Abstract

Replace the current abstract beginning with **"This project investigates image-based 3D reconstruction..."** with the following paragraph:

> **Abstract:** This project investigates image-based 3D reconstruction of pottery associated with Thai material culture for an interactive augmented or virtual reality exhibition. During the current reporting period, the team completed a structured survey of candidate vessels, compared marketplace claims with cultural and historical sources, consulted the course professor about the project's cultural scope, selected two contrasting pottery objects, and completed their purchase. The first object is an unpainted earthenware pot with carved details, while the second is a glossy dragon-motif jar connected to a Thai-Chinese ceramic tradition. The contrast between matte and reflective surfaces will allow the team to study how material appearance, ornamentation, and repeated geometry affect photogrammetric reconstruction. The team also implemented a Python program that extracts individual frames from 60 FPS video in preparation for dataset selection and COLMAP processing. After the objects arrive, the capture and preprocessing workflow will be tested and refined before reconstruction in COLMAP, mesh cleanup in Blender, and integration into the course exhibition platform.

## 2. Replacement Introduction Goal Paragraph

Replace the Introduction paragraph beginning with **"The topic was partly inspired..."** with the following paragraph:

> The topic was partly inspired by two team members from Myanmar who are familiar with similar clay water containers. This shared regional familiarity encouraged the team to study pottery used in Thailand while carefully distinguishing local traditions from visually similar Chinese, Indian, and other regional forms. The project recognizes that Thai material culture has also developed through long-term cultural exchange. Therefore, the selected objects may represent both locally rooted Thai pottery and forms that originated through Thai-Chinese adaptation. The main goal is to produce textured 3D models of at least two pottery objects associated with Thai cultural life and present them in an AR or VR environment where users can rotate, scale, and inspect them.

## 3. Replacement Overall Project Design Paragraph

Under **III. Methodology, A. Overall Project Design**, replace the paragraph beginning with **"The project follows an experimental..."** with:

> The project follows an experimental image-based reconstruction process. At least two real pottery objects associated with Thai material culture will be photographed or recorded from multiple viewpoints after delivery. The selected objects intentionally differ in material appearance and decoration: one has a matte, unpainted clay surface with carved details, while the other has a glossy glaze and dragon motif. Images selected from the captures will be processed to estimate camera positions, reconstruct object surfaces, generate textures, and create models suitable for real-time visualization.

## 4. Replacement Cultural Research and Object Selection Section

Under **III. Methodology, B. Cultural Research and Object Selection**, replace everything after that subsection heading and before **C. Equipment and Capture Area** with the following text:

> During the current reporting period, the team conducted a broad online survey of pottery and water-container listings rather than purchasing the first available objects. Candidate products were compared using their names, descriptions, forms, materials, surface decoration, stated origins, and visual similarities to documented pottery traditions. The team cross-referenced these observations with cultural and historical sources and rejected many listings that appeared to represent unrelated Chinese or Indian products, generic imported decoration, or unsupported claims of Thai origin. This process reduced the initial pool to objects for which the cultural relevance could be explained more carefully.
>
> The team also researched the historical development and household functions of pottery in Thailand. This review included earthenware and glazed ceramics, water-storage vessels, jars, regional craft traditions, and the influence of trade and migration on ceramic production. Instead of treating cultural influence as proof that an object is "not Thai," the team examined how imported techniques and motifs were adapted, produced, used, and recognized within Thailand over time.
>
> For each shortlisted object, the team began recording its likely Thai name, region, associated craft tradition, historical or household use, and supporting cultural sources. These fields remain under investigation and will be revised as stronger evidence becomes available. Marketplace titles are being used only to identify how products are sold; they are not being treated as sufficient proof of cultural origin.
>
> Candidate selection combined cultural relevance with photogrammetric suitability. The team considered whether each object was rigid, affordable, safe to handle, visible from many directions, and rich enough in shape, texture, carving, or color variation to support feature matching. Reflectivity was also considered because strong moving highlights can appear different between camera viewpoints and may interfere with reliable feature correspondence.
>
> Based on this screening, the project scope was expanded from one primary object to at least two pottery objects, and two different candidates have now been ordered. The first is an unpainted red earthenware pot with carved decorative bands, a lid, and a pedestal. Its matte surface and raised details are expected to provide useful visual features, although its rounded symmetry may still make camera orientation difficult to estimate.
>
> The second object is marketed as a 7-inch dragon jar and has a glossy brown glaze with a raised dragon motif. The team recognizes the motif's Chinese origin and discussed this concern with the course professor before confirming the object. The professor approved its inclusion because dragon jars are also established within Thai cultural life. Thai sources describe the Ratchaburi dragon-jar tradition as developing through Chinese immigrant craftsmanship, local clay and production, and adaptation to Thai household use. The jars became associated with Ratchaburi and were widely used in Thailand for water storage, food storage, and household decoration [7], [8]. The object is therefore included as a Thai-Chinese cultural form rather than being described as purely indigenous Thai pottery. Its exact place of manufacture will remain unconfirmed until the team obtains reliable product or workshop information.
>
> Selecting these two contrasting objects creates a stronger technical comparison. The first pot provides a mostly matte and visibly carved surface, whereas the second provides glaze, stronger reflections, color variation, and a repeated decorative motif. Their reconstruction results can be compared to determine how these surface properties affect feature detection, image registration, dense point-cloud coverage, mesh completeness, and texture quality.

## 5. Revised Figure Captions and Placement

Replace the existing Figure 1 caption with:

> **Fig. 1.** The first ordered candidate: an unpainted red earthenware pot with carved decorative bands, a lid, and a pedestal. Product image supplied by the team from T&T Pottery.

Insert the supplied image of the glossy dragon jar after the new cultural-selection text and before **C. Equipment and Capture Area**. Use this caption:

> **Fig. 2.** The second ordered candidate, marketed as a 7-inch dragon jar (`โอ่งมังกร 7 นิ้ว`), with a glossy brown glaze and raised dragon motif. The exact production location remains under verification. Product image supplied by the team.

## 6. Replacement Image Acquisition and Dataset Preparation Section

Under **III. Methodology, E. Image Acquisition and Dataset Preparation**, replace the existing paragraph with:

> The team has implemented a Python program that extracts individual frames from a 60 FPS video and saves them as separate image files. This creates a controllable image source for the photogrammetry pipeline and provides a repeatable alternative to manually capturing every photograph. However, every extracted frame will not automatically be sent to feature extraction or reconstruction. Adjacent frames in a 60 FPS video can be nearly identical, producing unnecessary computation and many redundant feature matches without adding a useful change in viewpoint.
>
> The preprocessing program will therefore be extended or configured to select frames at an adjustable interval, such as every second, third, or a larger number of frames, depending on the speed of camera or object rotation. Frame selection will aim to preserve substantial overlap while ensuring enough viewpoint change for camera-pose estimation. Blurred, obstructed, poorly exposed, and near-duplicate frames will also be removed. The interval is a starting plan rather than a fixed rule and will be changed after pilot reconstruction results are examined.
>
> Python will be used to prepare the dataset required by COLMAP. The planned preprocessing outputs include a selected image sequence with consistent file names, a contact sheet for rapid visual review, and a separate object mask corresponding to each selected frame when masking is needed. Only the selected frames and their matching masks will be provided to COLMAP for feature extraction, feature matching, camera-pose estimation, and reconstruction. The team will compare more-dense and less-dense frame selections to identify an efficient balance between image overlap, viewpoint change, processing time, registration success, and reconstruction completeness.

## 7. Replacement Evaluation and Expected Challenges Section

Under **III. Methodology, I. Evaluation and Expected Challenges**, replace the current paragraph with:

> Evaluation will combine technical measurements and visual inspection. The team plans to compare image-registration ratio, sparse and dense point counts, mean reprojection error, dimensional error, mesh completeness, texture quality, and optimized-model performance. The two selected objects introduce different expected challenges. The unpainted pot may be affected by rotational symmetry and repeated circular geometry, while the glazed dragon jar may produce viewpoint-dependent highlights and reflections. The raised dragon decoration may provide additional local features, but repeated sections of the motif may also create ambiguous matches. Video-frame sampling density will be evaluated as another experimental variable because frames that are too similar waste processing time, while frames that are too far apart may reduce matching reliability.

## 8. Replacement Current Progress and Preliminary Analysis Section

Under **IV. Current Progress and Preliminary Analysis**, replace all three existing paragraphs with:

> The current reporting period produced several tangible outcomes. The team completed a broad online candidate survey, established a process for cross-checking marketplace claims against cultural sources, filtered out unsuitable or weakly supported products, and researched Thai pottery history, household uses, regional traditions, and cross-cultural influences. The team also consulted the course professor about using a dragon-motif vessel with Chinese origins and received approval to include it as a Thai-Chinese cultural form. The scope was expanded to at least two objects, and both selected pots have now been ordered.
>
> The two purchases create a deliberate comparison instead of duplicating the same object type. The first pot has an unpainted matte surface and carved ornamentation. The second has a glossy glaze, color variation, and a raised dragon design. This variation will allow the project to examine how surface reflectivity, texture, ornamentation, and symmetry influence feature matching and reconstruction quality. Cultural documentation is still in progress, particularly the exact name, production region, workshop tradition, and historical use of each purchased object.
>
> Technical preparation has also begun before the pots arrive. The team implemented a Python program that extracts individual image frames from 60 FPS video. The next development step is to add configurable frame sampling and dataset review so that highly similar adjacent frames can be excluded. The team plans to generate or organize masks only for the selected frames and then provide the selected image-and-mask sequence to COLMAP. Pilot tests will compare sampling intervals and determine whether video-based capture provides sufficient sharpness, overlap, and viewpoint change for reliable reconstruction.
>
> The immediate next milestone begins when the ordered pots arrive. The team will inspect their condition and surface behavior, record physical measurements, photograph reference views, verify remaining cultural information, and capture pilot videos or image sequences under controlled lighting. The initial COLMAP results will be used to revise the frame interval, masking method, camera path, lighting arrangement, and number of viewpoints. These procedures remain adjustable, and both successful and failed experiments will be recorded in later weekly progress reports.

## 9. References to Append

Append these entries after the current reference `[6]`:

> [7] Fine Arts Department, Ministry of Culture, Thailand, "The Origin of Ratchaburi Dragon Jars" (in Thai). [Online]. Available: https://www.finearts.go.th/promotion/view/36293-. Accessed: Aug. 20, 2026.
>
> [8] Sirindhorn Anthropology Centre, "Ratchaburi Dragon Jar," Traditional Objects of Everyday Use (in Thai). [Online]. Available: https://traditional-objects.sac.or.th/th/equipment-detail.php?ob_id=245. Accessed: Aug. 20, 2026.

## 10. Copy-Paste Map for Draft 2

1. **Abstract:** Replace paragraph 10, beginning `Abstract—This project investigates...`, with Section 1 of this file.
2. **Introduction:** Replace paragraph 16, beginning `The topic was partly inspired...`, with Section 2.
3. **Methodology A:** Replace paragraph 27, beginning `The project follows an experimental...`, with Section 3.
4. **Methodology B:** Replace paragraphs 33 through 46, beginning `During the current reporting period...` and ending `...accuracy evaluation.`, with Section 4.
5. **Figure 1:** Replace the current caption beginning `Fig. 1. One of the two unpainted...` with the first caption in Section 5.
6. **Figure 2:** Insert the new glossy-pot image and its caption immediately before `C. Equipment and Capture Area`.
7. **Methodology E:** Replace paragraph 59, beginning `A pilot dataset of approximately...`, with all three paragraphs in Section 6.
8. **Methodology I:** Replace paragraph 67, beginning `Evaluation will combine...`, with Section 7.
9. **Section IV:** Replace paragraphs 69 through 71, beginning `The current reporting period produced...` and ending `...weekly progress reports.`, with all four paragraphs in Section 8.
10. **References:** Add references `[7]` and `[8]` after the existing reference `[6]`.

## 11. Claims Still Requiring Later Verification

- Exact seller, workshop, province, and production method of the purchased dragon jar.
- Whether the product is genuinely connected to Ratchaburi production or only borrows the dragon-jar appearance.
- Thai name, region, tradition, and historical use of the first pot.
- Whether automatic or manually refined masks are more effective for the actual capture background.
- The final video-frame interval. It must be selected from pilot evidence rather than fixed in advance.

