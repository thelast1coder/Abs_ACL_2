# Related-work audit (as of 2026-11-11; verify every entry before citing)

| Work | What it did | Why no collision |
|---|---|---|
| KARL, 2604.22779 | GRPO abstention; Llama NQ truthful 38.7→42.7, false 55.3→16.7 | aggregate only; no frequency split; no threshold baseline. An ideal method to audit |
| Miao & Kearns, 2502.08666 (PNAS 2026) | monofacts, miscalibration, hallucination with bigram models on synthetic data and in-context tests | no real corpus counts; no abstention audit |
| Kalai et al., 2509.04664 (Nature 2026) | singleton bound; evaluations reward guessing | theory; we test it with exact counts and measure what abstention costs |
| Abstain-R1 2604.17073; TruthRL; I-CALM; GRAIT; Terven et al. 2610.13821; When Silence Is Golden (ICLR 2026) | abstention training and evaluation | all aggregate; none splits by frequency |
| R-Tuning (Zhang et al., NAACL 2024) | refusal-aware instruction tuning | a method we audit |
| Kandpal et al. 2023; Beyond Scaling (ACL 2026) | accuracy vs pretraining frequency in base models | no abstention |
| The Hallucination Tax of RFT, 2505.13988 | RL fine-tuning reduces refusal on unanswerable math | opposite direction; different object |
| Gekhman et al. 2024; Kaplan, Gekhman et al. COLM 2026 | new-knowledge fine-tuning causes hallucination and forgetting | mechanism neighbours; cite |
| Long-tail knowledge survey, 2602.16201 | taxonomy | no measurement |

**Sibling papers.** Cite these as anonymous concurrent work and state the difference:
- MB5 (sampler truncation);
- SNR (type priors, surprise-weighted floor).

**Weekly arXiv sweep terms:** "abstention long tail", "refusal frequency", "coverage by popularity", "selective prediction pretraining frequency".
