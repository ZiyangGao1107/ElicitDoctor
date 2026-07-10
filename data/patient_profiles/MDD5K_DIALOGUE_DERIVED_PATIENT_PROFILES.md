# MDD-5K Dialogue-Derived Patient Profiles

Date: 2026-06-09

## Purpose

The released MDD-5K files contain generated conversation variants and diagnosis labels, not the original clinical case profiles. This step reconstructs a dialogue-derived patient profile for each case so that the patient simulator has a case-level hidden state.

## Summary

- Profiles: 744
- Average dialogue variants per profile: 5.0
- Average mapped turns per profile: 93.43
- Average observed active slots per profile: 17.69
- Average empty active slots per profile: 0.64
- Slot assignment policy: `primary_only`

## Tree Types

| Tree type | Profiles |
|---|---:|
| `female_adult` | 545 |
| `female_teen` | 175 |
| `male_adult` | 24 |

## Slot Coverage

| Slot | Profiles with observed evidence |
|---|---:|
| `hopelessness_or_crying` | 744 |
| `sleep` | 742 |
| `personality` | 741 |
| `dizziness_or_headache` | 737 |
| `attention_decline` | 737 |
| `family_psychiatric_history` | 735 |
| `palpitation` | 734 |
| `hallucination` | 733 |
| `romantic_status` | 733 |
| `appetite_loss` | 732 |
| `chest_tightness` | 731 |
| `memory_problem` | 729 |
| `cognitive_slowing` | 715 |
| `suicide_or_self_harm` | 709 |
| `mania_screen` | 700 |
| `binge_eating` | 616 |
| `work_status` | 616 |
| `menstrual_status` | 602 |
| `parent_awareness` | 197 |
| `school_or_study_status` | 180 |

## Example Profile Preview

### `mdd5k_1`

- Diagnosis: 抑郁状态
- ICD: F32.901
- Tree type: `female_teen`
- Observed slots: 19

| Slot | Evidence units | Example unit |
|---|---:|---|
| `hopelessness_or_crying` | 20 | 每次哭泣之后 |
| `school_or_study_status` | 20 | 成绩一直很差 |
| `parent_awareness` | 20 | 暴饮暴食？ |
| `suicide_or_self_harm` | 20 | 我也曾想过自杀 |
| `sleep` | 20 | 经常难以入睡 |
| `memory_problem` | 20 | 我总是记不住 |
| `menstrual_status` | 20 | 月经周期还算规律 |
| `hallucination` | 20 | 但从来没有幻听的经历 |

### `mdd5k_10`

- Diagnosis: 抑郁状态
- ICD: F32.901
- Tree type: `female_teen`
- Observed slots: 19

| Slot | Evidence units | Example unit |
|---|---:|---|
| `hopelessness_or_crying` | 20 | 悲观绝望 |
| `personality` | 20 | 虽然性格内向 |
| `school_or_study_status` | 20 | 比如考试失败或者被人批评 |
| `parent_awareness` | 20 | 他们知道我现在的情况 |
| `memory_problem` | 20 | 怎么都想不起来 |
| `cognitive_slowing` | 9 | 现在却要反复思考 |
| `appetite_loss` | 20 | 胃口很差 |
| `dizziness_or_headache` | 20 | 头痛不算太频繁 |

## Method Notes

- This is not an original MDD-5K clinical case profile.
- Evidence units are extracted from released patient utterances.
- The default construction uses only the primary mapped tree slot for each doctor turn to reduce cross-slot contamination.
- Each unit keeps source dialogue/turn ids and spans for auditability; full text remains in the mapping output.
- Online RL doctors ask natural-language questions; the simulator-side query interpreter maps each question to an internal tree node.
- The patient controller retrieves from `slot_profiles[simulator_internal_target_node]`; this node is hidden from the doctor policy.
