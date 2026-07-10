# Final Patient Setting

## Frozen Protocol

The final patient simulator is a controlled LLM-realized patient:

`PCV3.2 controller + Qwen3-8B realizer + verifier-repair + verified cache`

The controller decides what the patient is allowed to disclose at each turn.
The realizer writes the actual patient utterance. The verifier checks whether
the utterance obeys the controller decision. Repair requests are generated for
failed outputs. The final cache contains only verified or repaired verified
responses.

Final online evaluation records must use cached verified LLM responses:

- `patient_realizer_mode == "verified_llm_cache"`
- no rule fallback in final records
- no verifier hard errors

## Controller State

The controller tracks cross-turn state rather than treating each doctor question
independently:

- mapped symptom slots from the current doctor query
- already disclosed evidence
- withheld or forbidden evidence
- trust/readiness and engagement state
- refusal or avoidance history
- whether the doctor used direct, gentle, or follow-up questioning
- severity-conditioned disclosure constraints

The intent is to model a patient who may not reveal sensitive evidence
immediately, but can gradually open up when the doctor asks relevant and
supportive follow-up questions.

## Disclosure Levels

Each turn partitions evidence into three buckets:

- `retained`: may be stated directly.
- `weakened`: may be stated only vaguely or with softened certainty.
- `removed`: must not be disclosed in this response.

Severity changes the disclosure policy:

- `mild`: more willing to answer direct relevant questions.
- `moderate`: some avoidance and partial disclosure.
- `severe`: stronger avoidance, more sensitive evidence withheld, and slower
  opening under follow-up.

This does not mean the severe patient is permanently silent. After sufficient
trust, repeated relevant follow-up, or high-quality supportive questioning, the
controller can move evidence from withheld to weakened or retained.

## LLM Realizer

The realizer receives a structured request containing:

- dialogue history
- current doctor question
- allowed evidence
- weakened evidence
- forbidden evidence
- severity and controller state
- style constraints for patient-like speech

The realizer should answer as the patient, not as a clinician or summarizer. It
should be responsive to the doctor question while staying inside the allowed
evidence budget.

## Verifier-Repair Loop

The verifier checks hard safety and consistency constraints:

- grounding: no hallucinated symptoms or contradictions
- disclosure control: no forbidden evidence leakage
- response validity: the patient response is not empty, off-topic, or written as
  a doctor/system note
- severity policy: severe responses should not over-disclose sensitive evidence

Failed outputs are converted into repair requests that include the verifier
feedback. The repaired response is verified again. Final records should use the
verified cache; fallback is only a last-resort probe mechanism and should not
appear in final training/evaluation records.

## Rubric Judge

The verifier is a hard gate, not a quality score. A separate rubric judge can
score patient-simulation quality across:

- grounding
- disclosure control
- avoidance quality
- query responsiveness
- dialogue naturalness

Rubric results are used to validate the simulator, while evidence-recovery
metrics are used to evaluate doctor inquiry performance.
