# Open issues

Priorities are ordered from highest to lowest.

1. **High — Analyze the low-round truncation warning.** The full-run log says
   the computation domain is too small and would require at least 103 bits.
   The run completed successfully, but the numerical and security implications
   remain to be characterized.
2. **High — Define timing scopes precisely.** Document which phases are included
   in MP-SPDZ's 933.263-second time and which are included in the 15:48.40
   wall-clock measurement.
3. **High — Archive complete execution provenance.** Preserve the exact command
   context, environment details, preprocessing configuration, and complete
   logs in a controlled evidence archive without exposing secrets.
4. **Medium — Repeat across seeds.** Run additional pre-registered seeds to
   assess variation in loss, accuracy, runtime, and communication.
5. **Medium — Repeat runs for confidence.** Replicate selected seeds under the
   same controlled environment to quantify run-to-run variance.
6. **Medium — Strengthen baseline confirmation.** The Plain BP source and
   recorded seed-42 CSV values are repository evidence, but the original
   baseline execution was not rerun during this audit. Confirm its provenance
   and environment if a strict paired comparison is required.
7. **Low — Clarify evaluation policy.** Record whether epoch-5 evaluation is
   the intended fixed checkpoint for all future comparisons and how validation
   checkpoint selection should be handled.
8. **Low — Review backup retention.** Confirm the retention and access policy
   for the pre-run Player-Data backup, including secret-handling procedures.

