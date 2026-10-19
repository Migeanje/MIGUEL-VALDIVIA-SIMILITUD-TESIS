"""Evaluation: clustering metrics, coherence, PR/F1, kappa, temporal split, and Likert analysis.

The metrics are pure functions. Each methodological parameter is an argument with no default,
declared by the experiment design (T18) before any run. When a metric has no value for a valid
input, it is ``None`` and the result's ``undefined_reason`` says why; an invalid input raises.
"""
