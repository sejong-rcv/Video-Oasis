"""Ambiguity test definitions."""

from criteria.common.specs import DiagnosticSpec


AMBIGUITY_SPECS = {
    "consistency": DiagnosticSpec(
        name="consistency",
        axis="ambiguity",
        backend="postprocess",
        paper_name="Consistency Test",
        description="Flags samples where diagnostic models do not reach consensus.",
    ),
    "redundancy": DiagnosticSpec(
        name="redundancy",
        axis="ambiguity",
        backend="postprocess",
        paper_name="Redundancy Test",
        description="Flags samples answerable from arbitrary temporal chunks.",
    ),
    "sensitivity": DiagnosticSpec(
        name="sensitivity",
        axis="ambiguity",
        backend="manual",
        paper_name="Sensitivity Test",
        description="Manual restoration of frame-shuffle false positives.",
    ),
}
