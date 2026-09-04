"""SEW-based retargeting methods."""

from .legacy_adapter import solve_legacy_sew_mimic
from .stereo import (
    StereoSew,
    StereoSewInverseResult,
    StereoSewReference,
    StereoSewSingularityError,
)
from .gen3_geometry import (
    Gen3SewPoints,
    Gen3StereoSewGeometry,
    Gen3StructuralResiduals,
    MarginStatistics,
    ReferenceSearchResult,
    angular_margins,
    project_stereo_sew_reference,
    sample_gen3_configurations,
    select_project_reference,
)

__all__ = [
    "StereoSew",
    "StereoSewInverseResult",
    "StereoSewReference",
    "StereoSewSingularityError",
    "Gen3SewPoints",
    "Gen3StereoSewGeometry",
    "Gen3StructuralResiduals",
    "MarginStatistics",
    "ReferenceSearchResult",
    "angular_margins",
    "project_stereo_sew_reference",
    "sample_gen3_configurations",
    "select_project_reference",
    "solve_legacy_sew_mimic",
]
