import pytest

from sew_mimic.pipeline import sample_frame_indices


def test_frame_sampling_is_deterministic_and_preserves_source_indices():
    expected = (2, 5, 8)
    assert sample_frame_indices(20, start_frame=2, max_frames=3, stride=3) == expected
    assert sample_frame_indices(20, start_frame=2, max_frames=3, stride=3) == expected


@pytest.mark.parametrize(
    "kwargs",
    (
        {"total_frames": 0},
        {"total_frames": 4, "start_frame": 4},
        {"total_frames": 4, "stride": 0},
        {"total_frames": 4, "max_frames": 0},
    ),
)
def test_frame_sampling_rejects_empty_or_invalid_selections(kwargs):
    with pytest.raises(ValueError):
        sample_frame_indices(**kwargs)
