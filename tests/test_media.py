import numpy as np
import pytest

from saresq.store.db import Store
from saresq.store.media import KIND_THERMAL_PATCH, MediaStore


@pytest.fixture()
def media(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    yield MediaStore(tmp_path / "media", store), store
    store.close()


def test_identical_bytes_are_stored_once_on_disk(media):
    ms, store = media
    a = ms.put_bytes(b"same-crop-bytes", "rgb_crop", target_id=None, priority=0.9)
    b = ms.put_bytes(b"same-crop-bytes", "rgb_crop", target_id=None, priority=0.5)

    assert a != b, "each reference gets its own row"
    assert store.get_media(a)["sha256"] == store.get_media(b)["sha256"]
    blobs = list((ms.root / "blobs").rglob("*.jpg"))
    assert len(blobs) == 1, "content-addressed storage must not duplicate the file"


def test_thermal_patch_is_exactly_1536_bytes_and_round_trips(media):
    ms, store = media
    # A real MLX90640 frame: 32x24, plausible Kolkata kelvin.
    rng = np.random.default_rng(0)
    kelvin = 305.0 + rng.normal(0, 2.5, size=(24, 32))
    mid = ms.put_thermal_patch(kelvin, target_id=None, priority=1.0)

    row = store.get_media(mid)
    assert row["bytes"] == 32 * 24 * 2 == 1536
    assert (row["width"], row["height"]) == (32, 24)

    back = ms.read_thermal_patch(mid)
    assert back.shape == kelvin.shape
    # Lossless to the 0.01 K quantisation step.
    assert np.max(np.abs(back - kelvin)) <= 0.005 + 1e-9


def test_thermal_patch_rejects_values_outside_the_storable_range(media):
    ms, _ = media
    with pytest.raises(ValueError, match="storable range"):
        ms.put_thermal_patch(np.full((4, 4), 700.0))
    with pytest.raises(ValueError, match="non-finite"):
        ms.put_thermal_patch(np.full((4, 4), np.nan))


def test_unknown_kind_is_rejected_before_touching_disk(media):
    ms, _ = media
    with pytest.raises(ValueError, match="unknown media kind"):
        ms.put_bytes(b"x", "screenshot")
    assert not list((ms.root / "blobs").rglob("*")), "nothing should have been written"


def test_pending_media_is_ordered_cheapest_and_most_confident_first(media):
    ms, store = media
    ms.put_bytes(b"clip-lo", "clip", priority=0.9)
    ms.put_bytes(b"crop-hi", "rgb_crop", priority=0.9)
    ms.put_bytes(b"crop-lo", "rgb_crop", priority=0.2)
    ms.put_bytes(b"thumb-lo", "thumb", priority=0.1)

    kinds = [(r["kind"], r["priority"]) for r in store.pending_media()]
    assert kinds == [
        ("thumb", 0.1),        # kind dominates: a thumbnail beats any crop
        ("rgb_crop", 0.9),     # then confidence within the kind
        ("rgb_crop", 0.2),
        ("clip", 0.9),
    ]


def test_thermal_patch_kind_constant_matches_the_schema_string():
    assert KIND_THERMAL_PATCH == "thermal_patch"
