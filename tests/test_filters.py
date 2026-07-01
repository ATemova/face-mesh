import random
import statistics

from facemesh.filters import OneEuroFilter, PoseSmoother


def test_one_euro_reduces_noise():
    random.seed(0)
    f = OneEuroFilter(freq=30, min_cutoff=0.5, beta=0.01)
    raw, filt, t = [], [], 0.0
    for _ in range(60):
        t += 1 / 30
        x = 10.0 + random.gauss(0, 2.0)
        raw.append(x)
        filt.append(f(x, t))
    assert statistics.pstdev(filt[5:]) < statistics.pstdev(raw[5:])


def test_one_euro_tracks_step():
    f = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.5)
    t, y = 0.0, 0.0
    for _ in range(120):
        t += 1 / 30
        target = 0.0 if t < 2 else 40.0
        y = f(target, t)
    assert abs(y - 40.0) < 2.0


def test_pose_smoother_returns_triple():
    s = PoseSmoother()
    out = s.update(1.0, 2.0, 3.0, timestamp=0.1)
    assert len(out) == 3
