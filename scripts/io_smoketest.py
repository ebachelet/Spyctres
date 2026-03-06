import sys
import numpy as np
from Spyctres.io import read_spectrum, concatenate_segments

def summarize(seg):
    w = seg.wave[seg.mask]
    f = seg.flux[seg.mask]
    msg = [
        "N={0}".format(len(seg.wave)),
        "N_used={0}".format(int(np.sum(seg.mask))),
        "wave=[{0:.2f},{1:.2f}]".format(float(w.min()), float(w.max())),
        "finite_flux_frac={0:.4f}".format(float(np.mean(np.isfinite(seg.flux)))),
    ]
    if seg.err is not None:
        e = seg.err[seg.mask]
        msg.append("median_err={0:.4g}".format(float(np.median(e))))
    else:
        msg.append("err=None")
    return "  ".join(msg)

if len(sys.argv) < 2:
    print("Usage: python scripts/io_smoketest.py <file1> [file2 ...]")
    raise SystemExit(2)

segs = []
for p in sys.argv[1:]:
    s = read_spectrum(p, instrument="pepsi")
    segs.append(s)
    print(p)
    print(summarize(s))

if len(segs) > 1:
    joined = concatenate_segments(segs, sort=True, name="pepsi_joined")
    print("\nJOINED")
    print(summarize(joined))
