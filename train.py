"""Fetch XAUUSD history and train a direction model per timeframe.

Usage:  python train.py
"""
import os
import time

import data
import ml


def main():
    for tf, cfg in data.TFS.items():
        t0 = time.time()
        try:
            d = data.get_candles(tf, force=True)
            candles = d.get("candles")
            if not candles:
                print(f"[{tf}] no data ({d.get('error')}) — skipped")
                continue
            model = ml.train_from_candles(candles)
            model.save(os.path.join("models", f"{tf}.json"))
            m = model.d
            print(f"[{tf}] {m['n']:>6} samples · source {d['source'][:40]} · "
                  f"train {m['train_acc']:.1%} · test {m['test_acc']:.1%} · "
                  f"balanced {m['test_bal_acc']:.1%} · {time.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"[{tf}] failed: {e}")


if __name__ == "__main__":
    main()
