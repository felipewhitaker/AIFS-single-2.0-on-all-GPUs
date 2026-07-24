import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Run an AIFS weather forecast.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--lead-time",
        type=int,
        default=24,
        help="Forecast horizon in hours (multiple of 6)",
    )
    p.add_argument(
        "--num-chunks",
        type=int,
        default=16,
        help="Attention chunks (increase to reduce VRAM usage)",
    )
    p.add_argument(
        "--fields",
        nargs="+",
        default=["2t", "msl", "tcw", "swh"],
        help="Fields to plot",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for output PNG files",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("../ic_cache"),
        help="Directory for IC .npz cache files",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download ICs even when a local cache exists",
    )
    p.add_argument(
        "--list-cache", action="store_true", help="Print cached IC dates and exit"
    )
    p.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    p.add_argument(
        "--dataset-repo",
        type=str,
        default=None,
        help="HF dataset repo id to upload results to, e.g. 'username/aifs-results'",
    )
    # Two mutually exclusive flags so users can actually make an upload public.
    privacy = p.add_mutually_exclusive_group()
    privacy.add_argument(
        "--private",
        dest="private",
        action="store_true",
        help="Create the dataset repo as private (default)",
    )
    privacy.add_argument(
        "--public",
        dest="private",
        action="store_false",
        help="Create the dataset repo as public",
    )
    p.set_defaults(private=True)
    return p.parse_args()


def main():
    args = parse_args()

    # Lazy import so --help is fast even without heavy deps installed
    import warnings

    warnings.filterwarnings("ignore")

    from aifs.initial_conditions import load_ics, list_cached
    from aifs.device import device_label
    from aifs.forecast import run_forecast
    import numpy as np
    import matplotlib.pyplot as plt
    from aifs.plot import plot_field

    # ── List cache and exit ────────────────────────────────────────────────────
    if args.list_cache:
        cached = list_cached(args.cache_dir)
        if not cached:
            print("No cached ICs found in", args.cache_dir)
        else:
            print(f"Cached ICs in {args.cache_dir}:")
            for path in cached:
                sz = path.stat().st_size / 1e6
                print(f"  • {path.stem.replace('ic_', '')}  ({sz:.0f} MB)")
        sys.exit(0)

    # ── Main forecast pipeline ─────────────────────────────────────────────────
    print("=" * 60)
    print("  AIFS Forecast Runner")
    print("=" * 60)
    print(f"  Device     : {device_label()}")
    print(f"  Lead time  : {args.lead_time} h")
    print(f"  Num chunks : {args.num_chunks}")
    print(f"  Fields     : {args.fields}")
    print(f"  Output dir : {args.output_dir}")
    print("=" * 60)

    # 1. Initial conditions
    fields, date = load_ics(
        cache_dir=args.cache_dir,
        force=args.force_download,
    )

    # 2. Run forecast
    states = run_forecast(
        fields,
        date,
        lead_time=args.lead_time,
        num_chunks=args.num_chunks,
    )

    print(f"\n✅  Forecast complete.  {len(states)} steps produced.")

    # 3. Save the results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = date.strftime("%Y%m%dT%H%M")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾  Saving {len(states)} states to {run_dir} ...")

    for state in states:
        step_label = state["date"].strftime("%Y%m%dT%H%M")

        # Raw field data, compressed — one .npz per output step
        npz_path = run_dir / f"state_{step_label}.npz"
        np.savez_compressed(
            npz_path,
            date=str(state["date"]),
            latitudes=state["latitudes"],
            longitudes=state["longitudes"],
            **{f"field_{k}": v for k, v in state["fields"].items()},
        )

        # Plots, one PNG per requested field per step
        if not args.no_plots:
            for field in args.fields:
                if field not in state["fields"]:
                    print(
                        f"  ⚠️  Field '{field}' not in forecast output, skipping plot."
                    )
                    continue
                try:
                    fig = plot_field(state, field)
                    fig.savefig(
                        run_dir / f"{field}_{step_label}.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close(fig)
                except Exception as e:
                    print(f"  ⚠️  Could not plot {field} at {step_label}: {e}")

    print(f"✅  Saved {len(states)} states to {run_dir}")

    # 4. Upload to a Hugging Face dataset repo
    if args.dataset_repo:
        from huggingface_hub import HfApi

        print(f"\n☁️   Uploading results to dataset repo: {args.dataset_repo}")
        api = HfApi()
        api.create_repo(
            args.dataset_repo,
            repo_type="dataset",
            exist_ok=True,
            private=args.private,
        )
        api.upload_folder(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            folder_path=str(run_dir),
            path_in_repo=f"forecasts/{run_id}",
        )
        print(
            f"✅  Uploaded → https://huggingface.co/datasets/{args.dataset_repo}/tree/main/forecasts/{run_id}"
        )


if __name__ == "__main__":
    main()
