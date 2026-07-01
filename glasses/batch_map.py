import subprocess
import argparse
import json
from pathlib import Path
import concurrent.futures
import re
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


def should_process_instance(instance_id: str, args) -> bool:
    return not (args.output_dir / instance_id / f"mapping_seed_{args.seed}.json").exists()


def should_process_repo(instance_id: str, args) -> bool:
    bundle_root = args.output_dir / instance_id.split("__")[0]
    index_path = bundle_root / "index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            return True
        variants = index.get("variants", [])
        expected_seeds = [args.seed + idx for idx in range(args.variant_count)]
        variant_seeds = sorted(item.get("seed") for item in variants)
        return (
            index.get("level") != args.mapping_mode
            or len(variants) < args.variant_count
            or variant_seeds != expected_seeds
        )

    meta_path = bundle_root / "mapping_meta.json"
    if not meta_path.exists():
        return True
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return True
    return (
        meta.get("seed") != args.seed
        or meta.get("level") != args.mapping_mode
        or args.variant_count != 1
    )


def get_target_instance_ids(args):
    raw_instance_ids = [line.strip() for line in args.filter_file.read_text().splitlines() if line.strip()]
    if args.mapping_mode == "identity_only":
        return [iid for iid in raw_instance_ids if should_process_instance(iid, args)], "instances"

    repo_representatives = {}
    for iid in raw_instance_ids:
        repo_name = iid.split("__")[0]
        repo_representatives.setdefault(repo_name, iid)
    return [iid for iid in repo_representatives.values() if should_process_repo(iid, args)], "repos"

def run_single_instance(instance_id: str, args):
    try:
        cost = 0.0
        for variant_index in range(args.variant_count):
            cmd = [
                "python", str(Path(__file__).with_name("main.py")),
                "--instance-id", instance_id,
                "--seed", str(args.seed + variant_index),
                "--workers", str(args.extraction_workers),
                "--repo-root", str(args.repo_root),
                "--output-dir", str(args.output_dir),
                "--mapping-mode", args.mapping_mode,
                "--variant-index", str(variant_index),
                "--variant-count", str(args.variant_count),
            ]

            if args.model:
                cmd.extend(["--model", args.model])

            if args.api_base:
                cmd.extend(["--api-base", args.api_base])
            if args.api_key:
                cmd.extend(["--api-key", args.api_key])

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            match = re.search(r"FINAL_COST: ([\d\.]+)", result.stdout)
            if match:
                cost += float(match.group(1))
        return True, instance_id, cost, "Success"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        return False, instance_id, 0.0, error_msg

def main():
    parser = argparse.ArgumentParser(description="Parallel batch generate semantic mappings with cost tracking.")
    parser.add_argument("--filter-file", type=Path, required=True, help="Path to instance_ids.txt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--instance-workers", type=int, default=4, help="Number of instances to process in parallel")
    parser.add_argument("--extraction-workers", type=int, default=8, help="Workers per instance for file scanning")
    parser.add_argument("--model", type=str, default=None, help="Model for token mapping")
    parser.add_argument("--api-base", type=str, help="API base URL")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--repo-root", type=Path, default=Path("./repos"), help="Root directory of repos")
    parser.add_argument("--output-dir", type=Path, default=Path("../output/semantic_mappings"), help="Output directory")
    parser.add_argument("--variant-count", type=int, default=1, help="Number of repo-level bundle variants to generate")
    parser.add_argument(
        "--mapping-mode",
        choices=["identity_only", "namespace_l2", "identity_namespace_l2"],
        default="identity_only",
        help="Offline bundle generation mode.",
    )
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="Backward-compatible alias for --mapping-mode identity_only.",
    )

    args = parser.parse_args()
    if args.identity_only:
        args.mapping_mode = "identity_only"

    if not args.filter_file.exists():
        print(f"Error: Filter file {args.filter_file} not found.")
        return

    instance_ids, unit_label = get_target_instance_ids(args)
    total = len(instance_ids)

    print(f"🚀 Starting parallel batch mapping for {total} {unit_label}...")
    print(f"   Instance Parallelism: {args.instance_workers}")
    print(f"   Extraction Parallelism (per instance): {args.extraction_workers}")

    total_cost = 0.0
    success_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"[cyan]Processing {unit_label}...", total=total)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.instance_workers) as executor:
            future_to_id = {executor.submit(run_single_instance, iid, args): iid for iid in instance_ids}
            
            for future in concurrent.futures.as_completed(future_to_id):
                success, iid, cost, msg = future.result()
                total_cost += cost
                if success:
                    success_count += 1
                    progress.console.print(f"[green]✔[/green] {iid} completed. Cost: [yellow]${cost:.4f}[/yellow]")
                else:
                    progress.console.print(f"[red]✘[/red] {iid} failed: {msg.strip().splitlines()[-1] if msg.strip() else 'Unknown error'}")
                
                progress.advance(task)
                progress.update(task, description=f"[cyan]Processing... Total Cost: [bold yellow]${total_cost:.4f}[/bold yellow]")

    print(f"\n✨ Batch processing finished!")
    print(f"   Total {unit_label.title()}: {total}")
    print(f"   Succeeded: {success_count}")
    print(f"   Failed: {total - success_count}")
    print(f"   [bold green]Total Cost: ${total_cost:.4f}[/bold green]")

if __name__ == "__main__":
    main()
