"""Safely execute the isolated Secure BP Matrix Prototype numerical audit.

Ensures:
1. Complete backup of MP-SPDZ Player-Data with permissions and symlinks.
2. Placement and SHA256 verification of exact matrix prototype inputs.
3. Execution of bp_matrix_numerical_audit.mpc with logging to results/mpc/bp/matrix_numerical_audit_seed42.log.
4. Guaranteed restoration of Player-Data in all circumstances (success or failure).
5. Post-restoration verification of original file hashes and permissions.
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MP_SPDZ_ROOT = Path(os.environ.get("MP_SPDZ_ROOT", "/home/abdelmonem/MP-SPDZ"))
PLAYER_DATA = MP_SPDZ_ROOT / "Player-Data"
BACKUP_DIR = MP_SPDZ_ROOT / "Player-Data-backup-audit"
INPUT_DIR = Path.home() / "secure-bp-runs/matrix_proto_seed42/inputs"
LOG_PATH = REPO_ROOT / "results/mpc/bp/matrix_numerical_audit_seed42.log"

EXPECTED_INPUT_HASHES = {
    "Input-P0-0": "d30e32c1e3ac3b6df5a790f1b5060edfafdd28ab6c4cb35188a88f528940900c",
    "Input-P1-0": "7150269c114364f2af55f881c4ec67de0a4e93800a9fe8af2d5597d0ba8ab275",
    "Input-P2-0": "9de5bdefd7805693afdc1240e8b78ac294f035b8ccef1bd36ff5f06d025b209b",
}


def get_manifest(directory):
    manifest = {}
    for p in sorted(directory.glob("*")):
        if p.is_symlink():
            manifest[p.name] = ("SYMLINK", str(p.readlink()), oct(p.lstat().st_mode))
        elif p.is_file():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest[p.name] = ("FILE", digest, oct(p.stat().st_mode))
        elif p.is_dir():
            manifest[p.name] = ("DIR", p.name, oct(p.stat().st_mode))
    return manifest


def main():
    if not PLAYER_DATA.is_dir():
        sys.exit(f"Player-Data does not exist: {PLAYER_DATA}")
    if not INPUT_DIR.is_dir():
        sys.exit(f"Input directory does not exist: {INPUT_DIR}")

    print("Step 1: Inspecting current Player-Data...")
    initial_manifest = get_manifest(PLAYER_DATA)
    print(f"Initial Player-Data entries: {len(initial_manifest)}")

    print(f"Step 2: Creating full backup to {BACKUP_DIR}...")
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(PLAYER_DATA, BACKUP_DIR, symlinks=True)
    backup_manifest = get_manifest(BACKUP_DIR)
    if backup_manifest != initial_manifest:
        sys.exit("CRITICAL: Backup manifest does not match initial manifest!")
    print("Backup verified.")

    try:
        print("Step 3: Staging matrix prototype inputs...")
        for name, expected_hash in EXPECTED_INPUT_HASHES.items():
            src = INPUT_DIR / name
            dst = PLAYER_DATA / name
            src_hash = hashlib.sha256(src.read_bytes()).hexdigest()
            if src_hash != expected_hash:
                raise ValueError(f"Source {name} hash {src_hash} != {expected_hash}")
            shutil.copy2(src, dst)
            dst_hash = hashlib.sha256(dst.read_bytes()).hexdigest()
            if dst_hash != expected_hash:
                raise ValueError(f"Staged {name} hash {dst_hash} != {expected_hash}")
            print(f"  {name}: {dst_hash} (VERIFIED)")

        print("Step 4: Running isolated MPC diagnostic...")
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["MP_SPDZ_ROOT"] = str(MP_SPDZ_ROOT)
        env["MP_SPDZ_PYTHON"] = "python3"

        cmd = [
            str(REPO_ROOT / "scripts/run_mpc.sh"),
            "bp_matrix_numerical_audit.mpc",
            str(LOG_PATH),
        ]
        proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            print(f"MPC run exited with non-zero status: {proc.returncode}", file=sys.stderr)
        else:
            print(f"MPC run completed successfully. Log saved to: {LOG_PATH}")

    finally:
        print("Step 5: Restoring Player-Data from backup...")
        # Remove current Player-Data contents and restore from backup
        for item in PLAYER_DATA.glob("*"):
            if item.is_symlink() or item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        for src_item in BACKUP_DIR.glob("*"):
            dst_item = PLAYER_DATA / src_item.name
            if src_item.is_symlink():
                link_target = os.readlink(src_item)
                os.symlink(link_target, dst_item)
            elif src_item.is_file():
                shutil.copy2(src_item, dst_item)
            elif src_item.is_dir():
                shutil.copytree(src_item, dst_item, symlinks=True)

        print("Step 6: Verifying restored Player-Data state...")
        restored_manifest = get_manifest(PLAYER_DATA)
        if restored_manifest != initial_manifest:
            print("ERROR: Restored manifest does not match initial manifest!", file=sys.stderr)
            diff_keys = set(restored_manifest.keys()) ^ set(initial_manifest.keys())
            if diff_keys:
                print(f"Key differences: {diff_keys}", file=sys.stderr)
            for k in set(restored_manifest.keys()) & set(initial_manifest.keys()):
                if restored_manifest[k] != initial_manifest[k]:
                    print(f"Entry difference for {k}: {restored_manifest[k]} vs {initial_manifest[k]}", file=sys.stderr)
            sys.exit(1)
        else:
            print("Player-Data state successfully and exactly restored to initial state.")
            shutil.rmtree(BACKUP_DIR)
            print("Backup directory cleaned up.")


if __name__ == "__main__":
    main()
