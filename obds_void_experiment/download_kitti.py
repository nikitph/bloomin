#!/usr/bin/env python3
"""
Download KITTI Odometry Dataset for LiDAR Void Detection
Downloads sequence 00 (velodyne point clouds) for testing
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def download_kitti_sample():
    """Download KITTI odometry sequence 00 (sample)"""
    
    # Create directory structure
    base_dir = Path("kitti_data")
    base_dir.mkdir(exist_ok=True)
    
    print("Downloading KITTI Odometry Dataset (Sequence 00)...")
    print("="*60)
    
    # KITTI odometry dataset URLs
    # Note: Full dataset is ~80GB, we'll download just sequence 00 (~1.5GB)
    urls = {
        'velodyne': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip',
        'calib': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip'
    }
    
    # Download velodyne data
    velodyne_zip = base_dir / "data_odometry_velodyne.zip"
    
    if not velodyne_zip.exists():
        print(f"\nDownloading velodyne data (~1.5GB)...")
        print("This may take several minutes...")
        
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end='', flush=True)
        
        try:
            urllib.request.urlretrieve(
                urls['velodyne'],
                velodyne_zip,
                reporthook=progress_hook
            )
            print("\n✓ Download complete!")
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            print("\nAlternative: Download manually from:")
            print("  https://www.cvlibs.net/datasets/kitti/eval_odometry.php")
            return False
    else:
        print("✓ Velodyne data already downloaded")
    
    # Extract
    print("\nExtracting data...")
    sequences_dir = base_dir / "sequences"
    
    if not (sequences_dir / "00" / "velodyne").exists():
        with zipfile.ZipFile(velodyne_zip, 'r') as zip_ref:
            # Extract only sequence 00 to save space
            members = [m for m in zip_ref.namelist() if m.startswith('dataset/sequences/00/')]
            for member in members:
                zip_ref.extract(member, base_dir)
        
        # Move to correct structure
        dataset_dir = base_dir / "dataset" / "sequences"
        if dataset_dir.exists():
            import shutil
            if sequences_dir.exists():
                shutil.rmtree(sequences_dir)
            shutil.move(str(dataset_dir), str(sequences_dir))
            shutil.rmtree(base_dir / "dataset")
        
        print("✓ Extraction complete!")
    else:
        print("✓ Data already extracted")
    
    # Verify
    seq_00_dir = sequences_dir / "00" / "velodyne"
    if seq_00_dir.exists():
        num_scans = len(list(seq_00_dir.glob("*.bin")))
        print(f"\n✓ Found {num_scans} LiDAR scans in sequence 00")
        print(f"\nDataset ready at: {base_dir.absolute()}")
        return True
    else:
        print("\n✗ Extraction failed")
        return False

if __name__ == "__main__":
    success = download_kitti_sample()
    
    if success:
        print("\n" + "="*60)
        print("KITTI Dataset Ready!")
        print("="*60)
        print("\nYou can now run:")
        print("  python3 lidar_void_detection.py")
    else:
        print("\n" + "="*60)
        print("Download Failed")
        print("="*60)
        print("\nPlease download manually from:")
        print("  https://www.cvlibs.net/datasets/kitti/eval_odometry.php")
        print("\nExtract to: kitti_data/sequences/00/velodyne/")
