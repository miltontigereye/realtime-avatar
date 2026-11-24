#!/usr/bin/env python3
"""Freeze a local research build without changing the working tree or GPU server."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]

def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return {"command": args, "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build_id', help='Unique immutable build name, e.g. baseline-001 or compile-001')
    parser.add_argument('--note', default='')
    parser.add_argument('--result', action='append', default=[], help='Existing benchmark JSON to preserve (repeatable)')
    parser.add_argument('--artifact', action='append', default=[], help='Input, video, log or other evidence to preserve (repeatable)')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]*', args.build_id):
        parser.error('build_id must contain only letters, numbers, dots, underscores and hyphens')
    evidence = []
    for name in args.result + args.artifact:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            parser.error(f'Evidence must be an existing file inside the checkout: {name}')
        evidence.append(path)
    destination = ROOT / 'benchmarks' / 'builds' / args.build_id
    if destination.exists():
        parser.error(f'{destination} already exists; choose a new build_id to preserve history')
    paths = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT
    ).decode().split('\0')
    # Build artifacts and environment directories are never recursively included.
    paths = sorted({name for name in paths if name and
                    not name.startswith(('benchmarks/builds/', '.venv/', '.bootstrap/', '.git/'))
                    and (ROOT / name).is_file()})
    destination.mkdir(parents=True)
    files = {}
    with tarfile.open(destination / 'source.tar.gz', 'w:gz') as archive:
        for name in paths:
            archive.add(ROOT / name, arcname=name, recursive=False)
            files[name] = digest(ROOT / name)
    evidence_manifest = []
    for path in sorted(set(evidence)):
        relative = path.relative_to(ROOT)
        target = destination / 'evidence' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        evidence_manifest.append({"original_path": str(relative),
                                  "saved_path": str(target.relative_to(destination)),
                                  "sha256": digest(target), "bytes": target.stat().st_size})
    packages = sorted(f'{p.metadata["Name"]}=={p.version}' for p in importlib.metadata.distributions())
    (destination / 'packages.txt').write_text('\n'.join(packages) + '\n')
    (destination / 'working-tree.patch').write_text(subprocess.check_output(['git','diff','HEAD','--binary'],cwd=ROOT,text=True))
    environment = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'gpu': command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,power.limit','--format=csv']),
        'cpu': command(['lscpu']), 'memory': command(['free','-b']),
        'ffmpeg': command(['ffmpeg','-version']),
        'launch_environment': {key: os.environ.get(key) for key in
            ['HF_HOME','OMP_NUM_THREADS','HF_HUB_DISABLE_XET','PYTORCH_CUDA_ALLOC_CONF','CUDA_VISIBLE_DEVICES']},
    }
    try:
        import torch
        environment['torch'] = {'version': torch.__version__, 'cuda': torch.version.cuda,
                                'cudnn': torch.backends.cudnn.version(), 'cuda_available': torch.cuda.is_available()}
    except ImportError:
        environment['torch'] = None
    # Snapshot revisions and Hugging Face blob identities pin downloaded weights,
    # without copying tens of GB or reading any credentials.
    hub = Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface'))) / 'hub'
    snapshots = []
    for snapshot in sorted(hub.glob('models--*/snapshots/*')):
        items = []
        for path in sorted(snapshot.rglob('*')):
            if path.is_file():
                items.append({'file': str(path.relative_to(snapshot)), 'bytes': path.stat().st_size,
                              'blob_id': path.resolve().name if path.is_symlink() else None})
        snapshots.append({'repository_cache': snapshot.parent.parent.name,
                          'revision': snapshot.name, 'files': items})
    environment['cached_model_snapshots'] = snapshots
    (destination/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    results = []
    for name in args.result:
        path = (ROOT/name).resolve()
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not data.get('chunks'):
            continue
        chunks = data['chunks']
        render = sum(c['render_seconds'] for c in chunks)
        generated = chunks[-1]['generated_seconds']
        fps = {'aura-1.3b':16, 'aura-5b':25}.get(data.get('model'))
        results.append({'model':data.get('model'), 'source':data.get('source'),
                        'evidence':str(path.relative_to(ROOT)), 'chunks':len(chunks),
                        'width':data['video']['width'], 'height':data['video']['height'],
                        'generated_seconds':generated, 'sum_chunk_render_seconds':round(render,4),
                        'render_realtime_factor':round(render/generated,4),
                        'aggregate_nominal_generation_fps':round(generated*fps/render,4) if fps else None,
                        'target_fps':fps, 'sustained_realtime_demonstrated':False,
                        'browser_test_wall_seconds':data.get('seconds'),
                        'frames_played_before_completion':data.get('frame_before_done')})
    (destination/'results.json').write_text(json.dumps({'runs':results,
        'methodology':'Imported historical short-clip results, not a fresh run. Chunk times include audio features, rendering and muxer enqueue; exclude model/portrait setup and final muxer drain. Nominal frames include first-chunk padding. Browser wall time includes setup and playback waits and is not inference-only latency. These runs do not measure microphone-to-display latency or establish sustained real time.'},indent=2)+'\n')
    manifest = {'schema_version':1, 'build_id':args.build_id,
                'captured_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(), 'note':args.note,
                'upstream_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'git_status':command(['git','status','--short'])['stdout'],
                'source_sha256':files, 'evidence':evidence_manifest,
                'weights_included':False, 'environment_included':False}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    checksums = []
    for path in sorted(destination.rglob('*')):
        if path.is_file(): checksums.append(f'{digest(path)}  {path.relative_to(destination)}')
    (destination/'SHA256SUMS').write_text('\n'.join(checksums)+'\n')
    print(destination)
    for run in results:
        print(f'{run["model"]} ({run["source"]}): {run["aggregate_nominal_generation_fps"]:.2f} fps, RTF {run["render_realtime_factor"]:.2f}')

if __name__ == '__main__':
    main()
