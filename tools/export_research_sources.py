"""Export readable research source without weights, data or generated artifacts."""
from pathlib import Path
import argparse, collections, hashlib, json, os, subprocess, zipfile

EXTENSIONS = set('.py .pyi .pyx .pxd .c .cc .cpp .cxx .cu .cuh .h .hpp .metal .cmake .sh .bash .zsh .bat .ps1 .slurm .js .jsx .ts .tsx .mjs .cjs .html .css .scss .vue .svelte .yaml .yml .toml .ini .cfg .xml .proto .patch .diff .in .md .mdx .rst .txt .ipynb .dockerfile .example'.split())
SKIP_DIRS = set('.git .hg .svn .venv venv env node_modules __pycache__ .pytest_cache .mypy_cache .ruff_cache .cache .tox .bootstrap .agents .claude .github .buildkite .idea .vscode AURA-Dataset web_sessions outputs output logs wandb checkpoints weights .next .nuxt dist build builds site-packages'.split())
SKIP_FILES = {'AGENTS.md', 'CLAUDE.md', 'uv.lock', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', '.DS_Store'}
NAMES = {'LICENSE', 'COPYING', 'NOTICE', 'Makefile', 'Dockerfile', 'Containerfile', 'CMakeLists.txt', 'MANIFEST.in', 'Justfile', 'requirements', 'VERSION', 'uv-excludes'}

def git(p, *args):
    r = subprocess.run(['git', '-C', str(p), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source_root', type=Path, help='Local RealTime-Avatar project root')
    ap.add_argument('destination', type=Path, help='New empty source export directory')
    args = ap.parse_args()
    source = args.source_root.resolve() / 'research'
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    manifest = {'purpose': 'Full locally available readable source; excludes weights, datasets, media, environments, VCS metadata and generated outputs.',
                'limitations': ['External dependencies and uninitialized submodules are not vendored.',
                               'Working-copy source can differ from the upstream revision.',
                               'SparkDiffusion/sparkdiffusion is physically stored in the adjacent sparkdiffusion directory.',
                               'AURA deployed optimized excerpts remain separately under research-context/sources/AURA-deployed.'],
                'projects': {}, 'files': {}}
    for project in sorted(source.iterdir()):
        if not project.is_dir():
            continue
        meta = {'revision': git(project, 'rev-parse', 'HEAD') if (project/'.git').exists() else None,
                'upstream': git(project, 'remote', 'get-url', 'origin') if (project/'.git').exists() else None,
                'files': 0, 'bytes': 0, 'excluded_by_reason': {}, 'submodules': []}
        if (project/'.gitmodules').is_file():
            entries = git(project, 'config', '-f', '.gitmodules', '--get-regexp', r'submodule\..*\.(path|url)')
            meta['submodule_declarations'] = entries.splitlines() if entries else []
        if (project/'.git').exists():
            for line in (git(project, 'ls-files', '--stage') or '').splitlines():
                if line.startswith('160000 '):
                    head, path = line.split('\t',1)
                    meta['submodules'].append({'path':path, 'commit':head.split()[1], 'available_locally':(project/path/'.git').exists()})
        excluded = collections.Counter()
        for base, dirs, files in os.walk(project, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.endswith(('.egg-info','.dist-info')) and not (Path(base)/d).is_symlink())
            for name in sorted(files):
                p = Path(base)/name
                rel = p.relative_to(project)
                suffix = p.suffix.lower()
                if p.is_symlink() or name.startswith('.') or name in SKIP_FILES:
                    excluded['metadata_or_symlink'] += 1
                    continue
                allowed = (p.stat().st_size == 0 and 'resource' in rel.parts) or suffix in EXTENSIONS or name in NAMES or name.startswith(('LICENSE', 'COPYING', 'NOTICE', 'Dockerfile'))
                # JSON is retained for configuration and dependency declarations, not datasets/results.
                if suffix == '.json':
                    allowed = (any('config' in part.lower() for part in rel.parts) or name in {'package.json','tsconfig.json','jsconfig.json','composer.json'})
                if not allowed:
                    excluded['non_source_or_data_format'] += 1
                    continue
                if suffix == '.txt' and not (name.lower().startswith(('requirements','license','notice','copying','third_party')) or name == 'CMakeLists.txt'):
                    excluded['text_data_or_generated_text'] += 1
                    continue
                data = p.read_bytes()
                try:
                    data.decode('utf-8')
                except UnicodeDecodeError:
                    excluded['binary'] += 1
                    continue
                if b'\0' in data:
                    excluded['binary'] += 1
                    continue
                if suffix == '.ipynb':
                    notebook = json.loads(data)
                    for cell in notebook.get('cells', []):
                        if cell.get('cell_type') == 'code':
                            cell['outputs'] = []
                            cell['execution_count'] = None
                    data = (json.dumps(notebook, ensure_ascii=False, indent=1)+'\n').encode()
                target = Path(project.name)/rel
                out = dest/target
                out.parent.mkdir(parents=True,exist_ok=True)
                out.write_bytes(data)
                manifest['files'][target.as_posix()] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
                meta['files'] += 1
                meta['bytes'] += len(data)
        meta['excluded_by_reason'] = dict(excluded)
        manifest['projects'][project.name] = meta
    # The adjacent implementation shares SparkDiffusion's upstream license.
    shared = dest/'sparkdiffusion/LICENSE.txt'
    if not shared.exists():
        data = (source/'SparkDiffusion/LICENSE.txt').read_bytes()
        shared.write_bytes(data)
        manifest['files']['sparkdiffusion/LICENSE.txt'] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        manifest['projects']['sparkdiffusion']['files'] += 1
        manifest['projects']['sparkdiffusion']['bytes'] += len(data)
    (dest/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name, meta in manifest['projects'].items():
        print(name,meta['files'],'files,',meta['bytes'],'bytes')
    print('Total:',len(manifest['files']),'source files,',sum(x['bytes'] for x in manifest['files'].values()),'bytes')

if __name__ == '__main__':
    main()
