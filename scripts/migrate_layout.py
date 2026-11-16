"""One-time, backed-up migration of support files; never moves raw data or runs."""
from pathlib import Path
from datetime import datetime
import ast
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
if (ROOT/'src/utils.py').exists() and not (ROOT/'utils.py').exists():
    print('Layout already migrated; no files changed.')
    raise SystemExit(0)
backup = ROOT/'run/layout_backup'/datetime.now().strftime('%Y%m%d_%H%M%S')
backup.mkdir(parents=True)
mapping = {}
for path in ROOT.glob('*.py'):
    mapping[path] = ROOT/('src' if path.name in ['utils.py','notebook_support.py'] else 'scripts')/path.name
for path in ROOT.glob('*.ps1'):
    mapping[path] = ROOT/'scripts'/path.name
for path in ROOT.glob('*.txt'):
    mapping[path] = ROOT/'environment'/path.name
for task in ['task1','task2','task3']:
    mapping[ROOT/task/(task+'.py')] = ROOT/task/'code'/(task+'.py')
for old, new in mapping.items():
    relative = old.relative_to(ROOT)
    target = backup/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old, target)
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)
for relative in ['README.md','AGENTS.md','assignments/Agent.md']:
    target = backup/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/relative, target)

bootstrap = '''"""Project paths for command-line tools; no model or training logic."""\nfrom pathlib import Path\nimport os\nimport sys\n\nROOT = Path(__file__).resolve().parents[1]\nfor folder in [ROOT, ROOT/"src", ROOT/"scripts"]:\n    sys.path.insert(0, str(folder))\nos.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT/"src"), str(ROOT), str(ROOT/"scripts"), os.environ.get("PYTHONPATH", "")])\n'''
(ROOT/'scripts/bootstrap.py').write_text(bootstrap, encoding='utf-8')
for path in (ROOT/'scripts').glob('*.py'):
    if path.name in ['bootstrap.py', 'migrate_layout.py']:
        continue
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    offset = 0
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            offset = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module=='__future__':
            offset = node.end_lineno
        else:
            break
    lines = source.splitlines(keepends=True)
    lines.insert(offset, '\nimport bootstrap  # Resolve src/ and the project root.\n')
    source = ''.join(lines)
    source = source.replace('Path(__file__).resolve().parent', 'Path(__file__).resolve().parents[1]')
    # The replacement above is for files formerly at the root only; normalize accidental plurals.
    source = source.replace('.parents[1]s[1]', '.parents[1]')
    source = source.replace('ROOT/"utils.py"', 'ROOT/"src/utils.py"').replace('ROOT / "utils.py"', 'ROOT / "src" / "utils.py"')
    source = source.replace("definition('utils.py',", "definition('src/utils.py',")
    source = source.replace("task+'/'+task+'.py'", "task+'/code/'+task+'.py'")
    source = source.replace('ROOT/task/(task+".py")', 'ROOT/task/"code"/(task+".py")')
    source = source.replace('ROOT/task/(task+".py")', 'ROOT/task/"code"/(task+".py")')
    source = source.replace('ROOT / task / (task + ".py")', 'ROOT / task / "code" / (task + ".py")')
    source = source.replace('task + "." + task', 'task + ".code." + task')
    source = source.replace('from task1.task1', 'from task1.code.task1').replace('from task2.task2', 'from task2.code.task2').replace('from task3.task3', 'from task3.code.task3')
    source = source.replace('(Path.cwd()/"utils.py").exists()', '(Path.cwd()/"src/utils.py").exists()')
    source = source.replace('sys.path.insert(0, str(ROOT))', 'sys.path.insert(0, str(ROOT))\n        sys.path.insert(0, str(ROOT/"src"))') if path.name=='presentation_notebooks.py' else source
    path.write_text(source, encoding='utf-8')
for path in [ROOT/'src/utils.py',ROOT/'src/notebook_support.py']:
    source=path.read_text(encoding='utf-8')
    source=source.replace('ROOT = Path(__file__).resolve().parent', 'ROOT = Path(__file__).resolve().parents[1]')
    source=source.replace('ROOT / "utils.py"', 'ROOT / "src" / "utils.py"')
    source=source.replace('ROOT / task / (task + ".py")', 'ROOT / task / "code" / (task + ".py")')
    path.write_text(source, encoding='utf-8')
for task in ['task1','task2','task3']:
    path=ROOT/task/'code'/(task+'.py')
    source=path.read_text(encoding='utf-8').replace('Path(__file__).resolve().parents[1]', 'Path(__file__).resolve().parents[2]')
    source=source.replace('(Path.cwd()/"utils.py").exists()', '(Path.cwd()/"src/utils.py").exists()')
    source=source.replace('sys.path.insert(0, str(PROJECT_ROOT))', 'sys.path.insert(0, str(PROJECT_ROOT))\nsys.path.insert(0, str(PROJECT_ROOT/"src"))')
    path.write_text(source,encoding='utf-8')
for path in (ROOT/'scripts').glob('*.ps1'):
    source=path.read_text(encoding='utf-8').replace('$PSScriptRoot','$projectRoot')
    source=source.replace('$ErrorActionPreference = "Stop"', '$ErrorActionPreference = "Stop"\n$projectRoot = Split-Path $PSScriptRoot -Parent',1)
    source=source.replace('"$Task\\$Task.py"','"$Task\\code\\$Task.py"')
    source=source.replace('"run_experiments.ps1"','"scripts\\run_experiments.ps1"')
    source=source.replace('-File run_experiments.ps1','-File scripts\\run_experiments.ps1')
    source=source.replace('-r requirements-cuda.txt','-r environment\\requirements-cuda.txt').replace('-r requirements.txt','-r environment\\requirements.txt')
    source=source.replace('& $python check_environment.py','& $python scripts\\check_environment.py')
    path.write_text(source,encoding='utf-8')
if (ROOT/'__pycache__').exists():
    (ROOT/'__pycache__').rename(backup/'root_pycache')
(backup/'moved_files.json').write_text(json.dumps({str(a.relative_to(ROOT)):str(b.relative_to(ROOT)) for a,b in mapping.items()},indent=2),encoding='utf-8')
print('Moved',len(mapping),'files. Backup:',backup)
