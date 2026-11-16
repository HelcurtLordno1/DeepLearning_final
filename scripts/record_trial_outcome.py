"""Record the observed interruption; never infer an unmeasured validation score."""

import bootstrap  # Resolve src/ and the project root.
import json
from utils import ROOT, save_json
prefix = ROOT/'run/safety/20261116_221420_119'
readings = [json.loads(line.lstrip('\ufeff')) for line in prefix.with_name(prefix.name+'_telemetry.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
folder = ROOT/'run/task3/attention_position_finetune_recovery_1epochs/42/20261116_221430_082488'
metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
metadata.update({'status': 'interrupted_monitor_lost', 'validation_evaluated': False,
                 'interruption': 'Python detected that its PowerShell monitor had stopped; no completed epoch validation score.'})
save_json(folder/'metadata.json', metadata)
save_json(prefix.with_name(prefix.name+'_outcome.json'), {
    'status': 'interrupted_monitor_lost', 'tool_session_exit_code': 143,
    'windows_boot_after_run': '2026-11-16T21:02:54.5000000+07:00',
    'project_training_processes_after_stop': 0,
    'maximum_observed_gpu_temperature_c': max(r['gpu_temperature_c'] for r in readings),
    'maximum_observed_gpu_memory_mib': max(r['gpu_memory_mib'] for r in readings),
    'minimum_observed_free_ram_gib': min(r['free_ram_gib'] for r in readings),
    'minimum_observed_free_commit_gib': min(r['free_commit_gib'] for r in readings),
    'telemetry_samples': len(readings), 'last_telemetry_time': readings[-1]['time'],
    'new_validation_score_available': False, 'test_evaluated': False,
    'automatic_retry': False, 'run': str(folder.relative_to(ROOT))})
print('Recorded incomplete trial. No score invented; no automatic retry.')
