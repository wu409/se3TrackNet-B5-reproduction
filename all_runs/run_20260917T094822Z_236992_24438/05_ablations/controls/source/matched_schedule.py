"""Read a GT-free trigger schedule exported by a completed matching full run."""
import csv
from pathlib import Path


def load_schedule(folder, episode, frame_ids):
    files = list(Path(folder).glob('checkpoint2_per_frame_' + episode + '_log_threshold*.csv'))
    if len(files) != 1:
        raise ValueError('Missing/ambiguous matching full episode: ' + episode)
    with files[0].open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [int(row['frame_id']) for row in rows]
    if ids != list(map(int, frame_ids)) or len(set(ids)) != len(ids):
        raise ValueError('Reference full schedule/frame order mismatch: ' + episode)
    for row in rows:
        if row['policy_version'] != 'four_mode_relocalization_v2' or row['policy_variant'] != 'full':
            raise ValueError('Schedule requires a four-mode FULL result')
        if row['relocalization_attempted'] not in ('0', '1'):
            raise ValueError('Invalid trigger indicator')
    # No error, GT, or success field enters the control schedule.
    return {int(row['frame_id']): row['relocalization_attempted'] == '1' for row in rows}
