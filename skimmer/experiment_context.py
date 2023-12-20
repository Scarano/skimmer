import hashlib
import subprocess
import os
from datetime import datetime

from pydantic import BaseModel


class ExperimentContext(BaseModel):
    start: str
    code_dir: str
    commit_hash: str
    diff_hash: str
    code_state_id: str
    work_dir: str
    dataset_dir: str
    dataset: str
    dataset_split: str
    dataset_subset: float
    experiment_name: str


def make_experiment_context(
        work_dir: str, dataset_dir: str,
        dataset_name: str, dataset_split: str, dataset_subset: float):

    run_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code_dir)\
            .decode('utf-8').strip()
    except subprocess.CalledProcessError:
        # Not running from a git repo
        # TODO: use version number instead?
        commit_hash = "?"

    try:
        diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=code_dir)\
            .decode('utf-8')
        diff_hash = hashlib.sha1(diff.encode('utf-8')).hexdigest()
    except subprocess.CalledProcessError:
        diff_hash = "?"

    code_state_id = f"{commit_hash[:6]}-{diff_hash[:5]}"

    dataset_split_short = 'val' if dataset_split == 'validation' else dataset_split

    if dataset_subset < 1.0:
        subset_suffix = ''
    else:
        subset_suffix = str(dataset_subset)[1:]
    dataset_desc = f"{dataset_name}-{dataset_split_short}{subset_suffix}"

    return ExperimentContext(
        start=run_start,
        code_dir=code_dir,
        commit_hash=commit_hash,
        diff_hash=diff_hash,
        code_state_id=code_state_id,
        work_dir=work_dir,
        dataset_dir=dataset_dir,
        dataset=dataset_desc,
        dataset_split=dataset_split,
        dataset_subset=dataset_subset,
        experiment_name=f"{code_state_id}-{dataset_desc}"
    )

