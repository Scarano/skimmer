import argparse
import logging
import os
import tempfile
from typing import Iterable

import sys

import random
from pyrouge import Rouge155
from tqdm import tqdm

from skimmer import logger
from skimmer.cnn_dm import CNN_DM
from skimmer.config import build_config_dict
from skimmer.eval_common import ReferenceSummarySet
from skimmer.abridger_builder import build_abridger_from_config


def rouge_eval(ref_summary_sets: Iterable[ReferenceSummarySet], candidate_summaries: Iterable[str]):

    with tempfile.TemporaryDirectory() as temp_dir:

        reference_dir = os.path.join(temp_dir, "reference")
        os.mkdir(reference_dir)
        candidate_dir = os.path.join(temp_dir, "candidate")
        os.mkdir(candidate_dir)

        for i, (ref_summary_set, candidate_summary) in enumerate(zip(ref_summary_sets, candidate_summaries)):
            if len(ref_summary_set.summaries) < 1:
                logger.warning("No summaries found in ref summary set '%s...'",
                               ref_summary_set.doc[:40])
                continue
            if len(ref_summary_set.summaries) > 1:
                logger.warning("Multiple summaries found in ref summary set '%s...' "
                               "-- this is not currently supported.",
                               ref_summary_set.doc[:40])

            with open(os.path.join(reference_dir, f"ref.{i}.txt"), 'w', encoding='utf-8') as f:
                f.write(ref_summary_set.summaries[0])
            with open(os.path.join(candidate_dir, f"cand.{i}.txt"), 'w', encoding='utf-8') as f:
                f.write(candidate_summary)

        rouge = Rouge155()
        rouge.log.setLevel(logging.DEBUG)
        rouge.model_dir = reference_dir
        rouge.system_dir = candidate_dir
        rouge.model_filename_pattern = 'ref.#ID#.txt'
        rouge.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = rouge.convert_and_evaluate()

        results_dict = rouge.output_to_dict(rouge_results)

        return results_dict


def main(raw_args):
    parser = argparse.ArgumentParser(description='Evaluate ROUGE')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('-o', '--override', action='append',
                        help='Configuration override. Use as "-o <key>=<yaml-value>"')
    parser.add_argument('--work-dir', type=str, required=True,
                        help='Path to working directory')
    parser.add_argument('--cnn-dm-dir', type=str, required=True,
                        help='Path to CNN/DailyMail dataset')
    parser.add_argument('--split', type=str, default='validation',
                        help='Dataset split to use (test or validation)')
    parser.add_argument('--subset', type=float, default=1.0,
                        help='Proportion of test data to include')
    args = parser.parse_args(raw_args)

    config = build_config_dict(args.config, args.override)

    abridger = build_abridger_from_config(config, args.work_dir)

    cnn_dm_dir = CNN_DM(args.cnn_dm_dir)
    split = CNN_DM.DataSplit.of(args.split)
    ref_summary_sets = list(cnn_dm_dir.read(split, subset=args.subset))
    # shuffle the order of the ref_summary_sets so that parallel runs
    # process and cache results for different documents.
    random.shuffle(ref_summary_sets)
    logger.info("Loaded %d docs....", len(ref_summary_sets))

    summaries = []
    for ref_summary in tqdm(ref_summary_sets):
        summaries.append(abridger.abridge(ref_summary.doc))

    print(rouge_eval(ref_summary_sets, summaries))


if __name__ == '__main__':
    main(sys.argv[1:])