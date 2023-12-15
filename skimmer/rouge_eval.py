import argparse
import logging
import os
import tempfile
from typing import Iterable

import sys
from pyrouge import Rouge155

from skimmer import logger
from skimmer.cnn_dm import CNN_DM
from skimmer.eval_common import ReferenceSummarySet



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
            rouge.log.setLevel(logging.WARN)
            rouge.model_dir = reference_dir
            rouge.system_dir = candidate_dir
            rouge.model_filename_pattern = 'ref.#ID#.txt'
            rouge.system_filename_pattern = r'cand.(\d+).txt'
            rouge_results = rouge.convert_and_evaluate()

            return rouge.output_to_dict(rouge_results)


def main(raw_args):
    parser = argparse.ArgumentParser(description='Evaluate ROUGE')
    parser.add_argument('--cnndm-dir', type=str,
                        help='Path to CNN/DailyMail dataset')
    parser.add_argument('--subset', type=float, default=1.0,
                        help='Proportion of test data to include')
    parser.add_argument('--work-dir', type=str,
                        help='Path to working directory')
    parser.add_argument('--')
    args = parser.parse_args(raw_args)

    cnn_dm = CNN_DM(args.cnn_dm_dir)
    ref_summary_sets = cnn_dm.read(CNN_DM.DataSplit.TEST, subset=args.subset)


if __name__ == '__main__':
    main(sys.argv)