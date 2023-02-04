# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility module to call colabfold inference pipeline."""

import argparse
import logging
import sys

from colabfold.batch import get_queries, run
from colabfold.download import default_data_dir
from colabfold.utils import setup_logging
from pathlib import Path


def run_colabfold(args):
    """Invoke Colabfold inference pipeline."""
    if 'logging_setup' not in globals():
        setup_logging(Path(args.result_dir).joinpath("log.txt"))
        logging_setup = True

    queries, is_complex = get_queries(args.input_dir)
    run(
        queries=queries,
        result_dir=args.result_dir,
        use_templates=args.use_templates,
        use_amber=args.use_amber,
        msa_mode=args.msa_mode,
        model_type=args.model_type,
        num_models=args.num_models,
        num_recycles=args.num_recycles,
        model_order=args.model_order,
        is_complex=is_complex,
        data_dir=default_data_dir,
        keep_existing_results=args.do_not_overwrite_results,
        rank_by=args.rank_by,
        pair_mode=args.pair_mode,
        stop_at_score=args.stop_at_score,
        zip_results=args.zip_results,
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    """Parses the command line arguments"""
    # Create the parser
    parser = argparse.ArgumentParser(
        prog='ColabFold inference pipelin',
        description='ColabFold inference pipeline for AlphaFold2 and MMseq2.',
        allow_abbrev=False)

    parser.add_argument(
        '--input-dir',
        type=str,
        help='Full path to a locally mounted gcsfuse folder with the FASTA file.')
    parser.add_argument(
        '--result-dir',
        type=str,
        help='Full path to a locally mounted gcsfuse folder to store the results.')
    parser.add_argument(
        '--msa-mode',
        type=str,
        default='MMseqs2 (UniRef+Environmental)',
        help='MMseqs2 (UniRef+Environmental), MMseqs2 (UniRef only), single_sequence, custom')
    parser.add_argument(
        '--num-models',
        type=int,
        default=5,
        help='')
    parser.add_argument(
        '--num-recycles',
        type=int,
        default=3,
        help='')
    parser.add_argument(
        '--stop-at-score',
        type=int,
        default=100,
        help='')

    parser.add_argument(
        '--model-type',
        type=str,
        default='auto')
    parser.add_argument(
        '--rank-by',
        type=str,
        default='auto')
    parser.add_argument(
        '--pair-mode',
        type=str,
        default='unpaired+paired')

    parser.add_argument(
        '--model-order',
        nargs='*', 
        default=[3, 4, 5, 1, 2])

    parser.add_argument(
        '--use-custom-msa',
        type=str2bool, 
        nargs='?',
        const=True, 
        default=False,
        help='Default False')
    parser.add_argument(
        '--use-amber',
        type=str2bool, 
        nargs='?',
        const=True, 
        default=False,
        help='Default False')
    parser.add_argument(
        '--use-templates',
        type=str2bool, 
        nargs='?',
        const=True, 
        default=False,
        help='Default False')
    parser.add_argument(
        '--do-not-overwrite-results',
        type=str2bool, 
        nargs='?',
        const=True, 
        default=False,
        help='Default False')
    parser.add_argument(
        '--zip-results',
        type=str2bool, 
        nargs='?',
        const=True, 
        default=False,
        help='Default False')

    args = parser.parse_args()
    return args


if __name__=='__main__':
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s-%(message)s',
        level=logging.INFO,
        datefmt='%d-%m-%y %H:%M:%S',
        stream=sys.stdout)

    args = parse_arguments()

    run_colabfold(args)
