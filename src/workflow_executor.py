# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Execute a workflow and print the execution results."""
import json
import os
import typing as T

from datetime import datetime
from pathlib import Path

from google.cloud import firestore
from google.cloud import storage
from google.cloud import workflows_v1beta
from google.cloud.workflows import executions_v1beta


def prepare_args_for_experiment(
    project_id: str,
    region: str,
    input_dir: str,
    image_uri: str,
    job_gcs_path: str,
    labels: dict,
    machine_type: str = 'n1-standard-4',
    cpu_milli: int = 8000,
    memory_mib: int = 30000,
    boot_disk_mib: int = 200000,
    gpu_type: str = "nvidia-tesla-t4",
    gpu_count: int = 1,
    colabfold_machine_type: str = 'n1-standard-4-t4',
    job_gcsfuse_local_dir: str = '/mnt/disks/gcs/colabfold',
    parallelism: int = 8,
    use_templates: bool = False,
    use_amber: bool = False,
    msa_mode: str = 'MMseqs2 (UniRef+Environmental)',
    model_type: str = 'auto',
    num_models: int = 5,
    num_recycles: int = 3,
    model_order: list = [3, 4, 5, 1, 2],
    use_custom_msa: bool = False,
    do_not_overwrite_results: bool = False,
    rank_by: str = 'auto',
    pair_mode: str = 'unpaired+paired',
    stop_at_score: int = 100,
    zip_results: bool = False
) -> T.Dict:

    storage_client = storage.Client()
    db = firestore.Client()
    
    args = {}
    args['project_id'] = project_id
    args['region'] = region
    args['image_uri'] = image_uri
    args['colabfold_machine_type'] = colabfold_machine_type
    args['job_gcs_path'] = job_gcs_path
    args['parallelism'] = parallelism
    args['job_gcsfuse_local_dir'] = job_gcsfuse_local_dir
    args['machine_type'] = machine_type
    args['cpu_milli'] = cpu_milli
    args['memory_mib'] = memory_mib
    args['boot_disk_mib'] = boot_disk_mib
    args['gpu_type'] = gpu_type
    args['gpu_count'] = gpu_count

    # List all blobs in a GCS bucket/folder
    # Append blob information to a list
    if input_dir.endswith('/'):
        gcs_prefix = '/'.join(input_dir.split(sep='/')[1:])
    else:
        gcs_prefix = '/'.join(input_dir.split(sep='/')[1:]) + '/'

    blobs_listing = storage_client.list_blobs(
        job_gcs_path,
        prefix=gcs_prefix,
        delimiter='/')

    # Get only fasta files
    blobs = []
    for blob in blobs_listing:
        if Path(blob.name).suffix == '.fasta':
            blobs.append(blob)

    runners = []
    start_time = datetime.now().isoformat()

    doc_ref = db.collection('colabfold-experiments')

    input_bucket_name = input_dir.split(sep='/')[0]

    for blob in blobs:
        runner_args = {}
        exp_doc_ref = doc_ref.document()

        runner_args['job_id'] = f'job-colabfold-{exp_doc_ref.id.lower()}'
        runner_args['start_time'] = start_time
        runner_args['status'] = 'RUNNING'
        runner_args['colabfold_machine_type'] = colabfold_machine_type
        runner_args['job_gcs_path'] = job_gcs_path
        runner_args['labels'] = labels
        runner_args['filename'] = Path(blob.name).name
        runner_args['file_gcs_path'] = blob.name

        runner_args['use_templates'] = use_templates
        runner_args['use_amber'] = use_amber
        runner_args['msa_mode'] = msa_mode
        runner_args['model_type'] = model_type
        runner_args['num_models'] = num_models
        runner_args['num_recycles'] = num_recycles
        runner_args['model_order'] = model_order
        runner_args['do_not_overwrite_results'] = do_not_overwrite_results
        runner_args['rank_by'] = rank_by
        runner_args['pair_mode'] = pair_mode
        runner_args['stop_at_score'] = stop_at_score
        runner_args['zip_results'] = zip_results
        runner_args['firestore_ref'] = exp_doc_ref.id
        runner_args['use_custom_msa'] = use_custom_msa

        runner_args['output_gcs_path'] = os.path.join(
            'gs://',
            job_gcs_path,
            runner_args['job_id']            
        )

        runner_args['input_dir'] = os.path.join(
            job_gcsfuse_local_dir,
            runner_args['job_id']
        )
        runner_args['result_dir'] = runner_args['input_dir']

        source_bucket = storage_client.bucket(input_bucket_name)
        source_blob = source_bucket.blob(blob.name)
        destination_bucket = storage_client.bucket(job_gcs_path)

        source_bucket.copy_blob(
            source_blob, 
            destination_bucket,
            os.path.join(
                runner_args["job_id"], Path(blob.name).name)
        )

        # Write args to Firestore
        exp_doc_ref.set(runner_args, merge=True)

        runners.append(runner_args)

    args['runners'] = runners

    return args


def execute_workflow( 
    workflow_name: str,
    args: T.Dict
):
    """Submit a Cloud Workflows to process the Alphafold inference pipeline

    Args:
        workflow_name (str):
            Required. Name of the Cloud Workflows deployed in the setup stage.
        args (dict):
            Required. All the prepared arguments to execute the Workflows.
    Returns:
        (str) Full name of the Workflows path.
    Raises:
        None
    """
    # Set up API clients.
    execution_client = executions_v1beta.ExecutionsClient()
    workflows_client = workflows_v1beta.WorkflowsClient()

    # Construct the fully qualified location path.
    parent = workflows_client.workflow_path(
        args['project_id'], args['region'], workflow_name)

    execution = executions_v1beta.Execution(argument=json.dumps(args))
    exec_request = executions_v1beta.CreateExecutionRequest(
        parent = parent,
        execution = execution
    )

    # Execute the workflow.
    response = execution_client.create_execution(request=exec_request)

    print(f"Created execution: {response.name}")

    return response.name