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

FROM gcr.io/deeplearning-platform-release/base-cu113

WORKDIR colabfold
RUN pip install "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"
RUN pip install "jax[cuda]>=0.3.8,<0.4" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN conda install -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0
RUN conda install -c conda-forge openmm=7.5.1 pdbfixer

ENTRYPOINT [ "colabfold_batch" ]