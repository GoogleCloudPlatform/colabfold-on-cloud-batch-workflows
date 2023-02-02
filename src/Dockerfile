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

ARG CUDA=11.1.1
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \ 
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        cmake \
        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
        git \
        tzdata \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Install python
RUN apt-get update -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y
RUN apt-get install python3.8 python3.8-venv -y
RUN python3.8 -m ensurepip --upgrade

ENV VIRTUAL_ENV=/opt/venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip

# install dependencies
RUN pip install "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"
RUN pip install -U jax==0.3.25
RUN pip install -U jaxlib==0.3.25+cuda11.cudnn805 \
            -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN touch COLABFOLD_READY

# Download params (~1min)
RUN python -m colabfold.download

# # setup conda
RUN wget -qnc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local 2>&1 1>/dev/null
RUN rm Miniconda3-latest-Linux-x86_64.sh
RUN touch CONDA_READY

# setup template search
RUN conda install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 python="3.8" 2>&1 1>/dev/null
RUN touch HH_READY

# setup openmm for amber refinement
RUN conda install -y -c conda-forge openmm=7.5.1 python="3.8" pdbfixer 2>&1 1>/dev/null
RUN touch AMBER_READY

COPY residue_constants.py /opt/venv/lib/python3.8/site-packages/alphafold/common/
COPY templates.py /opt/venv/lib/python3.8/site-packages/alphafold/data/

WORKDIR /colabfold_batch
COPY run_colabfold.py .

ENV TF_FORCE_UNIFIED_MEMORY="1"
ENV XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"

# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real
RUN ldconfig