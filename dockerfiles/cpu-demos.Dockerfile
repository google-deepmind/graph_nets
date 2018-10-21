# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

FROM tensorflow/tensorflow

RUN pip install \
        graph_nets \
        tensorflow_probability \
        && \
    mkdir /.local && \
    chmod a+rwx /.local

RUN cd / && \
    curl -LOk https://github.com/ \
        https://github.com/deepmind/graph_nets/archive/master.tar.gz \
        | tar xzv graph_nets-master/graph_nets/demos/ --strip=2 \
        && \
    chmod -R a+rwx /demos

EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && \
        jupyter notebook \
            --notebook-dir=/demos \
            --ip 0.0.0.0 \
            --no-browser \
            --allow-root"]