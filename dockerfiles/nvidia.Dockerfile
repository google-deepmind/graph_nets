FROM tensorflow/tensorflow:latest-gpu

RUN pip install \
        graph_nets \
        tensorflow_probability_gpu \
        && \
    mkdir /.local && \
    chmod a+rwx /.local

WORKDIR /my-devel

EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && \
        jupyter notebook \
            --notebook-dir=/my-devel \
            --ip 0.0.0.0 \
            --no-browser \
            --allow-root"]
