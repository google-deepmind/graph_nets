FROM tensorflow/tensorflow

RUN pip install \
        graph_nets \
        tensorflow_probability \
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
