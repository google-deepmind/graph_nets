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