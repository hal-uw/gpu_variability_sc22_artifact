FROM nvcr.io/nvidia/pytorch:22.03-py3

#Root directory of the container
ENV ROOT=/sec5c 
WORKDIR ${ROOT}
COPY ./ $ROOT/


RUN echo "cd to src directory for pannotia"
RUN cd ${ROOT}/src/cuda/pannotia/pagerank_mod/ && make clean
RUN cd ${ROOT}/src/cuda/pannotia/pagerank_mod/ && env VARIANT=SPMV make
RUN echo "Compiled pagerank_spmv binary"
RUN echo "Getting input graph from Suite Sparse Matrix Collection"
RUN wget https://suitesparse-collection-website.herokuapp.com/MM/Rajat/rajat30.tar.gz
RUN tar -xvzf rajat30.tar.gz
RUN echo "Fetched input graph"
RUN echo "Build for PageRank complete, now launching run"

CMD ./run-pagerank.sh 0 1 0
