FROM docker.io/nvidia/cuda:10.1-devel-ubuntu16.04

#Root directory of the container
ENV ROOT=/sec4bc
WORKDIR ${ROOT}
COPY ./ $ROOT/

RUN echo "Building SGEMM for NVIDIA GPU"
RUN chmod +x build-sgemm-nvidia.sh
RUN echo which nvcc
RUN ./build-sgemm-nvidia.sh
RUN echo "Compilation complete, use docker run --gpus all <image_name>"

RUN chmod +x run-sgemm-nvidia.sh
CMD ./run-sgemm-nvidia.sh
