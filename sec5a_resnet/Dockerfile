FROM docker.io/pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

#Root directory of the container
ENV ROOT=/sec4bc
WORKDIR ${ROOT}
COPY ./ $ROOT/

# Run Resnet
CMD ./run-resnet.sh
