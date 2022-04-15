FROM nvcr.io/nvidia/pytorch:22.03-py3

#Root directory of the container
ENV ROOT=/project  
WORKDIR ${ROOT}
COPY ./ $ROOT/

RUN ls -la ${ROOT}
RUN echo "Building packages"
RUN cd $ROOT/src/ && make clean-all
RUN cd $ROOT/src/ && make yes-user-reaxc
RUN cd $ROOT/src/ && make yes-kokkos
RUN cd $ROOT/src/ && make -j kokkos_cuda_mpi KOKKOS_ARCH=Volta70
RUN echo "Compiling binary assuming ARCH=Volta70 - to change, edit Dockerfile"
RUN echo "Done building packages for LAMMPS"

# Running LAMMPS
CMD cd $ROOT/reax_benchmark/ && ./run-lammps.sh 0 1 0
RUN echo "Done running LAMMPS"
