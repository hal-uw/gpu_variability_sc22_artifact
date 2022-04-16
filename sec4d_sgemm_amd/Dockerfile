FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install build-essential git m4 scons zlib1g zlib1g-dev \
    libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
    python3-dev python3-six python-is-python3 doxygen libboost-all-dev \
    libhdf5-serial-dev python3-pydot libpng-dev libelf-dev pkg-config
# Requirements for ROCm
RUN apt -y install cmake mesa-common-dev libgflags-dev libgoogle-glog-dev
# Needed to get ROCm repo, build packages
RUN apt -y install wget gnupg2 rpm
RUN wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
# ROCm webpage says to use debian main, but the individual versions
# only have xenial
RUN echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/4.0.1/ xenial main' | tee /etc/apt/sources.list.d/rocm.list
RUN apt-get update && apt -y install hsakmt-roct hsakmt-roct-dev
RUN ln -s /opt/rocm-4.0.1 /opt/rocm
RUN git clone -b rocm-4.0.0 https://github.com/RadeonOpenCompute/ROCR-Runtime.git && \
    mkdir -p /ROCR-Runtime/src/build
WORKDIR /ROCR-Runtime/src/build
# need MEMFD_CREATE=OFF as MEMFD_CREATE syscall isn't implemented
RUN cmake -DIMAGE_SUPPORT=OFF -DHAVE_MEMFD_CREATE=OFF \
    -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) && make package
RUN apt -y install ./hsa-rocr-dev*.deb
WORKDIR /
# Dependencies for ROCclr
RUN apt -y install llvm-amdgpu libncurses5 libtinfo-dev rocm-device-libs comgr
RUN git clone -b rocm-4.0.0 \
    https://github.com/ROCm-Developer-Tools/ROCclr.git && \
    mkdir -p ROCclr/build
RUN git clone -b rocm-4.0.0 \
    https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git && \
    mkdir -p ROCm-OpenCL-Runtime/build
WORKDIR /ROCclr/build
RUN cmake -DOPENCL_DIR="/ROCm-OpenCL-Runtime" \
    -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && make install
WORKDIR /
WORKDIR ROCm-OpenCL-Runtime/build
RUN cmake -DUSE_COMGR_LIBRARY=ON -DCMAKE_PREFIX_PATH="/opt/rocm" \
    -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && make package
RUN apt -y install ./rocm-opencl-2.0.0-amd64.deb ./rocm-opencl-dev-2.0.0-amd64.deb
WORKDIR /
RUN git clone -b rocm-4.0.0 \
    https://github.com/ROCm-Developer-Tools/HIP.git && mkdir -p HIP/build
WORKDIR HIP/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DHSA_PATH=/usr/hsa \
    -DHIP_COMPILER=clang -DHIP_PLATFORM=rocclr -DCMAKE_PREFIX_PATH="/opt/rocm"\
    .. && make -j$(nproc) && make package
RUN apt -y install ./hip-base*.deb ./hip-rocclr*.deb
# These files here are needed but don't get installed through the .deb file,
# even though they seem to be included in the packaging, so symlink them
RUN ln -s /HIP/build/rocclr/CMakeFiles/Export/_opt/rocm/hip/lib/cmake/hip/* /opt/rocm/hip/lib/cmake/hip/
WORKDIR /
# rocBLAS downloads the most recent rocm-cmake if it isn't installed before
# building
RUN apt install rocm-cmake
RUN git clone -b rocm-4.0.0 \
    https://github.com/ROCmSoftwarePlatform/rocBLAS.git && mkdir rocBLAS/build
ENV HCC_AMDGPU_TARGET=gfx906
WORKDIR rocBLAS
# rocBLAS needs to be built from source otherwise gfx801 gets an error in HIP
# about there being no GPU binary available
RUN ./install.sh -d -a all -i
WORKDIR /

#Root directory of the container
ENV ROOT=/sec4d
WORKDIR ${ROOT}
COPY ./ $ROOT/

RUN echo "Building SGEMM for AMD GPU"
RUN chmod +x build-sgemm-amd.sh
RUN ./build-sgemm-amd.sh
RUN echo "Compilation complete, use docker run --gpus all <image_name>"

RUN chmod +x run-sgemm-amd.sh
CMD ./run-sgemm-amd.sh
