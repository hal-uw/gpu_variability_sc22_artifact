#GENCODE_SM10 ?= -gencode=arch=compute_10,code=\"sm_10,compute_10\"
#GENCODE_SM13 ?= -gencode=arch=compute_13,code=\"sm_13,compute_13\"
#GENCODE_SM20 ?= -gencode=arch=compute_20,code=\"sm_20,compute_20\"
#GENCODE_SM30 ?= -gencode=arch=compute_30,code=\"sm_30,compute_30\"
#GENCODE_SM35 ?= -gencode=arch=compute_35,code=\"sm_35,compute_35\"
#GENCODE_SM50 ?= -gencode=arch=compute_50,code=\"sm_50,compute_50\"
#GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
#GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"
#GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
#GENCODE_SM72 ?= -gencode=arch=compute_72,code=\"sm_72,compute_72\"
GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"
#GENCODE_SM80 ?= -gencode=arch=compute_80,code=\"sm_80,compute_80\"
#GENCODE_SM86 ?= -gencode=arch=compute_86,code=\"sm_86,compute_86\"

BINDIR ?= .
CPPSRC = ../graph_parser/util.cpp ../graph_parser/parse.cpp
CUSRC = 
BASEEXE = pagerank
VARIANT ?= DEFAULT
# CUDA source files (compiled with cudacc)
ifeq ($(VARIANT),DEFAULT)
    EXECUTABLE = $(BINDIR)/$(BASEEXE)
    CUSRC += pagerank.cu
else ifeq ($(VARIANT),SPMV)
    EXECUTABLE = $(BINDIR)/$(BASEEXE)_spmv
    CUSRC += pagerank_spmv.cu
endif
OPTS = -O3
CUOPTS = $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM72) $(GENCODE_SM75) $(GENCODE_SM80) $(GENCODE_SM86)

INC = -I$(NVIDIA_COMPUTE_SDK_LOCATION)/../4.2/C/common/inc -I$(CUDA_INSTALL_PATH)/include
LIBS = -L$(NVIDIA_COMPUTE_SDK_LOCATION)/../4.2/C/lib -lcudart

OBJS = $(notdir $(CPPSRC:.cpp=.o))

all: $(OBJS)
	nvcc $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC) *.o $(INC) -o $(EXECUTABLE) $(LIBS)

$(OBJS):
	g++ -c $(CPPSRC) $(OPTS)

clean:
	rm -f *.o *.cu.cpp $(EXECUTABLE)

cleancucpp:
	rm -f *.cu.cpp
