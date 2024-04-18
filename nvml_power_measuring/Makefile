TARGETS = dumpGpuPower

CC := g++
NVML_LIB_DIR := -L$(CUDA_INSTALL_PATH)/lib64/stubs/
CFLAGS := -I$(CUDA_INSTALL_PATH)/include/
LDFLAGS := $(NVML_LIB_DIR) -lnvidia-ml

all: $(TARGETS)

dumpGpuPower: dumpGpuPower.cpp
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(TARGETS)
