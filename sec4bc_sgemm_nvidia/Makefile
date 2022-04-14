all: sgemm_nvidia gen_data

sgemm_nvidia: sgemm.cu
	nvcc -arch=compute_70 -code=sm_70 -o $@ $< -lcurand -lcudart -lcublas

gen_data: gen_data.cpp
    g++ -o $@ $<

clean:
	rm sgemm_nvidia gen_data
