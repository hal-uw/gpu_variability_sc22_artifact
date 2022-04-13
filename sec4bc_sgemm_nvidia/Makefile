all: sgemm dgemm gen_data

gen_data: gen_data.cpp
	g++ -o $@ $<

sgemm: simple_sgemm.cu
	nvcc -arch=compute_70 -code=sm_70 -o $@ $< -lcurand -lcudart -lcublas

dgemm: simple_dgemm.cu
	nvcc -arch=compute_70 -code=sm_70 -o $@ $< -lcurand -lcudart -lcublas

clean:
	rm sgemm dgemm
