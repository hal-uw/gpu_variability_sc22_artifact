all: sgemm_amd gen_data

gen_data: gen_data.cpp
	g++ -o $@ $<

sgemm_amd: sgemm_rocblas.hip.cpp
	hipcc --amdgpu-target=gfx906 -I/opt/rocm-4.0.1/hiprand/include -I/opt/rocm-4.0.1/rocrand/include -L/opt/rocm-4.0.1/lib -o $@ $< -lhiprand -lrocblas

clean:
	rm gen_data sgemm_amd
