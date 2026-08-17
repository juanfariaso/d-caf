
# Installation

## About performance

I tested Dcaf on a specific version of amuse that allowed me to patch it in order to install PeTar with GPU.
A GPU is not mandatory but it considerably improves performance of the gradual formation of stars so it can grow naturallt to large number of particles.

On the CPU side the best performance is still using a single CPU. I have tried increasing the number of workers as the number of stars raise, e.g.: stopping PeTar, and initialize it again with more workers. However that has not been faster than letting a single worker to do the job.

On small enough systems (N<5000) it works fine. But larger systems would be most efficient using GPU instead with only 1 worker (no more). I already tested it and gives the best perfomance.

If you are doing only CPU, then the GPU part below can be skipped and in principle a newer version of AMUSE should not be a problem as far as PeTar works.


## Installation steps

Create a python environment and activate it:

```shell
python -m venv ~/.venv/dcaf
source ~/.venv/dcaf/bin/activate
```

Add the AMUSE environment variables to your environment `~/.bashrc` 

```shell
export AMUSE_DIR=$HOME/codes/amuse
export PYTHONPATH=$AMUSE_DIR/src:$PYTHONPATH
```

Clone [AMUSE](https://github.com/amusecode/amuse) to your `$AMUSE_DIR` and install it.
I have tested dcaf on a specific checkpoint of amuse, mainly for the PeTar GPU installation. 
 After cloning do:
```
cd $AMUSE_DIR
git checkout aea5b55
```

we add features such as stellar evolution.

Install dependencies if you don't have then:
```
pip install -r requirements.txt --upgrade
pip install docutils
./configure
```
Should finish with Configuration done.

For now, we only require the framework.
```
make framework
```

If you don't need GPU for PeTar you can just do
```
make stopcond.code petar.code
```

But even if you will use the GPU version, make it anyway to make sure you have the correct configuration to make it work.


## Appendix: SeBa build note on this machine

On this machine, building `SeBa` directly from the local AMUSE tree required
building `stopcond` first and then passing its include and library paths
explicitly.

Activate the environment and define the AMUSE path:

```shell
source /Users/juan.farias/venv/base/bin/activate
export AMUSE_DIR=/Users/juan.farias/Codes/amuse
export PYTHONPATH=$AMUSE_DIR/src:$PYTHONPATH
```

Build `stopcond`:

```shell
cd $AMUSE_DIR/lib/stopcond
make
```

Then build the `SeBa` worker:

```shell
cd $AMUSE_DIR/src/amuse_seba
make STOPCOND_CFLAGS="-I$AMUSE_DIR/lib/stopcond" \
     STOPCOND_LIBS="-L$AMUSE_DIR/lib/stopcond -lstopcond" \
     seba_worker
```

The worker should then exist at:

```shell
$AMUSE_DIR/src/amuse_seba/seba_worker
```

On this machine, the `SeBa` worker also required the `stopcond` library path
to be added at runtime. Before launching Python or a DCAF simulation, export:

```shell
export DYLD_LIBRARY_PATH=$AMUSE_DIR/lib/stopcond:$DYLD_LIBRARY_PATH
```

You can test the worker with:

```shell
python3 - <<'PY'
from amuse.community.seba.interface import SeBa
print("before")
se = SeBa(redirection='none')
print("after")
se.stop()
PY
```


This may be just enough if you want to use DCAF without GPU.
However, the framework will work much more efficiently with GPU enabled. This because we intend to grow the system gradually. Running a  few stars with many processors is quite inefficient, and using one processor only works well for small systems (<2000 stars).
The best performance is achieved if we manage to build PeTar with activated GPU since the GPU optimization can smoothly transition from a small system into a large one. The alternative would be to stop the code and add more workers on the go, but the initialization of PeTar in amuse several times is very inefficient and ended up being worse when I tested it.

For this however, we need to modify the installation instructions for PeTar in AMUSE. Unfortunately, after this, we will not be able to switch back to cpu case unless we build again.

## PeTar with GPU enabled in AMUSE

Below has been tested with cuda 12.6.9

First make sure PeTar can compile normally with amuse

make sure these environment variables are set:
```
export CUDA_TK=/path_to_cuda
```

We need to link the CUDA libraries to PeTar and backup the Makefile:
```
cd $AMUSE_DIR/src/amuse/community/petar
cp Makefile Makefile.cpu_backup
```

Now lets modify this block on the Makefile, add the GPU support block at this point (around line 38)
And remember to add the helpers on the INCLUDE at the beginning.

(Actually better to just provide the makefile)

```make
INCLUDE  += -I src/PeTar/src -I src/SDAR/src -I src/FDPS/src -I src/PeTar/cuda_helper

CXXFLAGS = ${INCLUDE} -fPIC -O2 -Wall -std=c++17 -fopenmp
CXXFLAGS += -D PARTICLE_SIMULATOR_THREAD_PARALLEL
CXXFLAGS += -D PARTICLE_SIMULATOR_MPI_PARALLEL
CXXFLAGS += -D MPICH_IGNORE_CXX_SEEKC
CXXFLAGS += -D SOFT_PERT -D AR_TTL -D AR_SLOWDOWN_TREE -D AR_SLOWDOWN_TIMESCALE -D CLUSTER_VELOCITY
CXXFLAGS += -D USE_QUAD
CXXFLAGS += -D STELLAR_EVOLUTION
CXXFLAGS += -D PROFILE
CXXFLAGS += -D HARD_CHECK_ENERGY
CXXFLAGS += -D TIDAL_TENSOR_3RD
CXXFLAGS += -D INTERFACE_DEBUG_PRINT
CXXFLAGS += -D INTERFACE_DEBUG

# ---------------- GPU support ----------------
CUDA_PREFIX ?= $(CUDA_TK)
NVCC        ?= nvcc

CXXFLAGS += -DUSE_GPU
CXXFLAGS += -D GPU_PROFILE
INCLUDE  += -I $(CUDA_PREFIX)/include

CUDALIBS = -L$(CUDA_PREFIX)/lib64 -L$(CUDA_PREFIX)/targets/x86_64-linux/lib -lcudart

LDFLAGS  += $(CUDALIBS)
# ---------------------------------------------

LDFLAGS  += -lm $(MUSE_LD_FLAGS)
```

Then we add the object we want and the rule to compile them at the end of the file (around line 55 if you added above)
```
OBJS = interface.o force_gpu_cuda.o

force_gpu_cuda.o: src/PeTar/src/force_gpu_cuda.cu
	$(NVCC) -c $< -o $@ -Xcompiler "$(CXXFLAGS)"
```

# Make sure the cuda headers are created
(again, better to provide these files)
We now need to create the `cuda_helper` folder that contains the cuda headers.

The folder `amuse/src/amuse/community/petar/src/PeTar/cuda_helper` must exist and contain the following files:

helper_cuda.h
```c++
#pragma once
#include <stdio.h>
#include <cuda_runtime.h>

inline void __checkCudaErr(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",
                cudaGetErrorString(err), file, line);
        exit(1);
    }
}

#define checkCudaErrors(val) __checkCudaErr((val), __FILE__, __LINE__)
```

helper_math.h
```c++
#pragma once

// minimal float3 helpers
inline __host__ __device__ float3 make_float3(float x, float y, float z) {
    float3 f; f.x=x; f.y=y; f.z=z; return f;
}

inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b){
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b){
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __host__ __device__ float3 operator*(const float3 &a, float s){
    return make_float3(a.x*s, a.y*s, a.z*s);
}

inline __host__ __device__ float3 operator/(const float3 &a, float s){
    return make_float3(a.x/s, a.y/s, a.z/s);
}
```

# Compile petar
With all this, we should be able to compile petar. Apparently it requires that `stopcond` code to be built first, so from the amuse root directory we do:
``
first try enable cuda on the amuse directory:

```
cd $AMUSE_DIR
./configure --enable-cuda
```
Since we are using an old checkpoint fo AMUSE the format it guesses for Cuda directiories is outdated. If this is the case for you, make the configuration without the cuda and modify config.mk manually.

 For this first we must make sure cuda libraries are working and properly linked on the system. Mainly, make sure this works with no errors:

```
echo 'int main(){return 0;}' > conftest.c && gcc conftest.c -L$CUDA_TK/targets/x86_64-linux/lib -lcudart 
```
if this fails, CUDA runtime libraries are not visible to the linker. **Fix that before continuing.**



Then if config.mk still has CUDA_ENABLED=no and CUDA_TK=/NOCUDACONFIGURED, edit config.mk manually:

```
CUDA_ENABLED=yes
NVCC=/path/to/nvcc
CUDA_TK=/path/to/cuda 
CUDA_LIBS= -lcuda -lcudart

```


 If nvcc fails with mpi.h: No such file or directory, make sure the OpenMPI include directory is available through CPATH.
```
export CPATH="/path_to_openmpi/include:$CPATH"
```

then, finally
``
```
cd $AMUSE_DIR/src/amuse/community/petar
make clean
cd $AMUSE_DIR
make petar.code
```

and if all goes fine, PeTar should be ready
