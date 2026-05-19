CUDA_VER?=
ifeq ($(CUDA_VER),)
	$(error "CUDA_VER is not set")
endif

CXX:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

TARGET_LIB:= libnvdsinfer_custom_impl_Yolo_obb.so

CFLAGS:= -Wall -std=c++17 -shared -fPIC -Wno-error=deprecated-declarations

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include
CUFLAGS:= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include

LIBS+= -lnvinfer -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lstdc++fs

LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

SRCS:= $(wildcard *.cpp)
SRCS+= $(wildcard *.cu)
INCS+= $(wildcard *.h)

OBJS:= $(addsuffix .o, $(basename $(SRCS)))
OBJS:= $(OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $(CUFLAGS) $<

$(TARGET_LIB) : $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LFLAGS)

clean:
	rm -rf $(OBJS) $(TARGET_LIB)
