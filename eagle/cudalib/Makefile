ifeq ($(VERBOSE), 1)
	ECHO:=
else
	ECHO:=@
endif
#############################
CUDA_HOME := /usr/local/cuda-9.0
CXX := g++
NVCC := $(CUDA_HOME)/bin/nvcc
CXXFLAGS := -std=c++11 -Wall
sm := 70
NVCCFLAGS :=--std=c++11 -m64 -Xcompiler -Wall -Xptxas -v -Wno-deprecated-gpu-targets 
NVCCFLAGS +=-gencode arch=compute_$(sm),code=sm_$(sm)
NVCC_LDFLAGS = -lcudart -lcublas -L$(CUDA_HOME)/lib64
# Compilation flags
ifeq ($(DEBUG), 1)
	CXXFLAGS += -g
	NVCCFLAGS += -g -G
else
	CXXFLAGS += -O2
	NVCCFLAGS += -O2
endif

# Target
TARGET := demo
TARGET_DIR := bin

# Directories
INC_DIRS := ./ $(CUDA_HOME)/include
LIB_DIRS := $(CUDA_HOME)/lib64

# Files
INCS := $(wildcard )
SRCS := $(wildcard ./*.cpp)
CUS  := $(wildcard ./*.cu)
CUS_OBJ := $(CUS:.cu=.o)
$(info CUS=$(CUS))
$(info CUS_OBJ=$(CUS_OBJ))
$(info SRCS=$(SRCS))

LIBS :=

# Make it all!
all : $(TARGET_DIR)/$(TARGET)

# Host executable target.
$(TARGET_DIR)/$(TARGET) : Makefile $(SRCS) $(INCS) $(TARGET_DIR) $(CUS_OBJ)
	$(ECHO)$(CXX) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
		$(foreach D,$(LIB_DIRS),-L$D)\
		$(foreach L,$(LIBS),-l$L)\
		$(SRCS) $(CUS_OBJ) $(NVCC_LDFLAGS)\
		-o $(TARGET_DIR)/$(TARGET)

$(CUS_OBJ):$(CUS)
	$(ECHO) $(NVCC) $(NVCCFLAGS) $(INCS) -c $< -o $@
		
$(TARGET_DIR) :
	$(ECHO)mkdir $(TARGET_DIR)
	
	
        
# Standard make targets
clean:
	$(ECHO)rm -f $(TARGET_DIR)/$(TARGET)
	$(ECHO)rm -f *.o

.PHONY : all clean