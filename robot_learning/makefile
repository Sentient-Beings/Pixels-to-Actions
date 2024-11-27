MUJOCO_PATH = /opt/mujoco/mujoco-3.2.5
MUJOCO_INCLUDE = $(MUJOCO_PATH)/include
MUJOCO_BIN = $(MUJOCO_PATH)/bin
MUJOCO_LIB = $(MUJOCO_PATH)/lib
GLFW_INCLUDE =/usr/include 

COMMON = -O2 -I$(MUJOCO_INCLUDE) -I$(GLFW_INCLUDE) -L$(MUJOCO_LIB) -mavx -pthread -Wl,-rpath,$(MUJOCO_BIN)
LIBS = -lmujoco -lGL -lm -lglfw
CC = gcc

ROOT := $(notdir $(CURDIR))

all:
	$(CC) $(COMMON) ./src/*.c $(LIBS) -o ./$(ROOT)

main.o:
	$(CC) $(COMMON) -c ./src/*.c

clean:
	rm -f *.o ./$(ROOT)