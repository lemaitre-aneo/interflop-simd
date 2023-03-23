LIB_ARCH_FLAGS=-mavx
EXE_ARCH_FLAGS=-mavx512f
EXE_FLAGS=-DDEBUG=0 -DFORCE_LOAD=1

CC=gcc
LD=gcc
AR=ar

all: exe

%.o: %.c
	$(CC) $(LIB_ARCH_FLAGS) -O2 -o $@ -c $^

libadd.a: add.o add.sse.o add.avx.o add.avx512.o
	$(AR) -rc $@ $^

main.o: main.c
	$(CC) $(EXE_ARCH_FLAGS) $(EXE_FLAGS) -O2 -o $@ -c $^

exe: main.o libadd.a
	$(LD) $^ -o $@

clean:
	rm -f *.o *.a exe

.PHONY: all clean
