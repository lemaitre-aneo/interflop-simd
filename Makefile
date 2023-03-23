LIB_ARCH_FLAGS=-mavx512f
EXE_ARCH_FLAGS=-mavx512f
EXE_FLAGS=-DDEBUG=0 -DFORCE_LOAD=1
VARIANT=VARIANT_PTR

CC=gcc
LD=gcc
AR=ar

all: exe

%.o: %.c
	$(CC) -DVARIANT=$(VARIANT) $(LIB_ARCH_FLAGS) -O2 -o $@ -c $^

libadd.a: add.o add.sse.o add.avx.o add.avx512.o
	$(AR) -rc $@ $^

main.o: main.c
	$(CC) -DVARIANT=$(VARIANT) $(EXE_ARCH_FLAGS) $(EXE_FLAGS) -O2 -o $@ -c $^

exe: main.o libadd.a
	$(LD) $^ -o $@

clean:
	rm -f *.o *.a exe

.PHONY: all clean
