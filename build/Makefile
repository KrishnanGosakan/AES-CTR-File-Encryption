all: aes

aes: fastAES.o encrypt.o encrypt256.o
	gcc -o aes encrypt.o encrypt256.o fastAES.o -pthread

fastAES.o: ../src/fastAES.c
	gcc -I../header ../src/fastAES.c -pthread -mavx2 -c

encrypt.o: ../src/encrypt.c
	gcc -I../header ../src/encrypt.c -mavx2 -c

encrypt256.o: ../src/encrypt256.c
	gcc -I../header ../src/encrypt256.c -mavx2 -c

clean:
	rm -rf *.o aes
