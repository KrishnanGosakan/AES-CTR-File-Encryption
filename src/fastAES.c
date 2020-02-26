#include <stdio.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/io.h>
#include <sys/mman.h>
#include <stdint.h>
#include <inttypes.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <pthread.h>

void print128_num(__m128i var)
{
	int64_t e0, e1;
	e0 = _mm_extract_epi64 (var, 0);
	e1 = _mm_extract_epi64 (var, 1);
    printf("%016"PRIx64"%016"PRIx64"\n", e0, e1);
}

struct inputBlock
{
	__m128 input;
	int    blockID;
};

struct fileReadArgs
{
	char *fileContent;
	int size;
};

void *fileReaderThread(void *fArgs)
{
	struct fileReadArgs *args = (struct fileReadArgs *)fArgs;
	int index = 0;
    while( index < args->size )
    {
    	uint8_t ip[16];
    	int i;
    	for(i=0;i<16;i++)
    		if(args->size-index > 0)
    			ip[i]=args->fileContent[index+i];
    		else
    			ip[i]=0;
    	
    	__m128i block_i = _mm_loadu_si128 ((const __m128i*) ip);
    	printf("block %d: ",index/16 + 1);
    	print128_num(block_i);
    	index += 16;
    }
}

int main()
{
	char *filename = "sampleFile";
	unsigned char *content;
    int size;
    struct stat s;
    int fd = open (filename, O_RDONLY);
    
    int status = fstat (fd, & s);
    size = s.st_size;
    
    //map input file to char* variable
    content = (char *) mmap (0, size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    //split content into 128 bit AES input and put it into queue
    struct fileReadArgs fInput;
    fInput.fileContent = content;
    fInput.size = size;
    pthread_t tid;
    pthread_create(&tid, NULL, fileReaderThread, &fInput);
    
    pthread_join(tid, NULL);
    
    return 0;
}
