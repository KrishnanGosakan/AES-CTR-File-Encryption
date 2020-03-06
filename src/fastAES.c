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
#include "encrypt.h"

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
	__m128i *blocks = (__m128i *)malloc(((args->size/16) + 1)*sizeof(__m128i));
	int bnum = 0;
    while( index < args->size )
    {
    	uint8_t ip[16];
    	int i;
    	for(i=0;i<16;i++)
    		if(args->size-(index+i) > 0)
    			ip[i]=args->fileContent[index+i];
    		else
    			ip[i]=index+16-args->size;
    	
    	blocks[bnum++] = get_m128i_variable_from_uint8_array (ip);
       	index += 16;
    }
    return (void *)blocks;
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
    
    int totalBlocks = size/16 + 1;
    __m128i *inputBlocks = (__m128i *)malloc(totalBlocks*sizeof(__m128i));
    
    pthread_join(tid, (void **)&inputBlocks);
   	
    char *keystring = "a l l is w e l l";
    __m128i key = get_m128i_variable_from_uint8_array (keystring);
    
    __m128i iv = _mm_setzero_si128();
    
    int64_t l64, r64;
	l64 = 0;
	r64 = 1;
	__m128i oneAdder = _mm_set_epi64x (l64, r64);
    
    __m128i keySchedules[11];
	keySchedules[0] = key;
	
	int i;
	for(i=1;i<=10;i++)
	{
		key = getKeySchedule (key, i);
		keySchedules[i] = key;
	}
    
    for(i=0;i<totalBlocks;i++)
    {
    	if(totalBlocks - i > 1)
    	{
    		__m128i block1, block2;
    		
    		__m256i twoBlocks = _mm256_insertf128_si256(_mm256_castsi128_si256(inputBlocks[i+1]), inputBlocks[i], 1);
    	    print256_num(encrypt_256(twoBlocks,keySchedules));
    		i++;
    	}
    	else
    	{
    		print128_num(encrypt_128(inputBlocks[i],keySchedules));
    	}
    }
    
    printf("\n");
    
    for(i=0;i<totalBlocks;i++)
    	print128_num(encrypt_128(inputBlocks[i],keySchedules));
    
    printf("\n");
    
    return 0;
}
