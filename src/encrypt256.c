#include <stdio.h>
#include <inttypes.h>

#include <emmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>

void print128_num(__m128i var)
{
	int64_t e0, e1, e2, e3;
	e0 = _mm_extract_epi64 (var, 0);
	e1 = _mm_extract_epi64 (var, 1);
	//e1 = _mm_extract_epi64 (var, 1);
	//e1 = _mm_extract_epi64 (var, 1);

    printf("%016"PRIx64"%016"PRIx64"\n", e1, e0);
}

void print256_num(__m256i var)
{
	int64_t e0, e1, e2, e3;
	e0 = _mm256_extract_epi64 (var, 0);
	e1 = _mm256_extract_epi64 (var, 1);
	e2 = _mm256_extract_epi64 (var, 2);
	e3 = _mm256_extract_epi64 (var, 3);

    printf("%016"PRIx64" %016"PRIx64" %016"PRIx64" %016"PRIx64"\n", e3, e2, e1, e0);
}

__m128i get_m128i_variable_from_uint8_array(uint8_t *inputArray)
{
	__m128i res;
	int i;
	int64_t l64, r64;
	l64 = 0;
	r64 = 0;
	
	for(i=0;i<16;i++)
	{
		if(i<8)
			l64 = l64 << 8 | inputArray[i];//most signif. bits
		else
			r64 = r64 << 8 | inputArray[i];//least signif. bits
	}
	res = _mm_set_epi64x (l64, r64);
	
	return res;
}

__m256i get_m256i_variable_from_uint8_array(uint8_t *inputArray)
{
	__m256i res;
	int i;
	int64_t b3, b2, b1, b0;
	
	b3 = 0;
	b2 = 0;
	b1 = 0;
	b0 = 0;
	
	for(i=0;i<32;i++)
	{
		if(i<8)
			b3 = b3 << 8 | inputArray[i];//most signif. bits
		else if(i<16)
			b2 = b2 << 8 | inputArray[i];
		else if(i<24)
			b1 = b1 << 8 | inputArray[i];
		else if(i<32)
			b0 = b0 << 8 | inputArray[i];//least signif. bits
	}
	res = _mm256_set_epi64x (b3, b2, b1, b0);
	
	return res;
}

__m256i ShiftRowsLayer (__m256i input)
{
	uint8_t shiftRowsScalar[] = {0x0f, 0x0a, 0x05, 0x00, 0x0b, 0x06, 0x01, 0x0c, 0x07, 0x02, 0x0d, 0x08, 0x03, 0x0e, 0x09, 0x04, 0x0f, 0x0a, 0x05, 0x00, 0x0b, 0x06, 0x01, 0x0c, 0x07, 0x02, 0x0d, 0x08, 0x03, 0x0e, 0x09, 0x04};
	__m256i shiftRows = get_m256i_variable_from_uint8_array(shiftRowsScalar);
	return _mm256_shuffle_epi8(input, shiftRows);
}

void inverseGF2P4 (__m128i input, __m128i *lOut, __m128i *rOut)
{
	//input is of form a+bt
	uint8_t lowersplitterscalar[] = {0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f};
	__m128i lowersplitter = get_m128i_variable_from_uint8_array(lowersplitterscalar);
	__m128i a = _mm_and_si128 (input, lowersplitter);

	uint8_t uppersplitterscalar[] = {0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0};
	__m128i uppersplitter = get_m128i_variable_from_uint8_array(uppersplitterscalar);
	__m128i b = _mm_and_si128 (input, uppersplitter);
	b = _mm_srli_epi32 (b, 4);
	
	//find a+b
	//let's call it x
	__m128i x = _mm_xor_si128 (a, b);
	
	//tables to be used below
	__m128i gf2p4Inverse, lambdaInverse, lambdaMul;
	uint8_t inv[] = {0x08, 0x03, 0x04, 0x0a, 0x05, 0x0c, 0x02, 0x0f, 0x06, 0x07, 0x0b, 0x0d, 0x0e, 0x09, 0x01, 0x80};
	uint8_t lambMul[] = {0x0e, 0x07, 0x0f, 0x06, 0x0c, 0x05, 0x0d, 0x04, 0x0a, 0x03, 0x0b, 0x02, 0x08, 0x01, 0x09, 0x00};
	uint8_t lambInv[] = {0x03, 0x06, 0x08, 0x07, 0x0a, 0x0b, 0x04, 0x0d, 0x0c, 0x0e, 0x05, 0x09, 0x0f, 0x01, 0x02, 0x80};
	gf2p4Inverse = get_m128i_variable_from_uint8_array(inv);
	lambdaInverse = get_m128i_variable_from_uint8_array(lambInv);
	lambdaMul = get_m128i_variable_from_uint8_array(lambMul);
	
	//find inverse of a
	__m128i aInv = _mm_shuffle_epi8(gf2p4Inverse, a);
		
	//find inverse of a+b
	__m128i xInv = _mm_shuffle_epi8(gf2p4Inverse, x);
	
	//find inverse of lambda*b
	__m128i blInv = _mm_shuffle_epi8(lambdaInverse, b);
	
	//find aInv + blInv
	aInv = _mm_xor_si128(aInv, blInv);
	
	//find xInv + blInv
	xInv = _mm_xor_si128(xInv, blInv);
	
	//find inverse(aInv + blInv)
	aInv = _mm_shuffle_epi8(gf2p4Inverse, aInv);
	
	//find inverse(xInv + blInv)
	xInv = _mm_shuffle_epi8(gf2p4Inverse, xInv);
	
	//find inverse(aInv+blInv)+x
	aInv = _mm_xor_si128(aInv, x);
	
	//find inverse(xInv+blInv)+a
	xInv = _mm_xor_si128(xInv, a);
	
	//find I1
	__m128i I1 = _mm_shuffle_epi8(gf2p4Inverse, aInv);
	
	//find I2
	__m128i I2 = _mm_shuffle_epi8(gf2p4Inverse, xInv);
	
	//let's assume c+dt be the inverse of input
	//then d=I1+I2
	//and c=I2+(I1+I2)lambda
	//__m128i d = _mm_xor_si128(I1, I2);
	*lOut = _mm_xor_si128(I1, I2);
	
	I1 = _mm_shuffle_epi8(lambdaMul, *lOut);
	
	//__m128i c = _mm_xor_si128(I1, I2);
	*rOut = _mm_xor_si128(I1, I2);
}

void substituteLayer (__m128i input, __m128i *sBoxOut, __m128i *sBoxOutDouble)
{
	__m128i res;
	
	uint8_t lowersplitterscalar[] = {0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f};
	__m128i lowersplitter = get_m128i_variable_from_uint8_array(lowersplitterscalar);
		
	uint8_t uppersplitterscalar[] = {0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0};
	__m128i uppersplitter = get_m128i_variable_from_uint8_array(uppersplitterscalar);
	
	__m128i inputLo = _mm_and_si128 (input, lowersplitter);
	__m128i inputHi = _mm_and_si128 (input, uppersplitter);
	inputHi = _mm_srli_epi32 (inputHi, 4);
	
	uint8_t isoMappingLowScalar[] = {0x25, 0x24, 0x0b, 0x0a, 0x6c, 0x6d, 0x42, 0x43, 0x66, 0x67, 0x48, 0x49, 0x2f, 0x2e, 0x01, 0x00};
	uint8_t isoMappingHighScalar[] = {0x31, 0x04, 0xe1, 0xd4, 0x0c, 0x39, 0xdc, 0xe9, 0xd8, 0xed, 0x08, 0x3d, 0xe5, 0xd0, 0x35, 0x00};

	__m128i isoMappingLow = get_m128i_variable_from_uint8_array(isoMappingLowScalar);
	__m128i isoMappingHigh = get_m128i_variable_from_uint8_array(isoMappingHighScalar);
	
	inputLo = _mm_shuffle_epi8(isoMappingLow, inputLo);
	inputHi = _mm_shuffle_epi8(isoMappingHigh, inputHi);
	input   = _mm_xor_si128 (inputLo, inputHi);
	
	inverseGF2P4(input, &inputHi, &inputLo);
	
	uint8_t invisoMappingLowScalar[] = {0x53, 0x4c, 0xe1, 0xfe, 0xf8, 0xe7, 0x4a, 0x55, 0x65, 0x7a, 0xd7, 0xc8, 0xce, 0xd1, 0x7c, 0x63};
	uint8_t invisoMappingHighScalar[] = {0xf2, 0xa6, 0xe3, 0xb7, 0xa7, 0xf3, 0xb6, 0xe2, 0x10, 0x44, 0x01, 0x55, 0x45, 0x11, 0x54, 0x00};
	
	__m128i invisoMappingLow = get_m128i_variable_from_uint8_array(invisoMappingLowScalar);
	__m128i invisoMappingHigh = get_m128i_variable_from_uint8_array(invisoMappingHighScalar);
	
	__m128i inputLoOut = _mm_shuffle_epi8(invisoMappingLow, inputLo);
	__m128i inputHiOut = _mm_shuffle_epi8(invisoMappingHigh, inputHi);
	
	*sBoxOut = _mm_xor_si128 (inputLoOut, inputHiOut);
	
	if(sBoxOutDouble!=NULL)
	{
		uint8_t invisoMappingLowDoubleScalar[] = {0xa6, 0x98, 0xd9, 0xe7, 0xeb, 0xd5, 0x94, 0xaa, 0xca, 0xf4, 0xb5, 0x8b, 0x87, 0xb9, 0xf8, 0xc6};
		uint8_t invisoMappingHighDoubleScalar[] = {0xff, 0x57, 0xdd, 0x75, 0x55, 0xfd, 0x77, 0xdf, 0x20, 0x88, 0x02, 0xaa, 0x8a, 0x22, 0xa8, 0x00};
		
		__m128i invisoMappingLowDouble = get_m128i_variable_from_uint8_array(invisoMappingLowDoubleScalar);
		__m128i invisoMappingHighDouble = get_m128i_variable_from_uint8_array(invisoMappingHighDoubleScalar);	
		
		__m128i inputLoDoubleOut = _mm_shuffle_epi8(invisoMappingLowDouble, inputLo);
		__m128i inputHiDoubleOut = _mm_shuffle_epi8(invisoMappingHighDouble, inputHi);
		
		*sBoxOutDouble = _mm_xor_si128 (inputLoDoubleOut, inputHiDoubleOut);
	}
}

__m128i mixColumnLayer (__m128i x, __m128i xDouble)
{
	uint8_t leftRotate1Scalar[] = {0x0e, 0x0d, 0x0c, 0x0f, 0x0a, 0x09, 0x08, 0x0b, 0x06, 0x05, 0x04, 0x07, 0x02, 0x01, 0x00, 0x03};
	__m128i leftRotate1 = get_m128i_variable_from_uint8_array(leftRotate1Scalar);

	uint8_t leftRotate3Scalar[] = {0x0d, 0x0c, 0x0f, 0x0e, 0x09, 0x08, 0x0b, 0x0a, 0x05, 0x04, 0x07, 0x06, 0x01, 0x00, 0x03, 0x02};
	__m128i leftRotate3 = get_m128i_variable_from_uint8_array(leftRotate3Scalar);
	
	x = _mm_shuffle_epi8 (x, leftRotate1);
	
	xDouble = _mm_xor_si128 (x, xDouble);
	
	x = _mm_shuffle_epi8 (x, leftRotate3);
	
	x = _mm_xor_si128 (x, xDouble);
	
	xDouble = _mm_shuffle_epi8 (xDouble, leftRotate1);
	
	x = _mm_xor_si128 (x, xDouble);
	
	return x;
}

__m128i encrypt_256 (__m128i input, __m128i keys[])
{
	int i=1;
	
	//initial addroundkey
	input = _mm_xor_si128 (input, keys[0]);
	//print128_num(input);
	//printf("\n");
	for(i=1;i<=10;i++)
	{
		//shiftrows
		//input = ShiftRowsLayer(input);

		//sbox
		__m128i xDouble;
		substituteLayer (input, &input, &xDouble);
		
		//mixcolumn
		if(i!=10)
			input = mixColumnLayer (input, xDouble);
		
		//addroundkey
		input = _mm_xor_si128 (input, keys[i]);
	}
	
	return input;
}

int main()
{
	uint8_t inputscalar[] = {0xAE, 0x2D, 0x8A, 0x57, 0x1E, 0x03, 0xAC, 0x9C, 0x9E, 0xB7, 0x6F, 0xAC, 0x45, 0xAF, 0x8E, 0x51, 0xab, 0x1e, 0x56, 0x75, 0x93, 0x15, 0x26, 0x88, 0x97, 0xa6, 0xbd, 0x7a, 0x9b, 0x0c, 0x1f, 0xae};
	__m256i input = get_m256i_variable_from_uint8_array(inputscalar);
	print256_num(input);
	input = ShiftRowsLayer (input);
	
	uint8_t keyscalar[] = {0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6, 0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C};
	__m128i key = get_m128i_variable_from_uint8_array(keyscalar);
	
	//__m128i cipherText = encrypt_128 (input, key);
	
	print256_num(input);
}
