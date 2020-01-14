

void inverseGF2P4(__m128i input, __m128i *lOut, __m128i *rOut)
{
	//input is of form a+bt
	uint8_t lowersplitterscalar[] = {0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
	__m128i lowersplitter = _mm_loadu_si128((const __m128i*)lowersplitterscalar);
	__m128i a = _mm_and_si128 (input, lowersplitter);

	uint8_t uppersplitterscalar[] = {0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
	__m128i uppersplitter = _mm_loadu_si128((const __m128i*)uppersplitterscalar);
	__m128i b = _mm_and_si128 (input, uppersplitter);
	b = _mm_srli_epi32 (b, 4);
	
	//find a+b
	//let's call it x
	__m128i x = _mm_xor_si128 (a, b);
	
	//tables to be used below
	__m128i gf2p4Inverse, lambdaInverse, lambdaMul;
	uint8_t inv[] = {0x80, 0x01, 0x09, 0x0e, 0x0d, 0x0b, 0x07, 0x06, 0x0f, 0x02, 0x0c, 0x05, 0x0a, 0x04, 0x03, 0x08};
	uint8_t lambMul[] = {0x00, 0x09, 0x01, 0x08, 0x02, 0x0b, 0x03, 0x0a, 0x04, 0x0d, 0x05, 0x0c, 0x06, 0x0f, 0x07, 0x0e};
	uint8_t lambInv[] = {0x80, 0x02, 0x01, 0x0f, 0x09, 0x05, 0x0e, 0x0c, 0x0d, 0x04, 0x0b, 0x0a, 0x07, 0x08, 0x06, 0x03};
	gf2p4Inverse = _mm_loadu_si128((const __m128i *)inv);
	lambdaInverse = _mm_loadu_si128((const __m128i *)lambInv);
	lambdaMul = _mm_loadu_si128((const __m128i *)lambMul);
	
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

__m128i substituteLayer (__m128i input)
{
	__m128i res;

	return res;
}

__m128i encrypt_128 (__m128i plainText, __m128i key)
{
	__m128i cipherText;
	
	return cipherText;
}
