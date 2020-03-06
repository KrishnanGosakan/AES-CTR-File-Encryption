void print128_num(__m128i var);
void print256_num(__m256i var);
__m128i get_m128i_variable_from_uint8_array(uint8_t *inputArray);
__m128i getKeySchedule(__m128i prevKey, int i);
__m128i encrypt_128 (__m128i input, __m128i keys[]);
__m256i encrypt_256 (__m256i input, __m128i keys[]);
