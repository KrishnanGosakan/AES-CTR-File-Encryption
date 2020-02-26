#include <stdio.h>
#include <stdint.h>

int bitwisexor4bit(uint8_t v)
{
	int ret = 0;
	switch(v)
	{
		case 0x00://0000
			ret = 0;
			break;
		case 0x01://0001
			ret = 1;
			break;
		case 0x02://0010
			ret = 1;
			break;
		case 0x03://0011
			ret = 0;
			break;
		case 0x04://0100
			ret = 1;
			break;
		case 0x05://0101
			ret = 0;
			break;
		case 0x06://0110
			ret = 0;
			break;
		case 0x07://0111
			ret = 1;
			break;
		case 0x08://1000
			ret = 1;
			break;
		case 0x09://1001
			ret = 0;
			break;
		case 0x0a://1010
			ret = 0;
			break;
		case 0x0b://1011
			ret = 1;
			break;
		case 0x0c://1100
			ret = 0;
			break;
		case 0x0d://1101
			ret = 1;
			break;
		case 0x0e://1110
			ret = 1;
			break;
		case 0x0f://1111
			ret = 0;
			break;
	}
	return ret;
}

void forwardmap()
{
	uint8_t iter;
	uint8_t mat[] = {0xdd, 0x0a, 0x52, 0xc6, 0x70, 0xd2, 0xac, 0xa0};
	for(iter=0x0f;iter>=0x00;iter--)
	{
		int i;
		uint8_t res = 0;
		for(i=0;i<8;i++)
		{
			uint8_t t = iter & mat[i];
			int r = bitwisexor4bit(t%16) ^ bitwisexor4bit(t/16);
			res |= r<<i;
		}
		printf("0x%02x, ",res);
		if(iter==0x00)
			break;
	}
	printf("\n");
	
	for(iter=0x0f;iter>=0x00;iter--)
	{
		int i;
		uint8_t res = 0;
		uint8_t ti = iter<<4;
		for(i=0;i<8;i++)
		{
			uint8_t t = ti & mat[i];
			int r = bitwisexor4bit(t%16) ^ bitwisexor4bit(t/16);
			res |= r<<i;
		}
		printf("0x%02x, ",res);
		if(iter==0x00)
			break;
	}
	printf("\n");
}

void reversemap()
{
	uint8_t iter;
	uint8_t mat[] = {0x65, 0x8f, 0x59, 0x05, 0x7b, 0x8e, 0xd0, 0x86};
	for(iter=0x0f;iter>=0x00;iter--)
	{
		int i;
		uint8_t res = 0;
		for(i=0;i<8;i++)
		{
			uint8_t t = iter & mat[i];
			int r = bitwisexor4bit(t%16) ^ bitwisexor4bit(t/16);
			res |= r<<i;
		}
		printf("0x%02x, ",res^0x63);
		if(iter==0x00)
			break;
	}
	printf("\n");
	
	for(iter=0x0f;iter>=0x00;iter--)
	{
		int i;
		uint8_t res = 0;
		uint8_t ti = iter<<4;
		for(i=0;i<8;i++)
		{
			uint8_t t = ti & mat[i];
			int r = bitwisexor4bit(t%16) ^ bitwisexor4bit(t/16);
			res |= r<<i;
		}
		printf("0x%02x, ",res);
		if(iter==0x00)
			break;
	}
	printf("\n");
}

uint8_t doubleRes(uint8_t input)
{
	uint8_t res;
	
	res = input<<1;
	if((input&0x80)>>7 == 1)
		res ^= 0x1b;
	return res;
}

void revDouble()
{
	uint8_t iter;
	uint8_t mat[] = {0x65, 0x8f, 0x59, 0x05, 0x7b, 0x8e, 0xd0, 0x86};
	for(iter=0x0f;iter>=0x00;iter--)
	{
		int i;
		uint8_t res = 0;
		for(i=0;i<8;i++)
		{
			uint8_t t = iter & mat[i];
			int r = bitwisexor4bit(t%16) ^ bitwisexor4bit(t/16);
			res |= r<<i;
		}
		printf("0x%02x, ",doubleRes(res^0x63));
		if(iter==0x00)
			break;
	}
	printf("\n");
	
	for(iter=0x0f;iter>=0x00;iter--)
	{
		int i;
		uint8_t res = 0;
		uint8_t ti = iter<<4;
		for(i=0;i<8;i++)
		{
			uint8_t t = ti & mat[i];
			int r = bitwisexor4bit(t%16) ^ bitwisexor4bit(t/16);
			res |= r<<i;
		}
		printf("0x%02x, ",doubleRes(res));
		if(iter==0x00)
			break;
	}
	printf("\n");
}

int main()
{
	forwardmap();
	reversemap();
	revDouble();
}
