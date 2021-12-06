#include "AffineArithmetic.h"
#include <stdlib.h>

__global__ void S_AA()
{
	AANum a0(1.0, 3.0);
	AANum a1(1.0, 3.0);
	a0.print();
	a1.print();
	AANum a3 = a0 * a1;
	a3.print();
}

int main()
{
	//int h_static_ser = 0;
	//cudaMemcpyToSymbol(&d_static_ser, &h_static_ser, sizeof(int));
	S_AA << <1, 1 >> > ();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	return 0;
}