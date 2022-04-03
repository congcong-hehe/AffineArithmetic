#include "RAA.h"
#include "AAList.h"
#include "AAVec.h"

using AANum = RAA3;

int main()
{
	AANum a1(1.0, 2.0);
	AANum a2(2.0, 3.0);

	AANum a3 = a1 + a2;
	AANum a4 = a1 * a2;

	double low = 0, high = 0;
	a3.toIA(&low, &high);
	printf("%f %f\n", low, high);
	a4.toIA(&low, &high);
	printf("%f %f", low, high);


	return 0;
}