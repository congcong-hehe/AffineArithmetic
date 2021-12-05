#include "AffineArithmetic3.h"
#include <time.h>

int main()
{
	time_t begin, end;
	
	AANum a1(1.0, 3.0);
	a1.print();
	AANum a2(1.0, 3.0);
	a2.print();

	/*AANum a3 = a1 + a2;
	a3.print();
	AANum a4 = a1 - a2;
	a4.print();*/

	AANum a5;
	//a5 = a1 * a2;
	//a1.print();
	//a2.print();
	//a5 = a1 * a2;
	begin = clock();
	for (int i = 0; i < 10000; ++i)
	{
		a5 = a1 * a2;
	}
	end = clock();
	printf("%lldms\n", end - begin);
	a5.print();

	return 0;
}