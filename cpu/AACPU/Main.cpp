//#include "AffineArithmetic_vec.h"
#include "AffineArithmecit_list.h"
#include <time.h>

int main()
{
	time_t begin, end;
	Affine* affine = new Affine;
	// base test
	/*{
		AANum aa0;
		affine->IAToAA(1, 3, aa0);
		aa0.print();
		AANum aa1;
		affine->IAToAA(1, 3, aa1);
		aa1.print();
		AANum aa2;
		affine->add(aa0, aa1, aa2);
		aa2.print();

		double low = 0, high = 0;
		affine->AAToIA(aa2, low, high);
		printf("%f %f\n", low, high);

		AANum aa3;
		affine->sub(aa0, aa1, aa3);
		aa3.print();
	}*/

	// test mul
	/*{
		AANum aa0;
		affine->IAToAA(1, 3, aa0);
		AANum aa1;
		affine->IAToAA(1, 3, aa1);

		AANum aa2;
		affine->mul(aa0, aa1, aa2);
		aa2.print();
	}*/

	// vec 时间测试 140ms
	/*{
		AANum aa0;
		affine->IAToAA(1, 3, aa0);
		AANum aa1;
		affine->IAToAA(1, 3, aa1);

		begin = clock();
		for (int i = 0; i < 10000; ++i)
		{
			AANum aa2;
			affine->mul(aa0, aa1, aa2);
		}
		end = clock();
		printf("%lld ms\n", end - begin);

	}*/

	// list 时间测试 250ms
	{
		shared_ptr<AANum> aa0 = affine->IAConvtAA(1.0, 3.0);
		shared_ptr<AANum> aa1 = affine->IAConvtAA(1.0, 3.0);
		shared_ptr<AANum> aa2;
		begin = clock();
		for (int i = 0; i < 10000; ++i)
		{
			aa2 = affine->Mul(aa0, aa1);
		}
		end = clock();
		printf("%lld ms\n", end - begin);
	}

	return 0;
}