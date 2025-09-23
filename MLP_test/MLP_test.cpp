#include <iostream>

#include "MLP.h"


CMLP MultiLayer;

int main()
{
	int numofHiddenLayer = 1;
	int HiddenNode[1] = { 2 };

	MultiLayer.Create(2, HiddenNode, 1, numofHiddenLayer);

	MultiLayer.m_Weight[0][0][1] = -7.3061;
	MultiLayer.m_Weight[0][1][1] = 4.7621;
	MultiLayer.m_Weight[0][2][1] = 4.7618;

	MultiLayer.m_Weight[0][0][2] = -2.8441;
	MultiLayer.m_Weight[0][1][2] = 6.3917;
	MultiLayer.m_Weight[0][2][2] = 6.3917;

	MultiLayer.m_Weight[1][0][1] = -4.5589;
	MultiLayer.m_Weight[1][1][1] = -10.3788;
	MultiLayer.m_Weight[1][2][1] = 9.7691;


	MultiLayer.Forward();
}
