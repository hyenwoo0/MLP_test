#include <iostream>

#include "MLP.h"


CMLP MultiLayer;

int main()
{
	int numofHiddenLayer = 1;
	int HiddenNode[1] = { 2 };

	MultiLayer.Create(2, HiddenNode, 1, numofHiddenLayer);

	MultiLayer.Forward();
}
