#pragma once

class CMLP
{
public:
	CMLP();		// 생성자, 소멸자 선언
	~CMLP();


public:		// 신경망 구조 선언을 위한 변수 
	int m_iNumInNodes;		// 입력 노드의 수
	int m_iNumOutNodes;		// 출력 노드의 수
	int m_iNumHiddenLayer;	// 히든 레이어의 수(hidden only)
	int m_iNumTotalLayer;	// 전체 레이어의 수(inputlayer + hidden layer + output layer)

	int* m_NumNodes;		// 배열로 하면 메모리가 남을 수 있어, 포인터로 선언 - 메모리 관리가 편함
	// [0]-input node, [1]-hidden node1, ..., [n]-output node

	double*** m_Weight;		// [시작 레이어][시작노드][연결노드]
	double** m_NodeOut;		// [layer][node]  노드가 0일때는 바이어스

	double *pInValue, *pOutValue;		// 입력 레이어, 출력 레이어	
	double* pCorrectOutValue;			// 정답 레이어

	bool Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer);
private:
	void InitW();
	double ActivationFunc(double weightsum);
public:
	void Forward();


};