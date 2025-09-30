#pragma once

class CMLP
{
public:
	CMLP();		// ������, �Ҹ��� ����
	~CMLP();


public:		// �Ű�� ���� ������ ���� ���� 
	int m_iNumInNodes;		// �Է� ����� ��
	int m_iNumOutNodes;		// ��� ����� ��
	int m_iNumHiddenLayer;	// ���� ���̾��� ��(hidden only)
	int m_iNumTotalLayer;	// ��ü ���̾��� ��(inputlayer + hidden layer + output layer)

	int* m_NumNodes;		// �迭�� �ϸ� �޸𸮰� ���� �� �־�, �����ͷ� ���� - �޸� ������ ����
	// [0]-input node, [1]-hidden node1, ..., [n]-output node

	double*** m_Weight;		// [���� ���̾�][���۳��][������]
	double** m_NodeOut;		// [layer][node]  ��尡 0�϶��� ���̾

	double *pInValue, *pOutValue;		// �Է� ���̾�, ��� ���̾�	
	double* pCorrectOutValue;			// ���� ���̾�

	bool Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer);
private:
	void InitW();
	double ActivationFunc(double weightsum);
public:
	void Forward();


};