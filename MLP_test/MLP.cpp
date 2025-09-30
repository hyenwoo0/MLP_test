#include "MLP.h"
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

CMLP::CMLP()
{
	m_NumNodes = nullptr;
	m_Weight = nullptr;
	m_NodeOut = nullptr;

	pInValue = nullptr;
	pOutValue = nullptr;
	pCorrectOutValue = nullptr;

}
CMLP::~CMLP()
{
	int layer, snode, enode;
	// �޸� �Ҵ�Ȱ� ���� - ������ �� ��ŭ ���� ���� 
	// m_NumNodes
	if (m_NumNodes != nullptr)
	{
		free(m_NumNodes);
	}
	//m_NodeOut
	if(m_NodeOut != nullptr)
	{
		for (layer = 0; layer < m_iNumTotalLayer + 1; layer++)
			free(m_NodeOut[layer]);
		free(m_NodeOut);
	}
	//m_Weight
	if(m_Weight != nullptr)
	{
		for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
		{
			if (m_Weight[layer] != nullptr)
			{
				for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)	// ���̾ ����  + 1
					free(m_Weight[layer][snode]);
				
				free(m_Weight[layer]);
			}
		}
		free(m_Weight);
	}
}

bool CMLP::Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer)
{
	int layer, snode, enode;

	m_iNumInNodes = InNode;
	m_iNumOutNodes = OutNode;
	m_iNumHiddenLayer = numHiddenLayer;		// �Է�, ��� ���̾� ������ ���� ���̾� ��
	m_iNumTotalLayer = numHiddenLayer + 2;	// input layer + hidden layer + output layer

	//m_NumNodes�� ���� �޸� �Ҵ�
	m_NumNodes = (int*)malloc((m_iNumTotalLayer+1) * sizeof(int));

	m_NumNodes[0] = m_iNumInNodes;

	for (layer = 0; layer < m_iNumHiddenLayer; layer++)
		m_NumNodes[layer + 1] = pHiddenNode[layer];
	m_NumNodes[m_iNumTotalLayer - 1] = m_iNumOutNodes;
	m_NumNodes[m_iNumTotalLayer] = m_iNumOutNodes;
	

	//m_NodeOut �� ��庰 ��� �޸� �Ҵ� = [layer][node]
	m_NodeOut = (double**)malloc((m_iNumTotalLayer + 1) * sizeof(double*));
	for(layer = 0; layer < m_iNumTotalLayer; layer++)
		m_NodeOut[layer] = (double*)malloc((m_NumNodes[layer] + 1) * sizeof(double));	// ���̾�� �߰��ϱ� ���� +1
	//����( ��³��� ���� ����, ���̾ �ʿ����)
	m_NodeOut[m_iNumTotalLayer] = (double*)malloc((m_NumNodes[m_iNumTotalLayer] + 1) * sizeof(double));		

	//m_Weight �޸� �Ҵ� = [layer][���۳��][������]
	m_Weight = (double***)malloc((m_iNumTotalLayer - 1) * sizeof(double**));
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		m_Weight[layer] = (double**)malloc((m_NumNodes[layer] + 1) * sizeof(double*));	// ���̾ ����  + 1
		for(snode = 0; snode < m_NumNodes[layer] + 1; snode++)	// ���̾ ����  + 1
			m_Weight[layer][snode] = (double*)malloc((m_NumNodes[layer + 1] + 1) * sizeof(double));		// ���� ���̾��� ���� + 1

	}

	pInValue = m_NodeOut[0];	// �Է� ���̾�
	pOutValue = m_NodeOut[m_iNumTotalLayer - 1];	// ��� ���̾�
	pCorrectOutValue = m_NodeOut[m_iNumTotalLayer];	// ���� ���̾�

	InitW();	// ����ġ �ʱ�ȭ

	// ���̾�� ���� ��°� = 1
	for (layer = 0; layer < m_iNumTotalLayer + 1; layer++)
		m_NodeOut[layer][0] = 1;

	return false;
}

void CMLP::InitW()
{
	int layer, snode, enode;
	// �������� ����ġ �ʱ�ȭ -1.0 ~ 1.0
	srand(time(NULL));

	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for(snode = 0; snode <= m_NumNodes[layer]; snode++)	 // for  ���̿��� [0] ������ <=
			for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++) // ���� ���̾��� ����	
				m_Weight[layer][snode][enode] = (double)rand() / RAND_MAX -0.5;	// -0.5 ~ 0.5


	}
}

void CMLP::Forward()
{
	int layer, snode, enode;
	double wsum;

	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		// ���� ���̾��� �� ��庰�� ��°��� ����ϱ� ���� enode ���� ����
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++) // ���̾�� ���� 0����
		{
			wsum = 0.0; // ��庰 ������
			wsum = m_Weight[layer][0][enode] * 1; // ���̾ ����ġ ��
			for (snode = 1; snode <= m_NumNodes[layer]; snode++)	 // 
				wsum += m_Weight[layer][snode][enode] * m_NodeOut[layer][snode];
			
			// Ȱ��ȭ �Լ�
			m_NodeOut[layer + 1][enode] = ActivationFunc(wsum);
		}
	}
}

double CMLP::ActivationFunc(double weightsum)
{
	// step function
	//return (weightsum > 0) ? 1.0 : 0.0;		// ���� �����ڸ� ���� ��� �Լ�

	// sigmoid function
	return 1.0 / (1.0 + exp(-weightsum));
}
