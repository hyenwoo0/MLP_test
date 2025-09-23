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
	// 메모리 할당된거 해제 - 생성된 수 만큼 해제 주의 
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
				for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)	// 바이어스 포함  + 1
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
	m_iNumHiddenLayer = numHiddenLayer;		// 입력, 출력 레이어 제외한 히든 레이어 수
	m_iNumTotalLayer = numHiddenLayer + 2;	// input layer + hidden layer + output layer

	//m_NumNodes를 위한 메모리 할당
	m_NumNodes = (int*)malloc((m_iNumTotalLayer+1) * sizeof(int));

	m_NumNodes[0] = m_iNumInNodes;

	for (layer = 0; layer < m_iNumHiddenLayer; layer++)
		m_NumNodes[layer + 1] = pHiddenNode[layer];
	m_NumNodes[m_iNumTotalLayer - 1] = m_iNumOutNodes;
	m_NumNodes[m_iNumTotalLayer] = m_iNumOutNodes;
	

	//m_NodeOut 각 노드별 출력 메모리 할당 = [layer][node]
	m_NodeOut = (double**)malloc((m_iNumTotalLayer + 1) * sizeof(double*));
	for(layer = 0; layer < m_iNumTotalLayer; layer++)
		m_NodeOut[layer] = (double*)malloc((m_NumNodes[layer] + 1) * sizeof(double));	// 바이어스를 추가하기 위해 +1
	//정답( 출력노드와 같은 개수, 바이어스 필요없음)
	m_NodeOut[m_iNumTotalLayer] = (double*)malloc((m_NumNodes[m_iNumTotalLayer] + 1) * sizeof(double));		

	//m_Weight 메모리 할당 = [layer][시작노드][연결노드]
	m_Weight = (double***)malloc((m_iNumTotalLayer - 1) * sizeof(double**));
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		m_Weight[layer] = (double**)malloc((m_NumNodes[layer] + 1) * sizeof(double*));	// 바이어스 포함  + 1
		for(snode = 0; snode < m_NumNodes[layer] + 1; snode++)	// 바이어스 포함  + 1
			m_Weight[layer][snode] = (double*)malloc((m_NumNodes[layer + 1] + 1) * sizeof(double));		// 다음 레이어의 노드수 + 1

	}

	pInValue = m_NodeOut[0];	// 입력 레이어
	pOutValue = m_NodeOut[m_iNumTotalLayer - 1];	// 출력 레이어
	pCorrectOutValue = m_NodeOut[m_iNumTotalLayer];	// 정답 레이어

	InitW();	// 가중치 초기화

	return false;
}

void CMLP::InitW()
{
	int layer, snode, enode;
	// 랜덤으로 가중치 초기화 -1.0 ~ 1.0
	srand(time(NULL));

	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for(snode = 0; snode <= m_NumNodes[layer]; snode++)	 // for  바이오스 [0] 때문에 <=
			for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++) // 다음 레이어의 노드수	
				m_Weight[layer][snode][enode] = (double)rand() / RAND_MAX -0.5;	// -0.5 ~ 0.5


	}
}
