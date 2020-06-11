#include "CNN.h"

Anime4KCPP::CNNProcessor* Anime4KCPP::CNNCreator::create(const CNNType& type)
{
	switch (type)
	{
	case CNNType::ACNet:
		return new ACNet();
	case CNNType::ACNetHDN:
		return new ACNetHDN();
	default:
		return nullptr;
	}
}

void Anime4KCPP::CNNCreator::release(CNNProcessor*& processor)
{
	if (processor != nullptr)
		delete processor;
}
