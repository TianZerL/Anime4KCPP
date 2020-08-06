#include "CNN.h"

Anime4KCPP::CNNProcessor* Anime4KCPP::CNNCreator::create(const CNNType& type)
{
	switch (type)
	{
	case CNNType::ACNet:
		return new ACNet();
	case CNNType::ACNetHDNL1:
		return new ACNetHDNL1();
	case CNNType::ACNetHDNL2:
		return new ACNetHDNL2();
	case CNNType::ACNetHDNL3:
		return new ACNetHDNL3();
	default:
		return nullptr;
	}
}

void Anime4KCPP::CNNCreator::release(CNNProcessor*& processor)
{
	if (processor != nullptr)
	{
		delete processor;
		processor = nullptr;
	}
}
