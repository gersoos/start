#include "utils.hpp"

#include <iostream>

namespace lane{

Operation::Operation()
{

}

Operation::~Operation()
{

}

void Operation::log(const std::string& msg){
	std::cout << msg << std::endl;
}

void Operation::error(const std::string& msg){
	std::cout << msg << std::endl;
	throw msg;
}

}
