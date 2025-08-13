#include <time.h>
#include <string>
#include <iostream>

class TimeMonitor
{
private:
    clock_t t0;
    clock_t t1;
    std::string name;

public:
    TimeMonitor(std::string func_name) : name(func_name)
    {
        t0 = clock();
    }
    ~TimeMonitor()
    {
        t1 = clock();
        double sum = (double)(t1 - t0) / CLOCKS_PER_SEC;
        std::cout << "TimeMonitor::" << name << ", took " << sum << " seconds" << std::endl;
    }
};