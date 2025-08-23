#include <time.h>
#include <chrono>
#include <string>
#include <iostream>

using std::chrono::high_resolution_clock;
class TimeMonitor
{
private:
    //cpu time
    clock_t t0, t1;

    //wall time
    std::chrono::high_resolution_clock::time_point wt0, wt1;

    std::string name;

public:
    TimeMonitor(std::string func_name) : name(func_name)
    {
        t0 = clock();
        wt0 = high_resolution_clock::now();
    }
    ~TimeMonitor()
    {
        t1 = clock();
        wt1 = high_resolution_clock::now();
        double sum = (double)(t1 - t0) / CLOCKS_PER_SEC;
        auto wsum = (wt1 -wt0).count() / 1E9;
        std::cout << "TimeMonitor::" << name << ", took " << sum << " seconds," 
            << " walltime: " << wsum  << std::endl;
    }
};