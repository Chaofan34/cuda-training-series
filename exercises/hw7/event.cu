#include <math.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include "include/error.h"
#include "include/timer.h"

#define USE_STREAMS

class CudaEvent
{
private:
    cudaEvent_t e;

public:
    CudaEvent()
    {
        cudaEventCreate(&e);
        cudaCheckErrors("create event");
    };
    ~CudaEvent()
    {
        cudaEventDestroy(e);
    };

    // 公有方法：记录事件到指定流
    void record(cudaStream_t stream = 0)
    {
        cudaEventRecord(e, stream);
        cudaCheckErrors("cudaEventRecord failed");
    }

    // 公有方法：等待事件完成
    void synchronize()
    {
        cudaEventSynchronize(e);
        cudaCheckErrors("cudaEventSynchronize failed");
    }

    // 重载二元-运算符：计算两个事件的时间差（毫秒）
    float operator-(const CudaEvent &start)
    {
        this->synchronize();
        float elapsed_time;
        // 通过getter方法访问私有成员
        cudaEventElapsedTime(&elapsed_time, start.getEvent(), this->getEvent());
        cudaCheckErrors("cudaEventElapsedTime failed");
        return elapsed_time;
    }

private:
    // 私有getter：仅在类内部及友元中使用
    cudaEvent_t getEvent() const
    {
        return e;
    }
};

__global__ void stream1_print()
{
    for (auto i = 0; i < 50000; i++)
    {
    }
    printf("stream1\n");
}

__global__ void stream2_print()
{
    for (auto i = 0; i < 100000; i++)
    {
    }
    printf("stream2\n");
}

__global__ void stream3_print()
{
    for (auto i = 0; i < 10; i++)
    {
    }
    printf("stream3\n");
}

int main()
{
    cudaStream_t stream1, stream2, stream3;
    CudaEvent e1, e2;

    CudaEvent s3, e3;
    // cudaEvent_t s3, e3;
    // cudaEventCreate(&s3);
    // cudaEventCreate(&e3);

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    {
        auto t3 = TimeMonitor("stream3");
        s3.record(stream3);
        // cudaEventRecord(s3, stream3);
        {
            auto t = TimeMonitor("stream1/stream2");
            stream1_print<<<1, 1, 0, stream1>>>();
            e1.record(stream1);

            stream2_print<<<1, 1, 0, stream2>>>();
            e2.record(stream2);

            e1.synchronize();
            e2.synchronize();
        }

        stream3_print<<<1, 1, 0, stream3>>>();
        e3.record(stream3);
        e3.synchronize();

        // cudaEventRecord(e3, stream3);
        // cudaEventSynchronize(e3);
        // float elapse = 0.0;
        // cudaEventElapsedTime(&elapse, s3, e3);
        std::cout << "stream3, event cost: " << e3 - s3 << std::endl;
    }
    cudaDeviceSynchronize();

    return 0;
}
