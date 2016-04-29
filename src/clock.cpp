#include "clock.hpp"

//  Windows
#ifdef _WIN32
#include <Windows.h>
double get_wall_time(){
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time(){
    FILETIME a,b,c,d;
    if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
            (double)(d.dwLowDateTime |
            ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
    }else{
        //  Handle error
        return 0;
    }
}

//  Posix/Linux
#else
#include <sys/time.h>
#include <time.h>
double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}
#endif

void Clock::start()
{
	_last = get_wall_time();
	_elapsed = 0;
	_started = true;
}

double Clock::restart()
{
	double ret = timeElapsed();
	start();
	return ret;
}

double Clock::timeElapsed() {
	if (_started)
		return _elapsed + get_wall_time() - _last;
	else
		return _elapsed;

}

void Clock::pause() {
	_elapsed += get_wall_time() - _last;
	_started = false;
}

void Clock::resume() {
	_last = get_wall_time();
	_started = true;
}
