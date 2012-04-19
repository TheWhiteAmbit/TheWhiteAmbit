#pragma once
#include <varargs.h>
class delegat
{
    delegat(){}
    typedef void* (*methodType)(void*, int, va_list);
    methodType function;
	void* object;
    template <class T, void* (T::*TMethod)(int, va_list)> static void* staticMethod(void* object, int count, va_list args) {
        T* pointer = static_cast<T*>(object);
        return (pointer->*TMethod)(count, args);
    }
public:
	template <class T, void* (T::*TMethod)(int, va_list)> static delegat from(T* object) {
        delegat make;
		make.function = &staticMethod<T, TMethod>;
        make.object = object;
        return make;
    }
    void* operator()(int count, ...) const {
		va_list args;
		va_start(args, count);
        return (*function)(object, count, args);
    }
};
