#include <support/allocation_failure.hpp>

#include <cstdlib>
#include <new>

namespace
{
thread_local int allocations_before_failure = -1;
} // namespace

namespace cg
{
namespace test
{
AllocationFailureGuard::AllocationFailureGuard(int successful_allocations)
{
    allocations_before_failure = successful_allocations;
}
AllocationFailureGuard::~AllocationFailureGuard()
{
    allocations_before_failure = -1;
}
} // namespace test
} // namespace cg

// Link this file only into the isolated allocation-failure test executable.
void *operator new(std::size_t size)
{
    if (allocations_before_failure == 0)
    {
        throw std::bad_alloc();
    }
    if (allocations_before_failure > 0)
    {
        --allocations_before_failure;
    }
    if (void *allocation = std::malloc(size == 0 ? 1 : size))
    {
        return allocation;
    }
    throw std::bad_alloc();
}
void *operator new[](std::size_t size)
{
    return ::operator new(size);
}
void *operator new(std::size_t size, const std::nothrow_t &) noexcept
{
    try
    {
        return ::operator new(size);
    }
    catch (...)
    {
        return nullptr;
    }
}
void *operator new[](std::size_t size, const std::nothrow_t &tag) noexcept
{
    return ::operator new(size, tag);
}
void operator delete(void *allocation) noexcept
{
    std::free(allocation);
}
void operator delete[](void *allocation) noexcept
{
    std::free(allocation);
}
void operator delete(void *allocation, std::size_t) noexcept
{
    std::free(allocation);
}
void operator delete[](void *allocation, std::size_t) noexcept
{
    std::free(allocation);
}
void operator delete(void *allocation, const std::nothrow_t &) noexcept
{
    std::free(allocation);
}
void operator delete[](void *allocation, const std::nothrow_t &) noexcept
{
    std::free(allocation);
}
