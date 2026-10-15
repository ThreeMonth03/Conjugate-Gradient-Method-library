#pragma once

namespace cg
{
namespace test
{

// A test-executable-local, thread-local allocation fault injector. Assertions
// run after this guard is destroyed so GoogleTest may allocate normally.
class AllocationFailureGuard
{
    public:
    explicit AllocationFailureGuard(int successful_allocations);
    ~AllocationFailureGuard();
    AllocationFailureGuard(const AllocationFailureGuard &) = delete;
    AllocationFailureGuard &operator=(const AllocationFailureGuard &) = delete;
};

} // namespace test
} // namespace cg
