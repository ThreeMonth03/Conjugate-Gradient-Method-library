#pragma once

namespace cg
{

enum class PreconditionerKind
{
    none,
    jacobi
};

constexpr const char *preconditioner_name(PreconditionerKind kind) noexcept
{
    switch (kind)
    {
    case PreconditionerKind::none:
        return "none";
    case PreconditionerKind::jacobi:
        return "jacobi";
    }
    return "unknown";
}

} // namespace cg
