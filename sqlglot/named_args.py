"""
Rewrite UDF calls that use named arguments (``f(x => 1)``) into positional calls.

Callers register the signatures of the functions they want to migrate. A signature
declares the positional order of the parameters and, optionally, a default
expression for parameters that may be omitted. The rewrite only touches calls that
match a registered signature; everything else is left untouched so the resulting
AST can be handed back to SQLGlot for normal SQL generation.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import SqlglotError

if t.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class NamedArgumentError(SqlglotError):
    """Raised when a named-argument call cannot be rewritten as a positional call."""


# Nondeterministic / side-effecting expressions. Reordering these, or interleaving
# them with other volatile evaluations, can change results, so such rewrites are
# rejected instead of being applied silently.
VOLATILE_EXPRESSION_TYPES = (exp.Rand, exp.Randn, exp.Randstr, exp.Uuid, exp.NextValueFor)

VOLATILE_FUNCTION_NAMES = {
    "GEN_RANDOM_UUID",
    "NEWID",
    "NEWSEQUENTIALID",
    "NEXTVAL",
    "RAND",
    "RANDOM",
    "SECURE_RANDOM",
    "UUID",
    "UUID_STRING",
}


def _default_is_volatile(node: exp.Expr) -> bool:
    return isinstance(node, VOLATILE_EXPRESSION_TYPES) or (
        isinstance(node, exp.Anonymous) and node.name.upper() in VOLATILE_FUNCTION_NAMES
    )


def _contains_volatile(
    expression: exp.Expr, is_volatile: t.Callable[[exp.Expr], bool]
) -> bool:
    return any(is_volatile(node) for node in expression.walk())


@dataclass
class Parameter:
    """A single formal parameter of a registered function signature.

    Args:
        name: the parameter name as it appears in named-argument calls.
        default: optional default expression, used to fill the parameter's slot when a
            call omits it but a later parameter is supplied. Accepts an expression or a
            SQL string, which is parsed when the signature is registered.
    """

    name: str
    default: exp.Expr | str | None = None


ParameterLike = t.Union[Parameter, str, t.Tuple[str, t.Any]]


class FunctionSignature:
    """The ordered parameters of a registered function."""

    def __init__(self, parameters: Iterable[ParameterLike]) -> None:
        self.parameters = [self._coerce(parameter) for parameter in parameters]

    @staticmethod
    def _coerce(parameter: ParameterLike) -> Parameter:
        if isinstance(parameter, Parameter):
            return parameter
        if isinstance(parameter, str):
            return Parameter(parameter)
        name, *rest = parameter
        return Parameter(name, *rest)


def _call_identifiers(anonymous: exp.Anonymous) -> list[exp.Identifier] | None:
    """Return the (possibly qualified) name parts of a call, innermost part last."""
    node: exp.Expr = anonymous
    parent = node.parent
    while isinstance(parent, exp.Dot) and parent.expression is node:
        node = parent
        parent = node.parent

    parts = list(node.flatten()) if isinstance(node, exp.Dot) else [node]

    identifiers = []
    for part in parts:
        if isinstance(part, exp.Anonymous):
            this = part.this
            if isinstance(this, exp.Identifier):
                identifiers.append(this)
            elif isinstance(this, str):
                identifiers.append(exp.Identifier(this=this, quoted=False))
            else:
                return None
        elif isinstance(part, exp.Identifier):
            identifiers.append(part)
        else:
            return None

    return identifiers


class SignatureRegistry:
    """Caller-registered UDF signatures used to rewrite named arguments.

    Args:
        dialect: dialect used to normalize function and parameter names when matching
            (e.g. Snowflake uppercases unquoted identifiers, quoted ones stay
            case-sensitive) and to parse SQL string defaults.
    """

    def __init__(self, dialect: DialectType = None) -> None:
        self.dialect = Dialect.get_or_raise(dialect)
        self._signatures: dict[tuple[str, ...], FunctionSignature] = {}

    def register(
        self, name: str, parameters: Iterable[ParameterLike]
    ) -> SignatureRegistry:
        """Register a signature for a (possibly qualified) function name."""
        signature = FunctionSignature(parameters)
        for parameter in signature.parameters:
            if isinstance(parameter.default, str):
                parameter.default = exp.maybe_parse(parameter.default, dialect=self.dialect)

        table = exp.to_table(name, dialect=self.dialect)
        if not table or any(not isinstance(part, exp.Identifier) for part in table.parts):
            raise NamedArgumentError(f"Invalid function name: {name}")

        self._signatures[self._key(table.parts)] = signature
        return self

    def get(self, identifiers: list[exp.Identifier]) -> FunctionSignature | None:
        return self._signatures.get(self._key(identifiers))

    def _key(self, identifiers: list[exp.Identifier]) -> tuple[str, ...]:
        return tuple(self._normalize(identifier) for identifier in identifiers)

    def _normalize(self, identifier: exp.Identifier) -> str:
        return self.dialect.normalize_identifier(identifier.copy()).name

    def normalize_name(self, name: str) -> str:
        return self._normalize(exp.to_identifier(name))


def _bind_arguments(
    registry: SignatureRegistry,
    signature: FunctionSignature,
    display_name: str,
    args: list[exp.Expr],
) -> tuple[list[exp.Expr], list[exp.Expr], list[exp.Expr]]:
    """Bind call arguments to signature slots.

    Returns the new positional argument list, the user-supplied arguments in their
    original (textual) order, and the inserted default expressions.
    """
    parameters = signature.parameters
    index_by_name: dict[str, int] = {}
    for index, parameter in enumerate(parameters):
        normalized = registry.normalize_name(parameter.name)
        if normalized in index_by_name:
            raise NamedArgumentError(
                f"Duplicate parameter '{parameter.name}' in signature of {display_name}"
            )
        index_by_name[normalized] = index

    slots: list[exp.Expr | None] = [None] * len(parameters)
    user_args: list[exp.Expr] = []
    seen_named = False

    for arg in args:
        if isinstance(arg, exp.Kwarg):
            seen_named = True
            name = arg.this.name
            index = index_by_name.get(registry.normalize_name(name))
            if index is None:
                raise NamedArgumentError(
                    f"Unknown argument '{name}' for function {display_name}"
                )
            if slots[index] is not None:
                raise NamedArgumentError(
                    f"Argument '{name}' of function {display_name} got multiple values"
                )
            slots[index] = arg.expression
            user_args.append(arg.expression)
        else:
            if seen_named:
                raise NamedArgumentError(
                    f"Positional argument follows named argument in call to {display_name}"
                )
            position = len(user_args)
            if position >= len(parameters):
                raise NamedArgumentError(
                    f"Too many positional arguments in call to {display_name}"
                )
            slots[position] = arg
            user_args.append(arg)

    filled = [index for index, slot in enumerate(slots) if slot is not None]
    last = max(filled, default=-1)

    new_args: list[exp.Expr] = []
    inserted: list[exp.Expr] = []
    for index in range(last + 1):
        slot = slots[index]
        if slot is None:
            default = parameters[index].default
            if default is None:
                raise NamedArgumentError(
                    f"Missing required argument '{parameters[index].name}' for function "
                    f"{display_name}"
                )
            # Copy so a registered default is never shared between ASTs
            slot = t.cast(exp.Expr, default).copy()
            inserted.append(slot)
        new_args.append(slot)

    return new_args, user_args, inserted


def named_args_to_positional(
    expression: exp.Expr,
    signatures: SignatureRegistry | Mapping[str, Iterable[ParameterLike]],
    dialect: DialectType = None,
    copy: bool = True,
    is_volatile: t.Callable[[exp.Expr], bool] = _default_is_volatile,
) -> exp.Expr:
    """Rewrite registered named-argument calls into positional calls.

    Args:
        expression: the AST to transform.
        signatures: a ``SignatureRegistry``, or a mapping of (possibly qualified)
            function names to parameter lists, e.g.
            ``{"my_schema.my_func": ["a", ("b", "0")]}``.
        dialect: dialect used for name matching when ``signatures`` is a mapping.
        copy: if True, transform a copy of ``expression`` and leave the input AST
            unchanged.
        is_volatile: predicate marking nondeterministic / side-effecting expressions.
            A rewrite that would change the evaluation order of volatile expressions
            (reordering volatile arguments, or interleaving them with volatile
            defaults) is rejected with a ``NamedArgumentError``.

    Returns:
        The transformed expression.
    """
    if isinstance(signatures, SignatureRegistry):
        registry = signatures
    else:
        registry = SignatureRegistry(dialect=dialect)
        for name, parameters in signatures.items():
            registry.register(name, parameters)

    result = expression.copy() if copy else expression

    for anonymous in list(result.find_all(exp.Anonymous)):
        args = anonymous.expressions
        if not any(isinstance(arg, exp.Kwarg) for arg in args):
            continue

        identifiers = _call_identifiers(anonymous)
        if identifiers is None:
            continue

        signature = registry.get(identifiers)
        if signature is None:
            continue

        display_name = ".".join(identifier.name for identifier in identifiers)
        new_args, user_args, inserted = _bind_arguments(
            registry, signature, display_name, args
        )

        if any(_contains_volatile(arg, is_volatile) for arg in user_args):
            user_arg_ids = {id(arg) for arg in user_args}
            reordered = [id(arg) for arg in new_args if id(arg) in user_arg_ids] != [
                id(arg) for arg in user_args
            ]
            if reordered:
                raise NamedArgumentError(
                    f"Refusing to reorder volatile arguments in call to {display_name}: "
                    "evaluation order affects results"
                )
            if any(_contains_volatile(default, is_volatile) for default in inserted):
                raise NamedArgumentError(
                    f"Refusing to interleave volatile defaults with volatile arguments "
                    f"in call to {display_name}: evaluation order affects results"
                )

        anonymous.set("expressions", new_args)

    return result
