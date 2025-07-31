from __future__ import annotations

import math
import operator as op
from copy import copy
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Sequence,
    SupportsAbs,
    SupportsIndex,
    SupportsRound,
    TypeAlias,
    Union,
    cast,
    overload,
)

from vstools import R, SupportsFloatOrIndex, SupportsRichComparison, SupportsTrunc, T

from .exprop import ExprOp

if TYPE_CHECKING:
    from .variables import ComputedVar, ExprOtherT, ExprVar

__all__ = [
    "BaseOperator",
    "BinaryBaseOperator",
    "BinaryBoolOperator",
    "BinaryMathOperator",
    "BinaryOperator",
    "ExprOperators",
    "TernaryBaseOperator",
    "TernaryCompOperator",
    "TernaryIfOperator",
    "TernaryOperator",
    "TernaryPixelAccessOperator",
    "UnaryBaseOperator",
    "UnaryBoolOperator",
    "UnaryMathOperator",
    "UnaryOperator",
]

SuppRC: TypeAlias = SupportsRichComparison


def _norm_lit(arg: str | ExprOtherT | BaseOperator) -> ExprVar | BaseOperator:
    from .variables import ExprVar, LiteralVar

    if isinstance(arg, (ExprVar, BaseOperator)):
        return arg

    return LiteralVar(arg)


def _normalize_args(*args: str | ExprOtherT | BaseOperator) -> Iterable[ExprVar | BaseOperator]:
    for arg in args:
        yield _norm_lit(arg)


@dataclass
class BaseOperator:
    rpn_name: ExprOp

    def to_str(self, **kwargs: Any) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.rpn_name


class UnaryBaseOperator(BaseOperator):
    def __call__(self, x: ExprOtherT) -> ComputedVar:
        from .variables import ComputedVar

        return ComputedVar(_normalize_args(x, self))  # pyright: ignore[reportArgumentType]


class BinaryBaseOperator(BaseOperator):
    def __call__(self, x: ExprOtherT, y: ExprOtherT) -> ComputedVar:
        from .variables import ComputedVar

        return ComputedVar(_normalize_args(x, y, self))  # pyright: ignore[reportArgumentType]


class TernaryBaseOperator(BaseOperator):
    def __call__(self, x: ExprOtherT, y: ExprOtherT, z: ExprOtherT) -> ComputedVar:
        from .variables import ComputedVar

        return ComputedVar(_normalize_args(x, y, z, self))  # pyright: ignore[reportArgumentType]


@dataclass
class UnaryOperator(Generic[T], UnaryBaseOperator):
    function: Callable[[T], T]


@dataclass
class UnaryMathOperator(Generic[T, R], UnaryBaseOperator):
    function: Callable[[T], R]


@dataclass
class UnaryBoolOperator(UnaryBaseOperator):
    function: Callable[[object], bool]


@dataclass
class BinaryOperator(Generic[T, R], BinaryBaseOperator):
    function: Callable[[T, R], T | R]


@dataclass
class BinaryMathOperator(Generic[T, R], BinaryBaseOperator):
    function: Callable[[T, T], R]


@dataclass
class BinaryBoolOperator(BinaryBaseOperator):
    function: Callable[[Any, Any], bool]


@dataclass
class TernaryOperator(Generic[T, R], TernaryBaseOperator):
    function: Callable[[bool, T, R], T | R]


@dataclass
class TernaryIfOperator(TernaryOperator["ExprOtherT", "ExprOtherT"]):
    def __call__(self, cond: ExprOtherT, if_true: ExprOtherT, if_false: ExprOtherT) -> ComputedVar:
        return super().__call__(cond, if_true, if_false)


@dataclass
class TernaryCompOperator(TernaryBaseOperator):
    function: Callable[[SuppRC, SuppRC, SuppRC], SuppRC]


@dataclass
class TernaryClampOperator(TernaryCompOperator):
    def __call__(self, x: ExprOtherT, min: ExprOtherT, max: ExprOtherT) -> ComputedVar:
        return super().__call__(x, min, max)


class TernaryPixelAccessOperator(Generic[T], TernaryBaseOperator):
    char: str
    x: T
    y: T

    def __call__(self, char: str, x: T, y: T) -> ComputedVar:  # type: ignore
        from .variables import ComputedVar

        self.set_vars(char, x, y)
        return ComputedVar([copy(self)])  # pyright: ignore[reportArgumentType]

    def set_vars(self, char: str, x: T, y: T) -> None:
        self.char = char
        self.x = x
        self.y = y

    def __str__(self) -> str:
        if not hasattr(self, "char"):
            raise ValueError("TernaryPixelAccessOperator: You have to call set_vars!")

        return self.rpn_name.format(char=str(self.char), x=int(self.x), y=int(self.y))  # type: ignore[call-overload]


class ExprOperators:
    # 1 Argument
    @staticmethod
    def EXP(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates e**x for every element in x."""
        return UnaryMathOperator(ExprOp.EXP, math.exp)(x)

    @staticmethod
    def LOG(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the natural logarithm for every element in x."""
        return UnaryMathOperator(ExprOp.LOG, math.log)(x)

    @staticmethod
    def SQRT(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the square root for every element in x."""
        return UnaryMathOperator(ExprOp.SQRT, math.sqrt)(x)

    @staticmethod
    def SIN(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the sine for every element in x."""
        return UnaryMathOperator(ExprOp.SIN, math.sin)(x)

    @staticmethod
    def COS(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the cosine for every element in x."""
        return UnaryMathOperator(ExprOp.COS, math.cos)(x)

    @staticmethod
    def ABS(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the absolute value for every element in x."""
        return UnaryMathOperator[SupportsAbs[SupportsIndex], SupportsIndex](ExprOp.ABS, abs)(x)

    @staticmethod
    def NOT(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a logical NOT on every element in x."""
        return UnaryBoolOperator(ExprOp.NOT, op.not_)(x)

    @staticmethod
    def DUP(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Duplicates the top of the stack."""
        return UnaryBaseOperator(ExprOp.DUP)(x)

    @staticmethod
    def DUPN(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Duplicates the nth element from the top of the stack."""
        return UnaryBaseOperator(ExprOp.DUPN)(x)

    @staticmethod
    def TRUNC(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Truncates every element in x."""
        return UnaryMathOperator[SupportsTrunc, int](ExprOp.TRUNC, math.trunc)(x)

    @staticmethod
    def ROUND(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Rounds every element in x."""
        return UnaryMathOperator[SupportsRound[int], int](ExprOp.ROUND, lambda val: round(val))(x)

    @staticmethod
    def FLOOR(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Floors every element in x."""
        return UnaryMathOperator[SupportsFloatOrIndex, int](ExprOp.FLOOR, math.floor)(x)

    # DROP / DROPN / SORTN / VAR_STORE / VAR_PUSH ??

    # 2 Arguments
    @staticmethod
    def MAX(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the maximum of x and y."""
        return BinaryMathOperator[SuppRC, SuppRC](ExprOp.MAX, max)(x, y)

    @staticmethod
    def MIN(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Calculates the minimum of x and y."""
        return BinaryMathOperator[SuppRC, SuppRC](ExprOp.MIN, min)(x, y)

    @staticmethod
    def ADD(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs addition of two elements (x + y)."""
        return BinaryOperator(ExprOp.ADD, op.add)(x, y)

    @staticmethod
    def SUB(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs subtraction of two elements (x - y)."""
        return BinaryOperator(ExprOp.SUB, op.sub)(x, y)

    @staticmethod
    def MUL(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs multiplication of two elements (x * y)."""
        return BinaryOperator(ExprOp.MUL, op.mul)(x, y)

    @staticmethod
    def DIV(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs division of two elements (x / y)."""
        return BinaryOperator(ExprOp.DIV, op.truediv)(x, y)

    @staticmethod
    def POW(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a to the power of b (x ** y)."""
        return BinaryOperator(ExprOp.POW, op.pow)(x, y)

    @staticmethod
    def GT(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs x > y."""
        return BinaryBoolOperator(ExprOp.GT, op.gt)(x, y)

    @staticmethod
    def LT(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs x < y."""
        return BinaryBoolOperator(ExprOp.LT, op.lt)(x, y)

    @staticmethod
    def EQ(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs x == y."""
        return BinaryBoolOperator(ExprOp.EQ, op.eq)(x, y)

    @staticmethod
    def GTE(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs x >= y."""
        return BinaryBoolOperator(ExprOp.GTE, op.ge)(x, y)

    @staticmethod
    def LTE(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs x <= y."""
        return BinaryBoolOperator(ExprOp.LTE, op.le)(x, y)

    @staticmethod
    def AND(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a logical AND."""
        return BinaryBoolOperator(ExprOp.AND, op.and_)(x, y)

    @staticmethod
    def OR(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a logical OR."""
        return BinaryBoolOperator(ExprOp.OR, op.or_)(x, y)

    @staticmethod
    def XOR(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a logical XOR."""
        return BinaryOperator(ExprOp.XOR, op.xor)(x, y)

    @staticmethod
    def SWAP(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Swaps the top two elements of the stack."""
        return BinaryBaseOperator(ExprOp.SWAP)(x, y)

    @staticmethod
    def SWAPN(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Swaps the top element with the nth element from the top of the stack."""
        return BinaryBaseOperator(ExprOp.SWAPN)(x, y)

    @staticmethod
    def MOD(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a % b."""
        return BinaryOperator(ExprOp.MOD, op.mod)(x, y)

    @staticmethod
    def BITNOT(x: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a bitwise NOT on every element in x."""
        # //TODO Type error here
        return UnaryOperator(ExprOp.BITNOT, op.inv)(x)

    @staticmethod
    def BITAND(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a bitwise AND."""
        return BinaryOperator(ExprOp.BITAND, op.and_)(x, y)

    @staticmethod
    def BITOR(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a bitwise OR."""
        return BinaryOperator(ExprOp.BITOR, op.or_)(x, y)

    @staticmethod
    def BITXOR(x: ExprOtherT, y: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs a bitwise XOR."""
        return BinaryOperator(ExprOp.BITXOR, op.xor)(x, y)

    # 3 Arguments
    @staticmethod
    def TERN(cond: ExprOtherT, if_true: ExprOtherT, if_false: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Ternary operator (if cond then if_true else if_false)."""
        return TernaryIfOperator(ExprOp.TERN, lambda x, y, z: (x if z else y))(cond, if_true, if_false)

    @staticmethod
    def CLAMP(value: ExprOtherT, min_val: ExprOtherT, max_val: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Clamps a value between a min and a max."""
        return TernaryCompOperator(ExprOp.CLAMP, lambda x, y, z: max(y, min(x, z)))(value, min_val, max_val)

    # Aliases
    IF = TERN
    """Ternary operator (if cond then if_true else if_false)."""

    # Special Operators
    REL_PIX = TernaryPixelAccessOperator[int](ExprOp.REL_PIX)
    """Relative pixel access."""

    ABS_PIX = TernaryPixelAccessOperator[Union[int, "ExprVar"]](ExprOp.ABS_PIX)
    """Absolute pixel access."""

    @staticmethod
    def LERP(x: ExprOtherT, y: ExprOtherT, z: ExprOtherT) -> ComputedVar:  # noqa: N802
        """Performs linear interpolation. `x y z lerp` is equivalent to `x * z + y * (1 - z)`."""
        return TernaryBaseOperator(ExprOp.LERP)(x, y, z)

    # N Arguments
    @staticmethod
    def POLYVAL(x: ExprOtherT, *coeffs: ExprOtherT) -> ComputedVar:
        """
        Evaluates a polynomial
        `c0 c1 ... cN-1 x polyvalN`
        Usage: POLYVAL(x, c0, c1, ..., cN)
        """
        from .variables import ComputedVar

        n = len(coeffs) - 1
        op_string = f"{ExprOp.POLYVAL}{n}"

        return ComputedVar(_normalize_args(*coeffs, x, op_string))

    POLYEVAL = POLYVAL

    # Helper Functions

    @overload
    @classmethod
    def as_var(cls, arg0: ExprOtherT) -> ComputedVar:
        pass

    @overload
    @classmethod
    def as_var(cls, arg0: Sequence[ExprOtherT]) -> list[ComputedVar]:
        pass

    @classmethod
    def as_var(cls, arg0: ExprOtherT | Sequence[ExprOtherT]) -> ComputedVar | list[ComputedVar]:
        from .variables import ComputedVar

        if isinstance(arg0, Sequence):
            return cast(list[ComputedVar], list(arg0))
        return cast(ComputedVar, arg0)
