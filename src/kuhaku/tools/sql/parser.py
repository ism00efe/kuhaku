"""A parser for the restricted, read-only SELECT subset the SQL tool accepts.

This is an allow-list grammar, not a general SQL parser: anything it does not
recognise is a syntax error, and a syntax error is a rejection. That is the point --
a permissive parser that accepts every dialect's quirks has to be paired with a deny
list of the dangerous ones, and a deny list is only as good as its author's memory.

What the grammar covers::

    query       := [WITH cte ("," cte)*] core (("UNION" ["ALL"] | "INTERSECT" | "EXCEPT") core)*
                   ["ORDER BY" order ("," order)*] ["LIMIT" int ["OFFSET" int]]
    cte         := name "AS" "(" query ")"
    core        := "SELECT" ["DISTINCT" | "ALL"] result ("," result)*
                   ["FROM" from] ["WHERE" expr] ["GROUP BY" expr ("," expr)*] ["HAVING" expr]
    from        := item (("," | [INNER | LEFT [OUTER] | CROSS] "JOIN") item ["ON" expr])*
    item        := name [["AS"] alias] | "(" query ")" [["AS"] alias]

and expressions with literals, column references, whitelisted function calls,
arithmetic, comparison, ``IS``/``IN``/``BETWEEN``/``LIKE``, ``EXISTS``, ``CASE``,
``CAST`` and scalar subqueries. Comments, parameters, statement separators, blob and
hex literals, schema-qualified table names, ``RECURSIVE``, window functions, ``USING``,
``NATURAL``, ``RIGHT``/``FULL`` joins and every non-SELECT statement are not in it.

The parser only builds the tree. Name resolution, entitlement, the row cap and the
SQL that is finally executed are :mod:`.validator`'s -- the executed text is rendered
from this tree, never passed through from the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MAX_SQL_LENGTH = 20_000

# LIMIT/OFFSET and type sizes above a signed 64-bit integer overflow in the database
# (SQLite raises an uncaught OverflowError from the driver; PostgreSQL rejects it).
MAX_INT_LITERAL = 2**63 - 1

# Control characters have no business in an identifier; NUL in particular is the
# validator's internal placeholder mark.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
# Recursion guard: a pathological nesting of parentheses or subqueries must be a clean
# rejection, not a RecursionError.
MAX_DEPTH = 64


class SQLSyntaxError(ValueError):
    """The text is not in the accepted subset. ``code`` is a stable machine-readable
    reason; the message is safe to show the model (it names only what the text says)."""

    def __init__(self, message: str, *, code: str = "syntax") -> None:
        super().__init__(message)
        self.code = code


# --- tokens ---------------------------------------------------------------------------

RESERVED = frozenset(
    """
    ALL AND AS ASC BETWEEN BY CASE CAST CROSS DESC DISTINCT ELSE END ESCAPE EXCEPT
    EXISTS FALSE FROM FULL GROUP HAVING IN INNER INTERSECT IS JOIN LEFT LIKE LIMIT
    NATURAL NOT NULL NULLS OFFSET ON OR ORDER OUTER RECURSIVE RIGHT SELECT THEN TRUE
    UNION USING WHEN WHERE WITH
    INSERT UPDATE DELETE CREATE DROP ALTER ATTACH DETACH PRAGMA VACUUM INTO VALUES
    GLOB MATCH REGEXP COLLATE
    """.split()  # noqa: SIM905 -- a word list reads better than quoted strings
)
"""Words that can never be a bare identifier: the grammar's own keywords, plus the
write and schema statement keywords so that ``DELETE``, ``PRAGMA`` or ``INTO`` anywhere
is a clean, named rejection rather than being taken for a column name. This list is
for clear errors, not safety -- a word missing from it still has to fit the grammar.
A column that really is called ``update`` can be written as a quoted identifier."""

# Functions written without parentheses for a value -- each dialect decides which it
# allows; outside the allow list they are rejected like any unknown name.
NILADIC = frozenset({"CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP"})

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>--|/\*)
  | (?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<string>'(?:[^']|'')*')
  | (?P<qident>"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]*\])
  | (?P<param>:[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op><=|>=|<>|!=|==|\|\||[=<>+\-*/%])
  | (?P<punct>[(),.;])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int

    def is_word(self, *words: str) -> bool:
        return self.kind == "ident" and self.value.upper() in words


def tokenize(text: str, *, allow_params: bool = False) -> list[Token]:
    if len(text) > MAX_SQL_LENGTH:
        raise SQLSyntaxError(f"the SQL is longer than {MAX_SQL_LENGTH} characters", code="too_long")
    tokens: list[Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise SQLSyntaxError(f"unexpected character {text[pos]!r} at position {pos}")
        kind = match.lastgroup or ""
        value = match.group()
        if kind == "comment":
            # Comments carry nothing a query needs and are the classic way to hide the
            # tail of a statement from a reviewer, so there is no stripping them.
            raise SQLSyntaxError("comments are not allowed in the SQL", code="comment")
        if kind == "param" and not allow_params:
            raise SQLSyntaxError(
                "bind parameters are not allowed; write the value as a literal",
                code="syntax",
            )
        if kind == "qident" and _CONTROL_RE.search(value):
            raise SQLSyntaxError(
                f"control character in a quoted identifier at position {pos}"
            )
        if kind == "number":
            # A number glued to a word ("1abc", "0x1F") is not a number.
            end = match.end()
            if end < len(text) and (text[end].isalnum() or text[end] == "_"):
                raise SQLSyntaxError(f"malformed number at position {pos}")
        if kind != "ws":
            tokens.append(Token(kind, value, pos))
        pos = match.end()

    # One trailing statement terminator is tolerated -- models add it by habit. Any
    # other ";" means a second statement.
    if tokens and tokens[-1].value == ";":
        tokens.pop()
    if any(token.value == ";" for token in tokens):
        raise SQLSyntaxError("only a single statement is allowed", code="multiple_statements")
    tokens.append(Token("eof", "", len(text)))
    return tokens


def _unquote(token: Token) -> str:
    value = token.value
    if value[0] == "[":
        return value[1:-1]
    quote = value[0]
    return value[1:-1].replace(quote * 2, quote)


# --- the tree -------------------------------------------------------------------------


@dataclass(frozen=True)
class Ident:
    name: str
    quoted: bool = False

    @property
    def key(self) -> str:
        # SQLite (and the SQL standard, for unquoted names) compare identifiers without
        # regard to ASCII case. Resolution follows the most permissive of the two so
        # that a case or quoting trick can never reach a different table than intended.
        return self.name.casefold()


class Expr:
    """Base class of expression nodes."""


@dataclass(frozen=True)
class Literal(Expr):
    value: Any
    kind: str  # "string" | "number" | "null" | "bool"
    text: str = ""


@dataclass(frozen=True)
class Param(Expr):
    """``:name`` -- only ever produced for a trusted row-filter predicate."""

    name: str


@dataclass(frozen=True)
class ColumnRef(Expr):
    column: Ident
    table: Ident | None = None


@dataclass(frozen=True)
class Star(Expr):
    table: Ident | None = None


@dataclass(frozen=True)
class Unary(Expr):
    op: str
    operand: Expr


@dataclass(frozen=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Is(Expr):
    left: Expr
    right: Expr
    negated: bool


@dataclass(frozen=True)
class Like(Expr):
    operand: Expr
    pattern: Expr
    negated: bool
    escape: Expr | None = None


@dataclass(frozen=True)
class Between(Expr):
    operand: Expr
    low: Expr
    high: Expr
    negated: bool


@dataclass(frozen=True)
class InList(Expr):
    operand: Expr
    items: tuple[Expr, ...]
    negated: bool


@dataclass(frozen=True)
class InQuery(Expr):
    operand: Expr
    query: Query
    negated: bool


@dataclass(frozen=True)
class Exists(Expr):
    query: Query


@dataclass(frozen=True)
class ScalarSubquery(Expr):
    query: Query


@dataclass(frozen=True)
class FuncCall(Expr):
    name: str  # upper case
    args: tuple[Expr, ...]
    distinct: bool = False
    star: bool = False
    niladic: bool = False


@dataclass(frozen=True)
class Case(Expr):
    operand: Expr | None
    whens: tuple[tuple[Expr, Expr], ...]
    default: Expr | None


@dataclass(frozen=True)
class Cast(Expr):
    operand: Expr
    type_name: str


@dataclass(frozen=True)
class ResultColumn:
    expr: Expr
    alias: Ident | None
    source_text: str


@dataclass(frozen=True)
class TableItem:
    name: Ident
    alias: Ident | None


@dataclass(frozen=True)
class SubqueryItem:
    query: Query
    alias: Ident | None


FromItem = TableItem | SubqueryItem


@dataclass(frozen=True)
class Join:
    kind: str  # "FIRST" | "COMMA" | "INNER" | "LEFT" | "CROSS"
    item: FromItem
    on: Expr | None = None


@dataclass(frozen=True)
class Select:
    columns: tuple[ResultColumn, ...]
    distinct: bool = False
    from_: tuple[Join, ...] = ()
    where: Expr | None = None
    group_by: tuple[Expr, ...] = ()
    having: Expr | None = None


@dataclass(frozen=True)
class OrderItem:
    expr: Expr
    descending: bool | None = None
    nulls: str | None = None  # "FIRST" | "LAST"


@dataclass(frozen=True)
class CTE:
    name: Ident
    query: Query


@dataclass(frozen=True)
class Query:
    cores: tuple[Select, ...]
    operators: tuple[str, ...] = ()  # between consecutive cores: "UNION", "UNION ALL", ...
    ctes: tuple[CTE, ...] = ()
    order_by: tuple[OrderItem, ...] = ()
    limit: int | None = None
    offset: int | None = None


# --- the parser -----------------------------------------------------------------------


@dataclass
class _Parser:
    text: str
    tokens: list[Token]
    allow_params: bool = False
    pos: int = 0
    depth: int = 0
    _counter: list[int] = field(default_factory=lambda: [0])

    # --- cursor ---

    @property
    def tok(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def advance(self) -> Token:
        token = self.tokens[self.pos]
        if token.kind != "eof":
            self.pos += 1
        return token

    def accept_word(self, *words: str) -> Token | None:
        if self.tok.is_word(*words):
            return self.advance()
        return None

    def expect_word(self, word: str) -> Token:
        token = self.accept_word(word)
        if token is None:
            self.fail(f"expected {word}")
        return token  # type: ignore[return-value]

    def accept(self, value: str) -> Token | None:
        if self.tok.kind in ("op", "punct") and self.tok.value == value:
            return self.advance()
        return None

    def expect(self, value: str) -> Token:
        token = self.accept(value)
        if token is None:
            self.fail(f"expected '{value}'")
        return token  # type: ignore[return-value]

    def fail(self, message: str) -> None:
        token = self.tok
        found = "end of input" if token.kind == "eof" else repr(token.value)
        raise SQLSyntaxError(f"{message} at position {token.pos}, found {found}")

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise SQLSyntaxError("the SQL is nested too deeply", code="too_deep")

    def leave(self) -> None:
        self.depth -= 1

    # --- identifiers ---

    def is_ident(self, token: Token | None = None) -> bool:
        token = token or self.tok
        if token.kind == "qident":
            return True
        return token.kind == "ident" and token.value.upper() not in RESERVED

    def ident(self, what: str = "a name") -> Ident:
        token = self.tok
        if token.kind == "qident":
            self.advance()
            return Ident(_unquote(token), quoted=True)
        if token.kind == "ident":
            if token.value.upper() in RESERVED:
                self.reserved(token)
            self.advance()
            return Ident(token.value)
        self.fail(f"expected {what}")
        raise AssertionError  # unreachable

    def reserved(self, token: Token) -> None:
        word = token.value.upper()
        if word in _STATEMENT_WORDS:
            raise SQLSyntaxError(
                f"{word} is not allowed: only a single read-only SELECT statement is accepted",
                code="not_select",
            )
        raise SQLSyntaxError(
            f"{word} is not supported here (position {token.pos})", code="unsupported"
        )

    # --- statements ---

    def statement(self) -> Query:
        if not self.tok.is_word("SELECT", "WITH"):
            if self.tok.kind == "ident":
                raise SQLSyntaxError(
                    f"{self.tok.value.upper()} statements are not allowed: only a single "
                    f"read-only SELECT statement is accepted",
                    code="not_select",
                )
            self.fail("expected SELECT")
        query = self.query()
        if self.tok.kind != "eof":
            if self.tok.kind == "ident" and self.tok.value.upper() in RESERVED:
                self.reserved(self.tok)
            self.fail("unexpected text after the end of the query")
        return query

    def query(self) -> Query:
        self.enter()
        ctes: list[CTE] = []
        if self.accept_word("WITH"):
            if self.tok.is_word("RECURSIVE"):
                raise SQLSyntaxError("WITH RECURSIVE is not supported", code="unsupported")
            while True:
                name = self.ident("a CTE name")
                if self.tok.value == "(":
                    raise SQLSyntaxError(
                        "a column list on a CTE is not supported; alias the columns "
                        "inside its SELECT instead",
                        code="unsupported",
                    )
                self.expect_word("AS")
                if self.tok.is_word("MATERIALIZED", "NOT"):
                    raise SQLSyntaxError("CTE materialization hints are not supported",
                                         code="unsupported")
                self.expect("(")
                ctes.append(CTE(name, self.query()))
                self.expect(")")
                if not self.accept(","):
                    break

        cores = [self.select_core()]
        operators: list[str] = []
        while self.tok.is_word("UNION", "INTERSECT", "EXCEPT"):
            op = self.advance().value.upper()
            if op == "UNION" and self.accept_word("ALL"):
                op = "UNION ALL"
            operators.append(op)
            cores.append(self.select_core())

        order_by: list[OrderItem] = []
        if self.accept_word("ORDER"):
            self.expect_word("BY")
            order_by.append(self.order_item())
            while self.accept(","):
                order_by.append(self.order_item())

        limit = offset = None
        if self.accept_word("LIMIT"):
            limit = self.non_negative_int("LIMIT")
            if self.tok.value == ",":
                raise SQLSyntaxError("write LIMIT n OFFSET m, not LIMIT m, n", code="unsupported")
            if self.accept_word("OFFSET"):
                offset = self.non_negative_int("OFFSET")
        self.leave()
        return Query(tuple(cores), tuple(operators), tuple(ctes), tuple(order_by), limit, offset)

    def non_negative_int(self, clause: str) -> int:
        token = self.tok
        if token.kind != "number" or not token.value.isdigit():
            raise SQLSyntaxError(
                f"{clause} takes a non-negative whole number literal", code="limit"
            )
        self.advance()
        if self.tok.kind == "op":
            raise SQLSyntaxError(
                f"{clause} takes a non-negative whole number literal", code="limit"
            )
        # Length first: int() of a few thousand digits raises its own ValueError.
        if len(token.value.lstrip("0")) > 19 or int(token.value) > MAX_INT_LITERAL:
            raise SQLSyntaxError(f"{clause} is larger than {MAX_INT_LITERAL}", code="limit")
        return int(token.value)

    def order_item(self) -> OrderItem:
        expr = self.expr()
        descending = None
        if self.accept_word("ASC"):
            descending = False
        elif self.accept_word("DESC"):
            descending = True
        nulls = None
        if self.accept_word("NULLS"):
            token = self.accept_word("FIRST", "LAST")
            if token is None:
                self.fail("expected FIRST or LAST")
            nulls = token.value.upper()  # type: ignore[union-attr]
        return OrderItem(expr, descending, nulls)

    def select_core(self) -> Select:
        if self.tok.value == "(":
            raise SQLSyntaxError(
                "a parenthesized SELECT is not supported as a compound member",
                code="unsupported",
            )
        if self.tok.is_word("VALUES"):
            self.reserved(self.tok)
        self.expect_word("SELECT")
        distinct = bool(self.accept_word("DISTINCT"))
        if not distinct:
            self.accept_word("ALL")
        columns = [self.result_column()]
        while self.accept(","):
            columns.append(self.result_column())

        from_: list[Join] = []
        if self.accept_word("FROM"):
            from_.append(Join("FIRST", self.from_item()))
            while True:
                kind = self.join_kind()
                if kind is None:
                    break
                item = self.from_item()
                on = None
                if self.accept_word("ON"):
                    on = self.expr()
                elif self.tok.is_word("USING"):
                    raise SQLSyntaxError("JOIN ... USING is not supported; use ON",
                                         code="unsupported")
                if kind in ("INNER", "LEFT") and on is None:
                    self.fail(f"{kind} JOIN needs an ON condition")
                if kind in ("CROSS", "COMMA") and on is not None:
                    self.fail("ON is not allowed after a comma or CROSS JOIN")
                from_.append(Join(kind, item, on))

        where = self.expr() if self.accept_word("WHERE") else None
        group_by: list[Expr] = []
        if self.accept_word("GROUP"):
            self.expect_word("BY")
            group_by.append(self.expr())
            while self.accept(","):
                group_by.append(self.expr())
        having = self.expr() if self.accept_word("HAVING") else None
        return Select(tuple(columns), distinct, tuple(from_), where, tuple(group_by), having)

    def join_kind(self) -> str | None:
        if self.accept(","):
            return "COMMA"
        if self.tok.is_word("NATURAL", "RIGHT", "FULL"):
            raise SQLSyntaxError(f"{self.tok.value.upper()} joins are not supported",
                                 code="unsupported")
        if self.accept_word("JOIN"):
            return "INNER"
        if self.accept_word("INNER"):
            self.expect_word("JOIN")
            return "INNER"
        if self.accept_word("LEFT"):
            self.accept_word("OUTER")
            self.expect_word("JOIN")
            return "LEFT"
        if self.accept_word("CROSS"):
            self.expect_word("JOIN")
            return "CROSS"
        return None

    def from_item(self) -> FromItem:
        if self.accept("("):
            if not self.tok.is_word("SELECT", "WITH"):
                self.fail("expected a subquery")
            query = self.query()
            self.expect(")")
            return SubqueryItem(query, self.alias())
        name = self.ident("a table name")
        if self.tok.value == ".":
            raise SQLSyntaxError(
                f"unknown table '{name.name}.{self.peek().value}'", code="unknown_table"
            )
        if self.tok.value == "(":
            raise SQLSyntaxError("table-valued functions are not supported", code="unsupported")
        return TableItem(name, self.alias())

    def alias(self) -> Ident | None:
        if self.accept_word("AS"):
            return self.ident("an alias")
        if self.is_ident():
            return self.ident()
        return None

    def result_column(self) -> ResultColumn:
        start = self.tok.pos
        if self.accept("*"):
            return ResultColumn(Star(), None, "*")
        if self.is_ident() and self.peek().value == "." and self.peek(2).value == "*":
            table = self.ident()
            self.advance()
            self.advance()
            return ResultColumn(Star(table), None, f"{table.name}.*")
        expr = self.expr()
        end_token = self.tokens[self.pos - 1]
        source = self.text[start : end_token.pos + len(end_token.value)]
        alias = None
        if self.accept_word("AS"):
            alias = self.ident("an alias")
        elif self.is_ident():
            alias = self.ident()
        return ResultColumn(expr, alias, source)

    # --- expressions, lowest precedence first ---
    #
    # The validator renders every operator fully parenthesized, so the database
    # evaluates exactly the tree built here whatever its own precedence rules are.

    def expr(self) -> Expr:
        self.enter()
        result = self.or_expr()
        self.leave()
        return result

    def or_expr(self) -> Expr:
        left = self.and_expr()
        while self.accept_word("OR"):
            left = Binary("OR", left, self.and_expr())
        return left

    def and_expr(self) -> Expr:
        left = self.not_expr()
        while self.accept_word("AND"):
            left = Binary("AND", left, self.not_expr())
        return left

    def not_expr(self) -> Expr:
        if self.accept_word("NOT"):
            self.enter()
            operand = self.not_expr()
            self.leave()
            return Unary("NOT", operand)
        return self.equality()

    def equality(self) -> Expr:
        left = self.comparison()
        while True:
            token = self.tok
            if token.kind == "op" and token.value in ("=", "==", "!=", "<>"):
                self.advance()
                op = "=" if token.value in ("=", "==") else "<>"
                left = Binary(op, left, self.comparison())
                continue
            if token.is_word("IS"):
                self.advance()
                negated = bool(self.accept_word("NOT"))
                if self.tok.is_word("DISTINCT"):
                    raise SQLSyntaxError("IS DISTINCT FROM is not supported", code="unsupported")
                left = Is(left, self.comparison(), negated)
                continue
            negated = False
            if token.is_word("NOT") and self.peek().is_word("IN", "LIKE", "BETWEEN"):
                self.advance()
                negated = True
                token = self.tok
            if token.is_word("IN"):
                self.advance()
                left = self.in_tail(left, negated)
                continue
            if token.is_word("LIKE"):
                self.advance()
                pattern = self.comparison()
                escape = self.comparison() if self.accept_word("ESCAPE") else None
                left = Like(left, pattern, negated, escape)
                continue
            if token.is_word("BETWEEN"):
                self.advance()
                low = self.comparison()
                self.expect_word("AND")
                left = Between(left, low, self.comparison(), negated)
                continue
            return left

    def in_tail(self, operand: Expr, negated: bool) -> Expr:
        self.expect("(")
        if self.tok.is_word("SELECT", "WITH"):
            query = self.query()
            self.expect(")")
            return InQuery(operand, query, negated)
        if self.tok.value == ")":
            self.fail("IN needs at least one value")
        items = [self.expr()]
        while self.accept(","):
            items.append(self.expr())
        self.expect(")")
        return InList(operand, tuple(items), negated)

    def comparison(self) -> Expr:
        left = self.additive()
        while self.tok.kind == "op" and self.tok.value in ("<", "<=", ">", ">="):
            op = self.advance().value
            left = Binary(op, left, self.additive())
        return left

    def additive(self) -> Expr:
        left = self.multiplicative()
        while self.tok.kind == "op" and self.tok.value in ("+", "-"):
            op = self.advance().value
            left = Binary(op, left, self.multiplicative())
        return left

    def multiplicative(self) -> Expr:
        left = self.concat()
        while self.tok.kind == "op" and self.tok.value in ("*", "/", "%"):
            op = self.advance().value
            left = Binary(op, left, self.concat())
        return left

    def concat(self) -> Expr:
        left = self.unary()
        while self.tok.kind == "op" and self.tok.value == "||":
            self.advance()
            left = Binary("||", left, self.unary())
        return left

    def unary(self) -> Expr:
        if self.tok.kind == "op" and self.tok.value in ("-", "+"):
            op = self.advance().value
            self.enter()
            operand = self.unary()
            self.leave()
            return Unary(op, operand)
        return self.primary()

    def primary(self) -> Expr:
        token = self.tok
        if token.kind == "number":
            self.advance()
            text = token.value
            if len(text) > 100:
                # int() refuses more than 4300 digits with a bare ValueError; no real
                # query needs a hundred-digit literal.
                raise SQLSyntaxError("number literal is too long")
            is_int = text.isdigit()
            return Literal(int(text) if is_int else float(text), "number", text)
        if token.kind == "string":
            self.advance()
            return Literal(token.value[1:-1].replace("''", "'"), "string", token.value)
        if token.kind == "param":
            self.advance()
            return Param(token.value[1:])
        if token.value == "(":
            self.advance()
            if self.tok.is_word("SELECT", "WITH"):
                query = self.query()
                self.expect(")")
                return ScalarSubquery(query)
            inner = self.expr()
            if self.tok.value == ",":
                raise SQLSyntaxError("row values are not supported", code="unsupported")
            self.expect(")")
            return inner
        if token.kind == "ident":
            word = token.value.upper()
            if word == "NULL":
                self.advance()
                return Literal(None, "null", "NULL")
            if word in ("TRUE", "FALSE"):
                self.advance()
                return Literal(word == "TRUE", "bool", word)
            if word == "EXISTS":
                self.advance()
                self.expect("(")
                if not self.tok.is_word("SELECT", "WITH"):
                    self.fail("expected a subquery")
                query = self.query()
                self.expect(")")
                return Exists(query)
            if word == "CASE":
                return self.case()
            if word == "CAST":
                return self.cast()
            if word in NILADIC and self.peek().value != "(":
                self.advance()
                return FuncCall(word, (), niladic=True)
            if self.peek().value == "(" and word not in RESERVED:
                return self.call()
        if self.is_ident():
            first = self.ident()
            if self.accept("."):
                second = self.ident("a column name")
                if self.tok.value == ".":
                    raise SQLSyntaxError(
                        f"unknown column '{first.name}.{second.name}.{self.peek().value}'",
                        code="unknown_column",
                    )
                return ColumnRef(second, first)
            return ColumnRef(first)
        if token.kind == "ident" and token.value.upper() in RESERVED:
            self.reserved(token)
        if token.kind == "op" and token.value == "*":
            self.fail("'*' is only allowed as a whole result column or in COUNT(*)")
        self.fail("expected an expression")
        raise AssertionError  # unreachable

    def call(self) -> Expr:
        name = self.advance().value.upper()
        self.expect("(")
        star = bool(self.accept("*"))
        distinct = False
        args: list[Expr] = []
        if not star:
            distinct = bool(self.accept_word("DISTINCT"))
            if self.tok.value != ")":
                args.append(self.expr())
                while self.accept(","):
                    args.append(self.expr())
        self.expect(")")
        if self.tok.is_word("OVER", "FILTER"):
            raise SQLSyntaxError("window functions and FILTER clauses are not supported",
                                 code="unsupported")
        return FuncCall(name, tuple(args), distinct=distinct, star=star)

    def case(self) -> Expr:
        self.expect_word("CASE")
        operand = None if self.tok.is_word("WHEN") else self.expr()
        whens: list[tuple[Expr, Expr]] = []
        while self.accept_word("WHEN"):
            condition = self.expr()
            self.expect_word("THEN")
            whens.append((condition, self.expr()))
        if not whens:
            self.fail("CASE needs at least one WHEN")
        default = self.expr() if self.accept_word("ELSE") else None
        self.expect_word("END")
        return Case(operand, tuple(whens), default)

    def cast(self) -> Expr:
        self.expect_word("CAST")
        self.expect("(")
        operand = self.expr()
        self.expect_word("AS")
        words: list[str] = []
        while self.tok.kind == "ident" and self.tok.value.upper() not in RESERVED:
            words.append(self.advance().value.upper())
        if not words:
            self.fail("expected a type name")
        type_name = " ".join(words)
        if self.accept("("):
            sizes = [self.non_negative_int("a type size")]
            while self.accept(","):
                sizes.append(self.non_negative_int("a type size"))
            self.expect(")")
            type_name += "(" + ", ".join(str(size) for size in sizes) + ")"
        self.expect(")")
        return Cast(operand, type_name)


_STATEMENT_WORDS = frozenset(
    "INSERT UPDATE DELETE CREATE DROP ALTER ATTACH DETACH PRAGMA VACUUM INTO VALUES".split()  # noqa: SIM905
)


def parse_query(sql: str) -> Query:
    """Parse one read-only SELECT statement, or raise :class:`SQLSyntaxError`."""

    if not isinstance(sql, str) or not sql.strip():
        raise SQLSyntaxError("the SQL is empty")
    parser = _Parser(sql, tokenize(sql))
    return parser.statement()


def parse_predicate(sql: str) -> Expr:
    """Parse a trusted row-filter predicate. ``:name`` parameters are allowed here, and
    only here."""

    parser = _Parser(sql, tokenize(sql, allow_params=True), allow_params=True)
    expr = parser.expr()
    if parser.tok.kind != "eof":
        parser.fail("unexpected text after the predicate")
    return expr
