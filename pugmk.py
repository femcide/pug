#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


KEYWORDS = {
    "пусть",
    "если",
    "иначе",
    "пока",
    "функция",
    "вернуть",
    "истина",
    "ложь",
    "и",
    "или",
    "не",
    "ничего",
}

TOKEN_SPEC = [
    ("NUMBER", r"\d+(?:\.\d+)?"),
    ("STRING", r'"(?:[^"\\]|\\.)*"'),
    ("OP", r"==|!=|<=|>=|[+\-*/%=<>(){},]"),
    ("IDENT", r"[A-Za-zА-Яа-я_][A-Za-zА-Яа-я0-9_]*"),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t\r]+"),
    ("COMMENT", r"#.*"),
]
TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC))


@dataclass
class Token:
    typ: str
    val: str
    pos: int


class ParseError(Exception):
    pass


class RuntimeErrorPug(Exception):
    pass


@dataclass
class Number:
    value: float | int


@dataclass
class String:
    value: str


@dataclass
class Bool:
    value: bool


@dataclass
class Null:
    pass


@dataclass
class Variable:
    name: str


@dataclass
class Binary:
    left: Any
    op: str
    right: Any


@dataclass
class Unary:
    op: str
    expr: Any


@dataclass
class Call:
    fn: Any
    args: list[Any]


@dataclass
class Assign:
    name: str
    expr: Any


@dataclass
class Let:
    name: str
    expr: Any


@dataclass
class ExprStmt:
    expr: Any


@dataclass
class If:
    cond: Any
    then_block: list[Any]
    else_block: list[Any] | None


@dataclass
class While:
    cond: Any
    block: list[Any]


@dataclass
class FunctionDef:
    name: str
    params: list[str]
    block: list[Any]


@dataclass
class Return:
    expr: Any | None


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> Token:
        return self.tokens[self.i]

    def prev(self) -> Token:
        return self.tokens[self.i - 1]

    def at_end(self) -> bool:
        return self.peek().typ == "EOF"

    def match(self, *values: str) -> bool:
        if self.peek().val in values:
            self.i += 1
            return True
        return False

    def match_type(self, *types: str) -> bool:
        if self.peek().typ in types:
            self.i += 1
            return True
        return False

    def consume(self, value: str, msg: str):
        if self.peek().val == value:
            self.i += 1
            return
        raise ParseError(msg + f" возле '{self.peek().val}'")

    def consume_type(self, typ: str, msg: str) -> Token:
        if self.peek().typ == typ:
            self.i += 1
            return self.prev()
        raise ParseError(msg + f" возле '{self.peek().val}'")

    def skip_newlines(self):
        while self.match_type("NEWLINE"):
            pass

    def parse(self) -> list[Any]:
        program: list[Any] = []
        self.skip_newlines()
        while not self.at_end():
            program.append(self.declaration())
            self.skip_newlines()
        return program

    def declaration(self):
        if self.match("функция"):
            return self.function_def()
        if self.match("пусть"):
            name = self.consume_type("IDENT", "Ожидалось имя переменной").val
            self.consume("=", "Ожидался '='")
            return Let(name, self.expression())
        return self.statement()

    def function_def(self):
        name = self.consume_type("IDENT", "Ожидалось имя функции").val
        self.consume("(", "Ожидалась '('")
        params: list[str] = []
        if self.peek().val != ")":
            while True:
                params.append(self.consume_type("IDENT", "Ожидалось имя параметра").val)
                if not self.match(","):
                    break
        self.consume(")", "Ожидалась ')'")
        block = self.block()
        return FunctionDef(name, params, block)

    def block(self) -> list[Any]:
        self.consume("{", "Ожидалась '{'")
        self.skip_newlines()
        items: list[Any] = []
        while self.peek().val != "}" and not self.at_end():
            items.append(self.declaration())
            self.skip_newlines()
        self.consume("}", "Ожидалась '}'")
        return items

    def statement(self):
        if self.match("если"):
            cond = self.expression()
            then_block = self.block()
            else_block = None
            if self.match("иначе"):
                else_block = self.block()
            return If(cond, then_block, else_block)
        if self.match("пока"):
            cond = self.expression()
            return While(cond, self.block())
        if self.match("вернуть"):
            if self.peek().typ in ("NEWLINE", "EOF") or self.peek().val == "}":
                return Return(None)
            return Return(self.expression())

        if self.peek().typ == "IDENT" and self.tokens[self.i + 1].val == "=":
            name = self.consume_type("IDENT", "Ожидалось имя").val
            self.consume("=", "Ожидался '='")
            return Assign(name, self.expression())
        return ExprStmt(self.expression())

    def expression(self):
        return self.logic_or()

    def logic_or(self):
        expr = self.logic_and()
        while self.match("или"):
            expr = Binary(expr, "или", self.logic_and())
        return expr

    def logic_and(self):
        expr = self.equality()
        while self.match("и"):
            expr = Binary(expr, "и", self.equality())
        return expr

    def equality(self):
        expr = self.comparison()
        while self.match("==", "!="):
            op = self.prev().val
            expr = Binary(expr, op, self.comparison())
        return expr

    def comparison(self):
        expr = self.term()
        while self.match("<", "<=", ">", ">="):
            op = self.prev().val
            expr = Binary(expr, op, self.term())
        return expr

    def term(self):
        expr = self.factor()
        while self.match("+", "-"):
            op = self.prev().val
            expr = Binary(expr, op, self.factor())
        return expr

    def factor(self):
        expr = self.unary()
        while self.match("*", "/", "%"):
            op = self.prev().val
            expr = Binary(expr, op, self.unary())
        return expr

    def unary(self):
        if self.match("-", "не"):
            return Unary(self.prev().val, self.unary())
        return self.call()

    def call(self):
        expr = self.primary()
        while self.match("("):
            args: list[Any] = []
            if self.peek().val != ")":
                while True:
                    args.append(self.expression())
                    if not self.match(","):
                        break
            self.consume(")", "Ожидалась ')' после аргументов")
            expr = Call(expr, args)
        return expr

    def primary(self):
        if self.match_type("NUMBER"):
            txt = self.prev().val
            return Number(float(txt) if "." in txt else int(txt))
        if self.match_type("STRING"):
            return String(ast.literal_eval(self.prev().val))
        if self.match("истина"):
            return Bool(True)
        if self.match("ложь"):
            return Bool(False)
        if self.match("ничего"):
            return Null()
        if self.match_type("IDENT"):
            return Variable(self.prev().val)
        if self.match("("):
            expr = self.expression()
            self.consume(")", "Ожидалась ')' ")
            return expr
        raise ParseError(f"Ожидалось выражение возле '{self.peek().val}'")


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(source):
        m = TOKEN_RE.match(source, pos)
        if not m:
            raise ParseError(f"Неизвестный символ: '{source[pos]}'")
        typ = m.lastgroup
        val = m.group()
        if typ == "IDENT" and val in KEYWORDS:
            tokens.append(Token("KEYWORD", val, pos))
        elif typ not in {"SKIP", "COMMENT"}:
            tokens.append(Token(typ, val, pos))
        pos = m.end()
    tokens.append(Token("EOF", "", pos))
    return tokens


class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


class Env:
    def __init__(self, parent: Env | None = None):
        self.parent = parent
        self.values: dict[str, Any] = {}

    def define(self, name: str, value: Any):
        self.values[name] = value

    def set(self, name: str, value: Any):
        if name in self.values:
            self.values[name] = value
            return
        if self.parent:
            self.parent.set(name, value)
            return
        raise RuntimeErrorPug(f"Переменная '{name}' не объявлена")

    def get(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        if self.parent:
            return self.parent.get(name)
        raise RuntimeErrorPug(f"Неизвестное имя: {name}")


class PugFunction:
    def __init__(self, node: FunctionDef, closure: Env):
        self.node = node
        self.closure = closure

    def __call__(self, interpreter: "Interpreter", args: list[Any]) -> Any:
        if len(args) != len(self.node.params):
            raise RuntimeErrorPug(
                f"Функция '{self.node.name}' ожидала {len(self.node.params)} арг., получено {len(args)}"
            )
        env = Env(self.closure)
        for name, value in zip(self.node.params, args):
            env.define(name, value)
        try:
            interpreter.exec_block(self.node.block, env)
        except ReturnSignal as r:
            return r.value
        return None


class Interpreter:
    def __init__(self):
        self.globals = Env()
        self.env = self.globals
        self.globals.define("печать", lambda *vals: print(*vals))
        self.globals.define("ввод", lambda prompt="": input(str(prompt)))
        self.globals.define("длина", lambda x: len(x))
        self.globals.define("сохранить_файл", self.save_file)

    def save_file(self, path: Any, contents: Any):
        target = Path(str(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(contents), encoding="utf-8")
        return None

    def run(self, program: Iterable[Any]):
        for stmt in program:
            self.execute(stmt)

    def execute(self, stmt: Any):
        match stmt:
            case Let(name, expr):
                self.env.define(name, self.evaluate(expr))
            case Assign(name, expr):
                self.env.set(name, self.evaluate(expr))
            case ExprStmt(expr):
                self.evaluate(expr)
            case If(cond, then_block, else_block):
                if self.is_truthy(self.evaluate(cond)):
                    self.exec_block(then_block, Env(self.env))
                elif else_block is not None:
                    self.exec_block(else_block, Env(self.env))
            case While(cond, block):
                while self.is_truthy(self.evaluate(cond)):
                    self.exec_block(block, Env(self.env))
            case FunctionDef():
                self.env.define(stmt.name, PugFunction(stmt, self.env))
            case Return(expr):
                value = None if expr is None else self.evaluate(expr)
                raise ReturnSignal(value)
            case _:
                raise RuntimeErrorPug(f"Неизвестный оператор: {stmt}")

    def exec_block(self, block: list[Any], env: Env):
        prev = self.env
        self.env = env
        try:
            for item in block:
                self.execute(item)
        finally:
            self.env = prev

    def evaluate(self, expr: Any) -> Any:
        match expr:
            case Number(value=value):
                return value
            case String(value=value):
                return value
            case Bool(value=value):
                return value
            case Null():
                return None
            case Variable(name=name):
                return self.env.get(name)
            case Unary(op=op, expr=inner):
                v = self.evaluate(inner)
                if op == "-":
                    return -v
                if op == "не":
                    return not self.is_truthy(v)
                raise RuntimeErrorPug(f"Неизвестный унарный оператор {op}")
            case Binary(left=left, op=op, right=right):
                l, r = self.evaluate(left), self.evaluate(right)
                return self.eval_binary(l, op, r)
            case Call(fn=fn, args=args):
                callee = self.evaluate(fn)
                values = [self.evaluate(arg) for arg in args]
                if isinstance(callee, PugFunction):
                    return callee(self, values)
                if callable(callee):
                    return callee(*values)
                raise RuntimeErrorPug("Объект не является функцией")
            case _:
                raise RuntimeErrorPug(f"Неизвестное выражение: {expr}")

    @staticmethod
    def is_truthy(value: Any) -> bool:
        return bool(value)

    @staticmethod
    def eval_binary(l: Any, op: str, r: Any) -> Any:
        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "%": lambda a, b: a % b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "и": lambda a, b: bool(a) and bool(b),
            "или": lambda a, b: bool(a) or bool(b),
        }
        if op not in ops:
            raise RuntimeErrorPug(f"Неизвестный бинарный оператор {op}")
        return ops[op](l, r)


def run_source(source: str):
    tokens = tokenize(source)
    parser = Parser(tokens)
    program = parser.parse()
    interpreter = Interpreter()
    interpreter.run(program)


def main():
    argp = argparse.ArgumentParser(description="PUGMK — русский язык программирования (.pugs)")
    argp.add_argument("file", nargs="?", help="Путь к .pugs файлу")
    args = argp.parse_args()

    if not args.file:
        print("PUGMK REPL. Введите код, Ctrl+D для выхода.")
        buffer = []
        for line in sys.stdin:
            buffer.append(line)
        if buffer:
            run_source("".join(buffer))
        return

    path = Path(args.file)
    if path.suffix != ".pugs":
        print("Подсказка: расширение файла должно быть .pugs", file=sys.stderr)
    run_source(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    try:
        main()
    except (ParseError, RuntimeErrorPug) as e:
        print(f"Ошибка PUGMK: {e}", file=sys.stderr)
        raise SystemExit(1)
