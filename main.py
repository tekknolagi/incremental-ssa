from __future__ import annotations
import collections
import dataclasses
import io

EMPTY = object()

@dataclasses.dataclass
class Peekable:
    iterator: object
    item: object = EMPTY

    def __iter__(self):
        return self

    def peek(self):
        if self.item is EMPTY:
            self.item = next(self.iterator)
        return self.item

    def __next__(self):
        if self.item is not EMPTY:
            result = self.item
            self.item = EMPTY
            return result
        return next(self.iterator)

@dataclasses.dataclass
class Token:
    pass

@dataclasses.dataclass
class TIdent(Token):
    value: str

@dataclasses.dataclass
class TInt(Token):
    value: int

@dataclasses.dataclass
class TBreak(Token):
    pass

@dataclasses.dataclass
class TElse(Token):
    pass

@dataclasses.dataclass
class TFalse(Token):
    pass

@dataclasses.dataclass
class TFunc(Token):
    pass

@dataclasses.dataclass
class TIf(Token):
    pass

@dataclasses.dataclass
class TPrint(Token):
    pass

@dataclasses.dataclass
class TReturn(Token):
    pass

@dataclasses.dataclass
class TTrue(Token):
    pass

@dataclasses.dataclass
class TWhile(Token):
    pass

@dataclasses.dataclass
class TVar(Token):
    pass

@dataclasses.dataclass
class TTypeInt(Token):
    pass

@dataclasses.dataclass
class TTypeFloat(Token):
    pass

@dataclasses.dataclass
class TChar(Token):
    pass

@dataclasses.dataclass
class TTypeBool(Token):
    pass

@dataclasses.dataclass
class TPunct(Token):
    value: str

KEYWORDS = {
    "break": TBreak(),
    "else": TElse(),
    "false": TFalse(),
    "func": TFunc(),
    "if": TIf(),
    "print": TPrint(),
    "return": TReturn(),
    "true": TTrue(),
    "while": TWhile(),
    "var": TVar(),
    "int": TTypeInt(),
    "float": TTypeFloat(),
    "char": TChar(),
    "bool": TTypeBool(),
}

PUNCTUATION = {
    "+",
    "-",
    "*",
    "/",
    "(",
    ")",
    "{",
    "}",
    ";",
    ">",
    ">=",
    "<",
    "<=",
    "=",
    "==",
    "!",
    "!=",
}

@dataclasses.dataclass
class PeekableString:
    iterator: object

    def __init__(self, source: str) -> None:
        self.iterator = Peekable(iter(source))

    def read(self, n: int) -> str:
        result = ""
        while n:
            try:
                result += next(self.iterator)
            except StopIteration:
                break
            n -= 1
        return result

    def peek(self) -> str:
        return self.iterator.peek()


@dataclasses.dataclass
class Lexer:
    source: io.BufferedReader

    def __iter__(self):
        return self

    def __next__(self):
        result = self.read_token()
        if result is None:
            raise StopIteration
        return result

    def peek(self) -> str:
        return self.source.peek()[0]

    def read_token(self) -> Token|None:
        while True:
            c = self.source.read(1)
            if c.isspace():
                continue
            if not c:
                return None
            if c.isalpha():
                return self.read_ident(c)
            if c.isnumeric():
                return self.read_number(c)
            if c == '=' and self.peek() == '=':
                c += self.source.read(1)
            elif c == '<' and self.peek() == '=':
                c += self.source.read(1)
            elif c == '>' and self.peek() == '=':
                c += self.source.read(1)
            elif c == '!' and self.peek() == '=':
                c += self.source.read(1)
            elif c == '/' and self.peek() == '/':
                self.source.read(1)
                # Eat line comment
                while (c := self.source.read(1)) != '\n':
                    continue
                continue
            if c in PUNCTUATION:
                return TPunct(c)
            raise NotImplementedError(f"Unexpected char `{c}'")

    def read_ident(self, text: str) -> Token:
        while (c := self.peek()).isalpha():
            text += c
            self.source.read(1)
        return KEYWORDS.get(text) or TIdent(text)

    def read_number(self, text: str) -> Token:
        while (c := self.peek()).isnumeric():
            text += c
            self.source.read(1)
        if c == '.':
            raise NotImplementedError("Sorry, float not (yet) supported")
        return TInt(int(text))

ir = dataclasses.dataclass

@ir
class Instr:
    pass

@ir
class Int(Instr):
    value: int

    def __hash__(self) -> int:
        return id(self)

@ir
class Undefined(Instr):
    pass

    def __hash__(self) -> int:
        return id(self)

@ir
class HasOperands(Instr):
    operands: tuple[Instr, ...] = dataclasses.field(init=False, default=())

    def __hash__(self) -> int:
        return id(self)

@ir
class Add(HasOperands):
    def __init__(self, left: Instr, right: Instr) -> None:
        self.operands = (left, right)

    def __hash__(self) -> int:
        return id(self)

@ir
class Less(HasOperands):
    def __init__(self, left: Instr, right: Instr) -> None:
        self.operands = (left, right)

    def __hash__(self) -> int:
        return id(self)

@ir
class Phi(HasOperands):
    block: Block

    def append_operand(self, instr: Instr) -> None:
        self.operands = self.operands + (instr,)

    def __hash__(self) -> int:
        return id(self)

@ir
class Print(HasOperands):
    def __init__(self, value: Instr) -> None:
        self.operands = (value,)

    def __hash__(self) -> int:
        return id(self)

@ir
class Terminator(Instr):
    def __hash__(self) -> int:
        return id(self)

@ir
class Return(HasOperands, Terminator):
    def __init__(self, value: Instr) -> None:
        self.operands = (value,)

    def __hash__(self) -> int:
        return id(self)

@ir
class Branch(Terminator):
    target: Block

    def __hash__(self) -> int:
        return id(self)

@ir
class CondBranch(HasOperands, Terminator):
    iftrue: Block
    iffalse: Block

    def __init__(self, cond: Instr, iftrue: Block, iffalse: Block) -> None:
        self.operands = (cond,)
        self.iftrue = iftrue
        self.iffalse = iffalse

    def __hash__(self) -> int:
        return id(self)

@dataclasses.dataclass
class Block:
    id: int
    instrs: list[Instr] = dataclasses.field(init=False, default_factory=list)

    def emit(self, instr: Instr) -> Instr:
        self.instrs.append(instr)
        return instr

    def __hash__(self) -> int:
        return object.__hash__(self)

    def __eq__(self, other) -> bool:
        return self is other

    @property
    def filled(self) -> bool:
        return self.instrs and isinstance(self.instrs[-1], Terminator)

    def name(self) -> str:
        return f"bb{self.id}"

    def has_terminator(self) -> bool:
        return self.instrs and isinstance(self.instrs[-1], (Return, Branch, CondBranch))

@dataclasses.dataclass
class Function:
    name: str
    entry: Block
    blocks: list[Block]

    def __init__(self, name: str) -> None:
        self.name = name
        self.entry = Block(0)
        self.blocks = [self.entry]

    def new_block(self) -> Block:
        result = Block(len(self.blocks))
        self.blocks.append(result)
        return result

    def rpo(self) -> list[Block]:
        result = []
        self.po_from(self.entry, result, set())
        result.reverse()
        return result

    def po_from(self, block: Block, result: list[Block], visited: set[Block]):
        if block in visited:
            return
        visited.add(block)
        if not block.instrs:
            # TODO(max): Figure out what to do with empty blocks?
            result.append(block)
            return
        terminator = block.instrs[-1]
        if isinstance(terminator, Return):
            pass
        elif isinstance(terminator, Branch):
            self.po_from(terminator.target, result, visited)
        elif isinstance(terminator, CondBranch):
            self.po_from(terminator.iftrue, result, visited)
            self.po_from(terminator.iffalse, result, visited)
        else:
            raise RuntimeError(f"Unexpected terminator {terminator}")
        result.append(block)

@dataclasses.dataclass
class Program:
    functions: list[Function] = dataclasses.field(init=False, default_factory=list)

    def new_function(self, name: str) -> Function:
        result = Function(name)
        self.functions.append(result)
        return result

class ParseError(Exception):
    pass

PREC = {op: prec for prec, ops in enumerate([
    ["<"],
    ["+", "-"],
    ["*", "/"],
]) for op in ops}

OPS = set(PREC.keys())

ASSOC = {
    "<": "left",
    "+": "any",
    "-": "left",
    "*": "any",
    "/": "left",
}

@dataclasses.dataclass
class Parser:
    source: Peekable
    program: Program
    func: Function
    block: Block
    current_def: dict[str, dict[Block, Instr]]
    sealed_blocks: set[Block]
    incomplete_phis: dict[Block, dict[str, Instr]]
    preds: dict[Block, set[Block]]

    def __init__(self, lexer: Lexer) -> None:
        self.source = Peekable(lexer)
        self.program = Program()
        self.func = self.program.new_function("<toplevel>")
        self.block = self.func.entry
        self.current_def = collections.defaultdict(dict)
        self.sealed_blocks = set()
        self.incomplete_phis = collections.defaultdict(dict)
        self.preds = collections.defaultdict(set)

    def write_variable(self, variable: str, block: Block, value: Instr):
        self.current_def[variable][block] = value

    def read_variable(self, variable: str, block: Block) -> Instr:
        if block in self.current_def[variable]:
            # local value numbering
            return self.current_def[variable][block]
        return self.read_variable_recursive(variable, block)

    def read_variable_recursive(self, variable: str, block: Block) -> Instr:
        if block not in self.sealed_blocks:
            # Incomplete CFG
            result = self.emit(Phi(block))
            self.incomplete_phis[block][variable] = result
        elif len(self.preds[block]) == 1:
            # Optimize the common case of one predecessor: no phi needed
            (pred_value,) = self.preds[block]
            result = self.read_variable(variable, pred_value)
        else:
            # Break potential cycles with operandless phi
            result = self.emit(Phi(block))
            self.write_variable(variable, block, result)
            result = self.add_phi_operands(variable, result)
        self.write_variable(variable, block, result)
        return result

    def add_phi_operands(self, variable: str, phi: Phi):
        # Determine operands from predecessors
        for pred in self.preds[phi.block]:
            phi.append_operand(self.read_variable(variable, pred))
        return phi  # TODO(max): try_remove_trivial_phi

    def peek(self) -> Token|None:
        try:
            return self.source.peek()
        except StopIteration:
            return None

    def advance(self) -> Token|None:
        try:
            return next(self.source)
        except StopIteration:
            return None

    def match(self, token_type: type) -> Token|None:
        if isinstance(self.peek(), token_type):
            return self.advance()
        return None

    def match_punct(self, value: str) -> Token|None:
        peek = self.peek()
        if isinstance(peek, TPunct) and peek.value == value:
            return self.advance()
        return None

    def parse_error(self, message):
        raise ParseError(message)

    def expect(self, token_type: type) -> Token:
        if isinstance(self.peek(), token_type):
            return self.advance()
        self.parse_error(f"Unexpected token `{self.peek()}'")

    def expect_punct(self, token: str) -> Token:
        peek = self.peek()
        if isinstance(peek, TPunct) and peek.value == token:
            return self.advance()
        self.parse_error(f"Unexpected token `{peek}'")

    def parse_program(self) -> Program:
        while self.peek():
            self.parse_toplevel()
        if not self.block.has_terminator():
            self.emit(Return(self.emit(Int(0))))
        return self.program

    def parse_toplevel(self):
        if self.match(TFunc):
            return self.parse_func_decl()
        return self.parse_statement()

    def parse_func_decl(self):
        raise NotImplementedError("function declaration")

    def parse_statement(self):
        if self.match(TVar):
            return self.parse_var_decl()
        if self.match(TIf):
            return self.parse_if()
        if self.match(TPrint):
            value = self.parse_expression()
            self.expect_punct(";")
            self.emit(Print(value))
            return
        self.parse_expression()
        self.expect_punct(";")

    def parse_var_decl(self):
        block = self.block
        name = self.expect(TIdent)
        if self.match_punct(";"):
            self.write_variable(name.value, block, self.emit(Undefined()))
            return
        self.expect_punct("=")
        value = self.parse_expression()
        self.expect_punct(";")
        self.write_variable(name.value, block, value)

    def parse_statement_block(self):
        self.expect_punct("{")
        while (peek := self.peek()):
            if isinstance(peek, TPunct) and peek.value == "}":
                break
            self.parse_statement()
        self.expect_punct("}")

    def parse_if(self):
        cond = self.parse_expression()
        iftrue_block = self.func.new_block()
        iffalse_block = self.func.new_block()
        self.emit(CondBranch(cond, iftrue_block, iffalse_block))
        self.block = iftrue_block
        self.parse_statement_block()
        if self.match(TElse):
            join_block = self.func.new_block()
            self.emit(Branch(join_block))
            self.block = iffalse_block
            self.parse_statement_block()
            self.emit(Branch(join_block))
            self.block = join_block
        else:
            self.emit(Branch(iffalse_block))
            self.block = iffalse_block

    def emit(self, instr: Instr) -> Instr:
        if isinstance(instr, Branch):
            self.preds[instr.target].add(self.block)
        elif isinstance(instr, CondBranch):
            self.preds[instr.iftrue].add(self.block)
            self.preds[instr.iffalse].add(self.block)
        self.block.emit(instr)
        return instr

    def seal_block(self, block: Block):
        for variable in self.incomplete_phis[block]:
            self.add_phi_operands(variable, self.incomplete_phis[block][variable])
        self.sealed_blocks.add(block)

    def parse_atom(self) -> Instr:
        if (token := self.match(TInt)):
            return self.emit(Int(token.value))
        if (token := self.match(TIdent)):
            if self.match_punct("="):
                rhs = self.parse_expression()
                self.write_variable(token.value, self.block, rhs)
                return rhs
            return self.read_variable(token.value, self.block)
        self.parse_error(f"Unexpected token `{self.peek()}'")

    def parse_expression(self, min_prec: int = 0):
        lhs = self.parse_atom()
        while True:
            token = self.peek()
            if isinstance(token, TPunct) and (op := token.value) in OPS:
                op_prec = PREC[op]
                if op_prec < min_prec:
                    break
                self.advance()
                next_prec = op_prec + 1 if ASSOC[op] == "left" else op_prec
                rhs = self.parse_expression(next_prec)
                if op == "+":
                    lhs = self.emit(Add(lhs, rhs))
                elif op == "-":
                    lhs = self.emit(Sub(lhs, rhs))
                elif op == "*":
                    lhs = self.emit(Mul(lhs, rhs))
                elif op == "/":
                    lhs = self.emit(Div(lhs, rhs))
                elif op == "<":
                    lhs = self.emit(Less(lhs, rhs))
                else:
                    raise NotImplementedError(f"binary op {op}")
            elif isinstance(token, TPunct) and token.value == "(":
                raise NotImplementedError("function application")
            else:
                break
        return lhs

@dataclasses.dataclass
class InstrNumber:
    instrs: dict[Instr, int] = dataclasses.field(init=False, default_factory=dict)

    def name(self, instr: Instr) -> str:
        result = self.instrs.get(instr)
        if result is None:
            result = self.instrs[instr] = len(self.instrs)
        return f"v{result}"

def write_instr(f: io.BufferedWriter, gvn: InstrNumber, instr: Instr):
    if isinstance(instr, CondBranch):
        f.write(f"CondBranch {gvn.name(instr.operands[0])}, {instr.iftrue.name()}, {instr.iffalse.name()}")
    elif isinstance(instr, Branch):
        f.write(f"Branch {instr.target.name()}")
    elif isinstance(instr, HasOperands):
        operands = [gvn.name(operand) for operand in instr.operands]
        f.write(f"{type(instr).__name__} {', '.join(operands)}")
    else:
        f.write(str(instr))

def write_block(f: io.BufferedWriter, gvn: InstrNumber, block: Block):
    f.write(f"  {block.name()} {{\n")
    for instr in block.instrs:
        f.write("    ")
        if not isinstance(instr, Terminator):
            f.write(f"{gvn.name(instr)} = ")
        write_instr(f, gvn, instr)
        f.write("\n")
    f.write("  }\n")

def write_function(f: io.BufferedWriter, function: Function):
    gvn = InstrNumber()
    f.write(f"func {function.name} {{\n")
    for block in function.rpo():
        write_block(f, gvn, block)
    f.write("}\n")

def write_program(f: io.BufferedWriter, program: Program):
    for function in program.functions:
        write_function(f, function)

lexer = Lexer(PeekableString("""
var LAST = 30;
var a = 3;
var b = 4;
var c;
if a < b {
  c = 1;
} else {
  c = 2;
}
print c;

// A function declaration
// func fibonacci(n int) int {
//     if n > 1 {
//         return fibonacci(n-1) + fibonacci(n-2);
//     } else {
//         return 1;
//     }
// }
// 
// func main() int {
//     var n int = 0.5;
//     while n < LAST {
//         print fibonacci(n);
//         n = n + 1;
//     }
//     return 0;
// }
"""))
parser = Parser(lexer)
parser.parse_program()
with io.StringIO() as f:
    write_program(f, parser.program)
    print(f.getvalue())
