import ast
import inspect

import asttokens
import astunparse


def transform_top_level_function(node: ast.FunctionDef):

    tape_name = ast.Name()
    tape_name.id = "FunctionTape"

    tape = ast.Call()
    tape.func = tape_name
    arg_names = list((map(lambda arg: arg.arg, node.args.args)))
    tape.args = list((map(lambda arg: ast.Str(arg), arg_names)))
    tape.keywords = {}

    v = ast.withitem()
    v.context_expr = tape
    v.optional_vars = ast.Name(node.name)

    with_block = ast.With()
    with_block.body = node.body
    with_block.items = [v]
    return with_block, node.name, arg_names


class ControlFlowTransformer(ast.NodeTransformer):

    def __init__(self, function_name, arg_names, tokens, globals, line_num, functions_to_compile, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # constants
        self.tokens = tokens
        self.function_name = function_name
        self.arg_names = arg_names
        self.globals = globals
        self.lin_num = line_num
        self.functions_to_compile = functions_to_compile
        self.called_functions = set()

        # state
        self.pennylane_script_expr = False

    def visit_If(self, node):
        self.generic_visit(node)

        if_tape_name = ast.Name()
        if_tape_name.id = "IfTape"

        if_tape = ast.Call()
        if_tape.func = if_tape_name
        if_tape.args = [node.test]
        if_tape.keywords = {}

        with_block = ast.With()
        with_block.body = node.body
        with_block.items = [if_tape]
        return with_block

    def visit_While(self, node):
        self.generic_visit(node)

        tape_name = ast.Name()
        tape_name.id = "WhileTape"

        tape = ast.Call()
        tape.func = tape_name
        tape.args = [node.test]
        tape.keywords = {}

        with_block = ast.With()
        with_block.body = node.body
        with_block.items = [tape]
        return with_block

    def visit_For(self, node):
        self.generic_visit(node)

        tape_name = ast.Name()
        tape_name.id = "ForTape"

        tape = ast.Call()
        tape.func = tape_name
        tape.args = [node.test]
        tape.keywords = {}

        with_block = ast.With()
        with_block.body = node.body
        with_block.items = [tape]
        return with_block

    def visit_Starred(self, node):
        self.dont_allow(node)

    def visit_FunctionDef(self, node):
        self.dont_allow(node)

    def visit_Assign(self, node):
        self.dont_allow(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            # root node
            if self.globals[node.value.id].__name__ == "pennylane":
                return node
            else:
                raise ValueError("bad import")

        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "FunctionTape":
            return node
        if isinstance(node.func, ast.Name):
            # this is a local function
            self.called_functions.add(self.globals[astunparse.unparse(node.func).strip()])
            for i, arg in enumerate(node.args):
                l = ast.Lambda()
                l.args = []
                l.body = arg
                node.args[i] = l

        return node

    def dont_allow(self, node):
        code = self.tokens.get_text(node)
        code_lines = code.split("\n")
        code_lines_snippet = code_lines[:min(2, len(code_lines))]
        leading_spaces = len(code_lines_snippet[0]) - len(code_lines_snippet[0].lstrip())
        line_num = self.lin_num + node.lineno  # off by 2 for now
        code_lines_snippet[0] = f"{line_num} {code_lines_snippet[0]}"
        if len(code_lines_snippet) == 2:
            code_lines_snippet[1] = f"{line_num + 1} {code_lines_snippet[1]}"
        code_lines_snippet.insert(1, f"{' ' * (leading_spaces * len(str(line_num)))}<<< {type(node).__name__} is not allowed in pennylane-script")
        raise ValueError("\n" + "\n".join(code_lines_snippet))


    def visit_Name(self, node):

        if node.id in self.arg_names:
            attr = ast.Attribute()
            attr.attr = node.id
            attr.value = ast.Name(self.function_name)
            return attr
        return node


def script(fn):
    fn_source = inspect.getsource(fn)

    print(fn_source)
    fn_ast = ast.parse(fn_source)
    tokens = asttokens.ASTTokens(fn_source)
    tokens.mark_tokens(fn_ast)
    trimmed_ast = fn_ast.body[0]
    functions_to_compile = []
    tape_ast, function_name, arg_names = transform_top_level_function(trimmed_ast)
    parent_frame = inspect.currentframe().f_back
    fun_lin_num = parent_frame.f_lineno + 1
    cft = ControlFlowTransformer(function_name, arg_names, tokens, parent_frame.f_globals, fun_lin_num, functions_to_compile)
    transformed_ast = cft.visit(tape_ast)
    print(astunparse.unparse(transformed_ast))