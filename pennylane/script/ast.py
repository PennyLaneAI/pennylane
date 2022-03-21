import ast


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
    return with_block, arg_names


class ControlFlowTransformer(ast.NodeTransformer):

    def __init__(self, arg_names, tokens, globals, line_num, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # constants
        self.tokens = tokens
        self.arg_names = arg_names
        self.globals = globals
        self.lin_num = line_num

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

    def dont_allow(self, node):
        code = self.tokens.get_text(node)
        code_lines = code.split("\n")
        code_lines.insert(1, f"<<< {type(node).__name__} is not allowed in pennylane-script")
        code_lines_snippet = code_lines[:min(3, len(code_lines))]
        raise ValueError("\n" + "\n".join(code_lines_snippet))


    def visit_Name(self, node):

        if node.id in self.arg_names:
            attr = ast.Attribute()
            attr.attr = node.id
            attr.value = ast.Name("circuit")
            return attr
        return node