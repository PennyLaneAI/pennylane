import sys
from antlr4 import *
from build.ExprLexer import ExprLexer
from build.ExprParser import ExprParser
from build.ExprVisitor import ExprVisitor
from build.ExprListener import ExprListener

class ListenerInterp(ExprListener):
    def __init__(self):
        self.result = {}

    def exitAtom(self, ctx:ExprParser.AtomContext):
        self.result[ctx] = int(ctx.getText())

    def exitExpr(self, ctx:ExprParser.ExprContext):
        if ctx.getChildCount() == 3:
            if ctx.getChild(0).getText() == "(":
                self.result[ctx] = self.result[ctx.getChild(1)]
            else:
                opc = ctx.getChild(1).getText()
                v1 = self.result[ctx.getChild(0)]
                v2 = self.result[ctx.getChild(2)]
                if opc == "+":
                    self.result[ctx] = v1 + v2
                elif opc == "-":
                    self.result[ctx] = v1 - v2
                elif opc == "*":
                    self.result[ctx] = v1 * v2
                elif opc == "/":
                    self.result[ctx] = v1 / v2
                else:
                    ctx.result[ctx] = 0
        elif ctx.getChildCount() == 2:
            opc = ctx.getChild(0).getText()
            if opc == "+":
                v = self.result[ctx.getChild(1)]
                self.result[ctx] = v
            elif opc == "-":
                v = self.result[ctx.getChild(1)]
                self.result[ctx] = - v
        elif ctx.getChildCount() == 1:
            self.result[ctx] = self.result[ctx.getChild(0)]

    def exitStart_(self, ctx:ExprParser.Start_Context):
        for i in range(0, ctx.getChildCount(), 2):
            print(self.result[ctx.getChild(i)])

class VisitorInterp(ExprVisitor):
    def visitAtom(self, ctx:ExprParser.AtomContext):
        return int(ctx.getText())

    def visitExpr(self, ctx:ExprParser.ExprContext):
        if ctx.getChildCount() == 3:
            if ctx.getChild(0).getText() == "(":
                return self.visit(ctx.getChild(1))
            op = ctx.getChild(1).getText()
            v1 = self.visit(ctx.getChild(0))
            v2 = self.visit(ctx.getChild(2))
            if op == "+":
                return v1 + v2
            if op == "-":
                return v1 - v2
            if op == "*":
                return v1 * v2
            if op == "/":
                return v1 / v2
            return 0
        if ctx.getChildCount() == 2:
            opc = ctx.getChild(0).getText()
            if opc == "+":
                return self.visit(ctx.getChild(1))
            if opc == "-":
                return - self.visit(ctx.getChild(1))
            return 0
        if ctx.getChildCount() == 1:
            return self.visit(ctx.getChild(0))
        return 0

    def visitStart_(self, ctx:ExprParser.Start_Context):
        for i in range(0, ctx.getChildCount(), 2):
            print(self.visit(ctx.getChild(i)))
        return 0

def main(argv):
    input_stream = FileStream(argv[1])
    lexer = ExprLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ExprParser(stream)
    tree = parser.start_()
    if parser.getNumberOfSyntaxErrors() > 0:
        print("syntax errors")
    else:
        linterp = ListenerInterp()
        walker = ParseTreeWalker()
        walker.walk(linterp, tree)

if __name__ == '__main__':
    main(sys.argv)
