grammar Expr;
start_ : expr (';' expr)* EOF;
expr : atom | ('+' | '-') expr | expr '**' expr | expr ('*' | '/') expr | expr ('+' | '-') expr | '(' expr ')' | atom ;
atom : INT ;
INT : [0-9]+ ;
WS : [ \t\n\r]+ -> skip ;
