%{
#include <cstdlib>
#include <string>

// abstract node classes
#include "src/Node.hpp"
#include "src/StatementNode.hpp"
#include "src/ExpressionNode.hpp"

// concrete node classes
#include "src/AddNode.hpp"
#include "src/AndNode.hpp"
#include "src/AssignmentStatementNode.hpp"
#include "src/BodyNode.hpp"
#include "src/CharacterExpressionNode.hpp"
#include "src/CharacterLiteralNode.hpp"
#include "src/ConstantDeclarationNode.hpp"
#include "src/DivideNode.hpp"
#include "src/EqualExpressionNode.hpp"
#include "src/ForStatementNode.hpp"
#include "src/FormalParameter.hpp"
#include "src/FunctionCallNode.hpp"
#include "src/FunctionDeclarationNode.hpp"
#include "src/GreaterThanEqualNode.hpp"
#include "src/GreaterThanNode.hpp"
#include "src/IdentifierNode.hpp"
#include "src/IfStatementNode.hpp"
#include "src/IntegerLiteralNode.hpp"
#include "src/LessThanEqualNode.hpp"
#include "src/LessThanNode.hpp"
#include "src/ListNode.hpp"
#include "src/LvalueNode.hpp"
#include "src/MemberAccessNode.hpp"
#include "src/ModuloNode.hpp"
#include "src/MultiplyNode.hpp"
#include "src/NotEqualExpressionNode.hpp"
#include "src/NotNode.hpp"
#include "src/OrNode.hpp"
#include "src/OrdinalExpressionNode.hpp"
#include "src/PredecessorExpressionNode.hpp"
#include "src/ProcedureCallNode.hpp"
#include "src/ProcedureDeclarationNode.hpp"
#include "src/ProgramNode.hpp"
#include "src/ReadStatementNode.hpp"
#include "src/RepeatStatementNode.hpp"
#include "src/ReturnStatementNode.hpp"
#include "src/StopStatementNode.hpp"
#include "src/StringLiteralNode.hpp"
#include "src/SubscriptOperatorNode.hpp"
#include "src/SubtractNode.hpp"
#include "src/SuccessorExpressionNode.hpp"
#include "src/SymbolTable.hpp"
#include "src/TypeDeclarationNode.hpp"
#include "src/TypeNode.hpp"
#include "src/UnaryMinusNode.hpp"
#include "src/VariableDeclarationNode.hpp"
#include "src/WhileStatementNode.hpp"
#include "src/WriteStatementNode.hpp"

#include "parser.hpp"
extern "C" int yylex();

unsigned int yycolumn = 1;
std::string yylinetxt = "";

#define YY_USER_ACTION \
  yycolumn += yyleng;  \
  yylinetxt += yytext;

%}

%option yylineno noyywrap

%%

array|ARRAY           { return ARRAY_T; }
begin|BEGIN           { return BEGIN_T; }
chr|CHR               { return CHR_T; }
const|CONST           { return CONST_T; }
do|DO                 { return DO_T; }
downto|DOWNTO         { return DOWNTO_T; }
else|ELSE             { return ELSE_T; }
elseif|ELSEIF         { return ELSEIF_T; }
end|END               { return END_T; }
for|FOR               { return FOR_T; }
forward|FORWARD       { return FORWARD_T; }
function|FUNCTION     { return FUNCTION_T; }
if|IF                 { return IF_T; }
of|OF                 { return OF_T; }
ord|ORD               { return ORD_T; }
pred|PRED             { return PRED_T; }
procedure|PROCEDURE   { return PROCEDURE_T; }
read|READ             { return READ_T; }
record|RECORD         { return RECORD_T; }
ref|REF               { return REF_T; }
repeat|REPEAT         { return REPEAT_T; }
return|RETURN         { return RETURN_T; }
stop|STOP             { return STOP_T; }
succ|SUCC             { return SUCC_T; }
then|THEN             { return THEN_T; }
to|TO                 { return TO_T; }
type|TYPE             { return TYPE_T; }
until|UNTIL           { return UNTIL_T; }
var|VAR               { return VAR_T; }
while|WHILE           { return WHILE_T; }
write|WRITE           { return WRITE_T; }
                        
[a-zA-Z][a-zA-Z0-9_]* { yylval.str_val = strdup(yytext); return ID_T; }
                        
"+"                   { return PLUS_T; }
"-"                   { return MINUS_T; }
"*"                   { return MULTIPLY_T; }
"/"                   { return DIVIDE_T; }
"&"                   { return AND_T; }
"|"                   { return OR_T; }
"~"                   { return NOT_T; }
"="                   { return EQUAL_T; }
"<>"                  { return NEQUAL_T; }
"<"                   { return LT_T; }
"<="                  { return LTE_T; }
">"                   { return GT_T; }
">="                  { return GTE_T; }
"."                   { return DOT_T; }
","                   { return COMMA_T; }
":"                   { return COLON_T; }
";"                   { return SEMI_COLON_T; }
"("                   { return OPEN_PAREN_T; }
")"                   { return CLOSE_PAREN_T; }
"["                   { return OPEN_BRACKET_T; }
"]"                   { return CLOSE_BRACKET_T; }
":="                  { return ASSIGN_T; }
"%"                   { return MOD_T;}
                        
0[0-7]+               { yylval.int_val = strtol(yytext, nullptr, 8); return NUMBER_T; }
0[xX][a-fA-F0-9]+     { yylval.int_val = strtol(yytext, nullptr, 16); return NUMBER_T; }
[0-9]+                { yylval.int_val = strtol(yytext, nullptr, 10); return NUMBER_T; }
                        
'\\n'                 { yylval.char_val = '\n'; return CHAR_T; }
'\\r'                 { yylval.char_val = '\r'; return CHAR_T; }
'\\b'                 { yylval.char_val = '\b'; return CHAR_T; }
'\\t'                 { yylval.char_val = '\t'; return CHAR_T; }
'\\f'                 { yylval.char_val = '\f'; return CHAR_T; }
'.'                   { yylval.char_val = yytext[1]; return CHAR_T; }
'\\.'                 { yylval.char_val = yytext[1]; return CHAR_T; }
                        
\"(\\.|[^"\\])*\"     { yylval.str_val = strdup(yytext); return STRING_T; }
                        
\$.*$                 { }
                        
[ \t]*                { }

[\n]                  { yycolumn = 1; yylinetxt = "";}

%%
