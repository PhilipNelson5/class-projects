%{
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "src/Factory.hpp"

// abstract node classes
#include "src/Node.hpp"
#include "src/ExpressionNode.hpp"
#include "src/StatementNode.hpp"

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
#include "src/ProcedureOrFunctionDeclarationNode.hpp"
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

class Type;

#define YYERROR_VERBOSE 1
#define DEBUG 1

extern "C" int yylex();
extern char * yytext;
extern std::string yylinetxt;
extern unsigned int yylineno;
extern unsigned int yycolumn;
extern std::shared_ptr<ProgramNode> programNode;
void yyerror(const char*);

%}

%define parse.trace

%union
{
  int int_val;
  char char_val;
  char * str_val;

  Field * field;

  Node * node;
  StatementNode * statementNode;
  ExpressionNode * expressionNode;
  TypeNode * type;

  AssignmentStatementNode * assignmentNode;
  BodyNode * bodyNode;
  ConstantDeclarationNode * constDeclNode;
  FormalParameter * formalParameter;
  IdentifierNode * identifier;
  IfStatementNode * ifStatementNode;
  LvalueNode * lvalue;
  ProcedureDeclarationNode * procedureDeclarationNode;
  ProcedureCallNode * procedureCallNode;
  ForStatementNode * forStatementNode;
  FunctionCallNode * functionCallNode;
  FunctionDeclarationNode * functionDeclarationNode;
  ProcedureOrFunctionDeclarationNode * procedureOrFunctionDeclarationNode;
  ReadStatementNode * readStatementNode;
  ReturnStatementNode * returnStatementNode;
  StopStatementNode * stopStatementNode;
  TypeDeclarationNode * typeDeclarationNode;
  VariableDeclarationNode * varDeclNode;
  WriteStatementNode * writeStatementNode;

  ListNode<ConstantDeclarationNode> * constDelcList;
  ListNode<ExpressionNode> * expressionList;
  ListNode<Field> * fieldList;
  ListNode<FormalParameter> * formalParameterList;
  ListNode<LvalueNode> * lValueList;
  ListNode<StatementNode> * statementList;
  ListNode<TypeDeclarationNode> * typeDeclarationList;
  ListNode<VariableDeclarationNode> * varDelcList;
  ListNode<std::pair<std::shared_ptr<ExpressionNode>, std::vector<std::shared_ptr<StatementNode>>>> * elseIfList;
  ListNode<std::string> * identList;
  ListNode<ProcedureOrFunctionDeclarationNode> * procedureAndFunctionDeclList;
}

%token ARRAY_T
%token BEGIN_T
%token CHR_T
%token CONST_T
%token DO_T
%token DOWNTO_T
%token ELSE_T
%token ELSEIF_T
%token END_T
%token FOR_T
%token FORWARD_T
%token FUNCTION_T
%token IF_T
%token OF_T
%token ORD_T
%token PRED_T
%token PROCEDURE_T
%token READ_T
%token RECORD_T

%token REF_T
%token REPEAT_T
%token RETURN_T
%token STOP_T
%token SUCC_T
%token THEN_T
%token TO_T
%token TYPE_T
%token UNTIL_T
%token VAR_T
%token WHILE_T
%token WRITE_T

%token ID_T;

%token PLUS_T
%token MINUS_T
%token UNARY_MINUS_T
%token MULTIPLY_T
%token DIVIDE_T
%token AND_T
%token OR_T
%token NOT_T
%token EQUAL_T
%token NEQUAL_T
%token LT_T
%token LTE_T
%token GT_T
%token GTE_T
%token DOT_T
%token COMMA_T
%token COLON_T
%token SEMI_COLON_T
%token OPEN_PAREN_T
%token CLOSE_PAREN_T
%token OPEN_BRACKET_T
%token CLOSE_BRACKET_T
%token ASSIGN_T
%token MOD_T

%token NUMBER_T
%token CHAR_T
%token STRING_T

%type <int_val>  NUMBER_T
%type <str_val>  STRING_T
%type <char_val> CHAR_T
%type <str_val>  ID_T

%type <node> Program
%type <constDelcList> OptConstDecls
%type <constDelcList> ConstDeclList
%type <constDeclNode> ConstDecl
%type <procedureAndFunctionDeclList> OptProcedureAndFunctionDeclList
%type <procedureAndFunctionDeclList> ProcedureAndFunctionDeclList
%type <procedureDeclarationNode> ProcedureDecl
%type <functionDeclarationNode> FunctionDecl
%type <formalParameterList> FormalParameters
%type <formalParameterList> FormalParameterList
%type <formalParameter> FormalParameter
%type <bodyNode> Body
%type <statementList> Block
%type <typeDeclarationList> OptTypeDecls
%type <typeDeclarationList> TypeDeclList
%type <typeDeclarationNode> TypeDecl
%type <type> Type
%type <type> SimpleType
%type <type> RecordType
/*%type <recordType> RecordType*/
%type <fieldList> OptFieldList
%type <fieldList> FieldList
%type <field> Field
%type <type> ArrayType
/*%type <arrayType> ArrayType*/
%type <identList> IdentList
%type <varDelcList> OptVariableDecls
%type <varDelcList> VariableDeclList
%type <varDeclNode> VariableDecl
%type <statementList> StatementList
%type <statementNode> Statement
%type <assignmentNode> Assignment
%type <ifStatementNode> IfStatement
%type <elseIfList> OptElseIfStatementList
%type <elseIfList> ElseIfStatementList
%type <statementList> OptElseStatement
%type <statementNode> WhileStatement
%type <statementNode> RepeatStatement
%type <forStatementNode> ForStatement
%type <stopStatementNode> StopStatement
%type <returnStatementNode> ReturnStatement
%type <readStatementNode> ReadStatement
%type <lValueList> LValueList
%type <writeStatementNode> WriteStatement
%type <expressionList> OptExpressionList
%type <expressionList> ExpressionList
%type <functionCallNode> FunctionCall
%type <procedureCallNode> ProcedureCall
%type <expressionNode> Expression
%type <lvalue> LValue

%left      OR_T
%left      AND_T
%right     NOT_T
%nonassoc  EQUAL_T NEQUAL_T LT_T LTE_T GT_T GTE_T
%left      PLUS_T MINUS_T
%left      MULTIPLY_T DIVIDE_T MOD_T
%right     UNARY_MINUS_T

%%

/* Opt prefix means zero or one   (?) */
/* List postfix means one or more (+) */
/* OptList means zero or more     (*) */

Program                         : OptConstDecls
                                  OptTypeDecls
                                  OptVariableDecls
                                  OptProcedureAndFunctionDeclList
                                  Block DOT_T
                                  {
                                    programNode = std::make_shared<ProgramNode>($1, $2, $3, $4, $5);
                                  }
                                ;

/* 3.1.1 Constant Declerations */
OptConstDecls                   : CONST_T ConstDeclList { $$ = $2; }
                                | /* λ */ { $$ = nullptr; }
                                ;

ConstDeclList                   : ConstDeclList ConstDecl
                                  {
                                    $$ = new ListNode<ConstantDeclarationNode>($2, $1);
                                  }
                                | ConstDecl
                                  {
                                    $$ = new ListNode<ConstantDeclarationNode>($1);
                                  }
                                ;

ConstDecl                       : ID_T EQUAL_T Expression SEMI_COLON_T
                                  {
                                    $$ = makeConstantDeclarationNode($1, $3);
                                  }
                                ;

/* 3.1.2 Procedure and Function Declarations */
OptProcedureAndFunctionDeclList : ProcedureAndFunctionDeclList { $$ = $1; }
                                | /* λ */ { $$ = nullptr; }
                                ;

ProcedureAndFunctionDeclList    : ProcedureAndFunctionDeclList ProcedureDecl
                                  {
                                   $$ = new ListNode<ProcedureOrFunctionDeclarationNode>($2, $1);
                                  }
                                | ProcedureAndFunctionDeclList FunctionDecl
                                  {
                                   $$ = new ListNode<ProcedureOrFunctionDeclarationNode>($2, $1);
                                  }
                                | ProcedureDecl
                                  {
                                    $$ = new ListNode<ProcedureOrFunctionDeclarationNode>($1);
                                  }
                                | FunctionDecl
                                  {
                                    $$ = new ListNode<ProcedureOrFunctionDeclarationNode>($1);
                                  }
                                ;

ProcedureDecl                   : PROCEDURE_T ID_T OPEN_PAREN_T FormalParameters CLOSE_PAREN_T
                                    SEMI_COLON_T FORWARD_T SEMI_COLON_T
                                  {
                                    BodyNode * body = nullptr;
                                    $$ = new ProcedureDeclarationNode($2, $4, body);
                                  }
                                | PROCEDURE_T ID_T OPEN_PAREN_T FormalParameters CLOSE_PAREN_T
                                    SEMI_COLON_T Body SEMI_COLON_T
                                  {
                                    $$ = new ProcedureDeclarationNode($2, $4, $7);
                                  }
                                ;

FunctionDecl                    : FUNCTION_T ID_T OPEN_PAREN_T FormalParameters CLOSE_PAREN_T
                                    COLON_T Type SEMI_COLON_T FORWARD_T SEMI_COLON_T
                                  {
                                    BodyNode * body = nullptr;
                                    $$ = new FunctionDeclarationNode($2, $4, $7, body);
                                  }
                                | FUNCTION_T ID_T OPEN_PAREN_T FormalParameters CLOSE_PAREN_T
                                    COLON_T Type SEMI_COLON_T Body SEMI_COLON_T
                                  {
                                    $$ = new FunctionDeclarationNode($2, $4, $7, $9);
                                  }
                                ;

FormalParameters                : FormalParameterList  { $$ = $1; }
                                | /* λ */ { $$ = nullptr; }
                                ;

FormalParameterList             : FormalParameterList SEMI_COLON_T FormalParameter
                                  {
                                    $$ = new ListNode<FormalParameter>($3, $1);
                                  }
                                | FormalParameter
                                  {
                                    $$ = new ListNode<FormalParameter>($1);
                                  }
                                ;

FormalParameter                 : VAR_T IdentList COLON_T Type
                                  {
                                    $$ = new FormalParameter($2, $4, FormalParameter::PassBy::VAL);
                                  }
                                | REF_T IdentList COLON_T Type
                                  {
                                    $$ = new FormalParameter($2, $4, FormalParameter::PassBy::REF);
                                  }
                                |       IdentList COLON_T Type
                                  {
                                    $$ = new FormalParameter($1, $3);
                                  }
                                ;

Body                            : OptConstDecls OptTypeDecls OptVariableDecls Block
                                  {
                                    $$ = new BodyNode($1, $2, $3, $4);
                                  }
                                ;

Block                           : BEGIN_T StatementList END_T { $$ = $2; }
                                ;

/* 3.1.3 Type Declerations */
OptTypeDecls                    : TYPE_T TypeDeclList { $$ = $2; }
                                | /* λ */ { $$ = nullptr; }
                                ;

TypeDeclList                    : TypeDeclList TypeDecl
                                  {
                                    $$ = new ListNode<TypeDeclarationNode>($2, $1);
                                  }
                                | TypeDecl
                                  {
                                    $$ = new ListNode<TypeDeclarationNode>($1);
                                  }
                                ;

TypeDecl                        : ID_T EQUAL_T Type SEMI_COLON_T
                                  {
                                    $$ = new TypeDeclarationNode($1, $3);
                                  }
                                ;

Type                            : SimpleType { $$ = $1; }
                                | RecordType { $$ = $1; }
                                | ArrayType  { $$ = $1; }
                                ;

SimpleType                      : ID_T
                                  {
                                    $$ = new TypeNode($1);
                                  }
                                ;

RecordType                      : RECORD_T OptFieldList END_T
                                  {
                                    $$ = new TypeNode(std::make_shared<RecordType>($2));
                                  }
                                ;

OptFieldList                    : FieldList { $$ = $1; }
                                | /* λ */ { $$ = nullptr; }
                                ;

FieldList                       : FieldList Field
                                  {
                                    $$ = new ListNode<Field>($2, $1);
                                  }
                                | Field
                                  {
                                    $$ = new ListNode<Field>($1);
                                  }
                                ;

Field                           : IdentList COLON_T Type SEMI_COLON_T { $$ = new Field($1, $3); }
                                ;

ArrayType                       : ARRAY_T
                                  OPEN_BRACKET_T Expression COLON_T Expression CLOSE_BRACKET_T
                                  OF_T Type
                                  {
                                    $$ = new TypeNode(std::make_shared<ArrayType>($3, $5, $8));
                                  }
                                ;

IdentList                       : IdentList COMMA_T ID_T
                                  {
                                    $$ = new ListNode<std::string>(new std::string($3), $1);
                                  }
                                | ID_T
                                  {
                                    $$ = new ListNode<std::string>(new std::string($1));
                                  }
                                ;

/* 3.1.4 Variable Declerations */
OptVariableDecls                : VAR_T VariableDeclList { $$ = $2; }
                                | /* λ */ { $$ = nullptr; }
                                ;

VariableDeclList                : VariableDeclList VariableDecl
                                  {
                                    $$ = new ListNode<VariableDeclarationNode>($2, $1);
                                  }
                                | VariableDecl
                                  {
                                    $$ = new ListNode<VariableDeclarationNode>($1);
                                  }
                                ;

VariableDecl                    : IdentList COLON_T Type SEMI_COLON_T
                                  {
                                    $$ = new VariableDeclarationNode($1, $3);
                                  }
                                ;

/* 3.2   CPSL Statements */
StatementList                   : StatementList SEMI_COLON_T Statement
                                  {
                                    $$ = new ListNode<StatementNode>($3, $1);
                                  }
                                | Statement
                                  {
                                    $$ = new ListNode<StatementNode>($1);
                                  }
                                ;

Statement                       : Assignment      { $$ = $1; }
                                | IfStatement     { $$ = $1; }
                                | WhileStatement  { $$ = $1; }
                                | RepeatStatement { $$ = $1; }
                                | ForStatement    { $$ = $1; }
                                | StopStatement   { $$ = $1; }
                                | ReturnStatement { $$ = $1; }
                                | ReadStatement   { $$ = $1; }
                                | WriteStatement  { $$ = $1; }
                                | ProcedureCall   { $$ = $1; }
                                |     /* λ */     { $$ = nullptr; }
                                ;

Assignment                      : LValue ASSIGN_T Expression
                                  {
                                    $$ = new AssignmentStatementNode($1, $3);
                                  }
                                ;

IfStatement                     : IF_T Expression THEN_T
                                  StatementList
                                  OptElseIfStatementList
                                  OptElseStatement END_T
                                  {
                                    $$ = new IfStatementNode($2, $4, $5, $6);
                                  }
                                ;

OptElseIfStatementList          : ElseIfStatementList { $$ = $1; }
                                | /* λ */ { $$ = nullptr; }
                                ;

ElseIfStatementList             : ElseIfStatementList ELSEIF_T Expression THEN_T StatementList
                                  {
                                    $$ = new ListNode<
                                               std::pair<
                                                 std::shared_ptr<ExpressionNode>,
                                                 std::vector<std::shared_ptr<StatementNode>>
                                               >
                                             >(new std::pair<
                                                     std::shared_ptr<ExpressionNode>,
                                                     std::vector<std::shared_ptr<StatementNode>>
                                                   > ($3, ListNode<StatementNode>::makeVector($5)), $1);
                                  }
                                | ELSEIF_T Expression THEN_T StatementList
                                  {
                                    $$ = new ListNode<
                                               std::pair<
                                                 std::shared_ptr<ExpressionNode>,
                                                 std::vector<std::shared_ptr<StatementNode>>
                                               >
                                             >(new std::pair<
                                                     std::shared_ptr<ExpressionNode>,
                                                     std::vector<std::shared_ptr<StatementNode>>
                                                   > ($2, ListNode<StatementNode>::makeVector($4)));
                                  }

OptElseStatement                : ELSE_T StatementList { $$ = $2; }
                                | /* λ */ { $$ = nullptr; }
                                ;

WhileStatement                  : WHILE_T Expression DO_T StatementList END_T
                                  {
                                    $$ = new WhileStatementNode($2, $4);
                                  }
                                ;

RepeatStatement                 : REPEAT_T StatementList UNTIL_T Expression
                                  {
                                    $$ = new RepeatStatementNode($2, $4);
                                  }
                                ;

ForStatement                    : FOR_T ID_T ASSIGN_T Expression TO_T Expression
                                    DO_T StatementList END_T
                                  {
                                    $$ = new ForStatementNode(
                                      new IdentifierNode($2), $4, $6, $8, ForStatementNode::Type::TO);
                                  }
                                | FOR_T ID_T ASSIGN_T Expression DOWNTO_T Expression
                                    DO_T StatementList END_T
                                  {
                                    $$ = new ForStatementNode(
                                      new IdentifierNode($2), $4, $6, $8, ForStatementNode::Type::DOWNTO);
                                  }
                                ;

StopStatement                   : STOP_T { $$ = new StopStatementNode(); }
                                ;

ReturnStatement                 : RETURN_T { $$ = new ReturnStatementNode(); }
                                | RETURN_T Expression { $$ = new ReturnStatementNode($2); }
                                ;

ReadStatement                   : READ_T OPEN_PAREN_T LValueList CLOSE_PAREN_T
                                  {
                                    $$ = new ReadStatementNode($3);
                                  }
                                ;

LValueList                      : LValueList COMMA_T LValue
                                  {
                                    $$ = new ListNode<LvalueNode>($3, $1);
                                  }
                                | LValue
                                  {
                                    $$ = new ListNode<LvalueNode>($1);
                                  }
                                ;

WriteStatement                  : WRITE_T OPEN_PAREN_T ExpressionList CLOSE_PAREN_T
                                  {
                                    $$ = new WriteStatementNode($3);
                                  }
                                ;

FunctionCall                    : ID_T OPEN_PAREN_T OptExpressionList CLOSE_PAREN_T
                                  {
                                    $$ = new FunctionCallNode($1, $3);
                                  }

                                ;

ProcedureCall                   : ID_T OPEN_PAREN_T OptExpressionList CLOSE_PAREN_T
                                  {
                                    $$ = new ProcedureCallNode($1, $3);
                                  }
                                ;

OptExpressionList               : ExpressionList { $$ = $1; }
                                | /* λ */ { $$ = nullptr; }
                                ;

ExpressionList                  : ExpressionList COMMA_T Expression
                                  {
                                    $$ = new ListNode<ExpressionNode>($3, $1);
                                  }
                                | Expression
                                  {
                                    $$ = new ListNode<ExpressionNode>($1);
                                  }
                                ;

/* 3.3   Expressions */

Expression                      : Expression OR_T Expression                   { $$ = new OrNode($1, $3); }
                                | Expression AND_T Expression                  { $$ = new AndNode($1, $3); }
                                | Expression EQUAL_T Expression                { $$ = makeEqualNode($1, $3); }
                                | Expression NEQUAL_T Expression               { $$ = makeNotEqualNode($1, $3); }
                                | Expression LTE_T Expression                  { $$ = new LessThanEqualNode($1, $3); }
                                | Expression GTE_T Expression                  { $$ = new GreaterThanEqualNode($1, $3); }
                                | Expression LT_T Expression                   { $$ = new LessThanNode($1, $3); }
                                | Expression GT_T Expression                   { $$ = new GreaterThanNode($1, $3); }
                                | Expression PLUS_T Expression                 { $$ = makeAddNode($1, $3); }
                                | Expression MINUS_T Expression                { $$ = makeSubtractNode($1, $3); }
                                | Expression MULTIPLY_T Expression             { $$ = makeMultiplyNode($1, $3); }
                                | Expression DIVIDE_T Expression               { $$ = makeDivideNode($1, $3); }
                                | Expression MOD_T Expression                  { $$ = makeModuloNode($1, $3); }
                                | NOT_T Expression                             { $$ = new NotNode($2); }
                                | MINUS_T Expression %prec UNARY_MINUS_T       { $$ = makeUnaryMinusNode($2); }
                                | OPEN_PAREN_T Expression CLOSE_PAREN_T        { $$ = $2; }
                                | FunctionCall                                 { $$ = $1; }
                                | CHR_T OPEN_PAREN_T Expression CLOSE_PAREN_T  { $$ = new CharacterExpressionNode($3); }
                                | ORD_T OPEN_PAREN_T Expression CLOSE_PAREN_T  { $$ = new OrdinalExpressionNode($3); }
                                | PRED_T OPEN_PAREN_T Expression CLOSE_PAREN_T { $$ = new PredecessorExpressionNode($3); }
                                | SUCC_T OPEN_PAREN_T Expression CLOSE_PAREN_T { $$ = new SuccessorExpressionNode($3); }
                                | LValue                                       { $$ = $1; }
                                | NUMBER_T                                     { $$ = new IntegerLiteralNode($1); }
                                | STRING_T                                     { $$ = new StringLiteralNode($1); }
                                | CHAR_T                                       { $$ = new CharacterLiteralNode($1); }
                                ;

LValue                          : LValue DOT_T ID_T
                                  {
                                    $$ = new MemberAccessNode($1, $3);
                                  }
                                | LValue OPEN_BRACKET_T Expression CLOSE_BRACKET_T
                                  {
                                    $$ = new SubscriptOperatorNode($1, $3);
                                  }
                                | ID_T
                                  {
                                    $$ = new IdentifierNode($1);
                                  }
                                ;

%%

void yyerror(const char* msg)
{
  std::string restofline;
  std::getline(std::cin, restofline);

  std::cerr << msg << std::endl;

  std::stringstream ss;
  ss << yylineno << ":" << yycolumn - strlen(yytext) << " " ;

  std::cerr << ss.str();
  std::cerr << yylinetxt << restofline << std::endl;

  for(auto i = 0u; i < ss.str().length(); ++i)
  {
    std::cout << " ";
  }

  for(auto i = 0u; i < yylinetxt.length()-1; ++i)
  {
    std::cout << "~";
  }

  std::cout << "^" << std::endl;
}
