cmake_minimum_required(VERSION 2.8)
project(cpsl)

find_package(BISON)
find_package(FLEX)

bison_target(Parser parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
flex_target(Scanner scanner.l ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp)
add_flex_bison_dependency(Scanner Parser)

set(EXTRA_COMPILE_FLAGS "-g3 -std=c++17 -export-dynamic -DELPP_NO_DEFAULT_LOG_FILE")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

set(cpsl_srcs
  main.cpp
  src/Factory.hpp

  src/AddNode.cpp
  src/AddNode.hpp
  src/AndNode.cpp
  src/AndNode.hpp
  src/AssignmentStatementNode.cpp
  src/AssignmentStatementNode.hpp
  src/BodyNode.cpp
  src/BodyNode.hpp
  src/BooleanLiteralNode.cpp
  src/BooleanLiteralNode.hpp
  src/CharacterExpressionNode.cpp
  src/CharacterExpressionNode.hpp
  src/CharacterLiteralNode.cpp
  src/CharacterLiteralNode.hpp
  src/ConstantDeclarationNode.cpp
  src/ConstantDeclarationNode.hpp
  src/DeclarationNode.hpp
  src/DivideNode.cpp
  src/DivideNode.hpp
  src/EqualExpressionNode.cpp
  src/EqualExpressionNode.hpp
  src/ExpressionNode.cpp
  src/ExpressionNode.hpp
  src/ForStatementNode.cpp
  src/ForStatementNode.hpp
  src/FormalParameter.cpp
  src/FormalParameter.hpp
  src/FunctionCallNode.cpp
  src/FunctionCallNode.hpp
  src/FunctionDeclarationNode.cpp
  src/FunctionDeclarationNode.hpp
  src/GreaterThanEqualNode.cpp
  src/GreaterThanEqualNode.hpp
  src/GreaterThanNode.cpp
  src/GreaterThanNode.hpp
  src/IdentifierNode.cpp
  src/IdentifierNode.hpp
  src/IfStatementNode.cpp
  src/IfStatementNode.hpp
  src/IntegerLiteralNode.cpp
  src/IntegerLiteralNode.hpp
  src/LessThanEqualNode.cpp
  src/LessThanEqualNode.hpp
  src/LessThanNode.cpp
  src/LessThanNode.hpp
  src/ListNode.hpp
  src/LiteralNode.hpp
  src/LvalueNode.hpp
  src/MemberAccessNode.cpp
  src/MemberAccessNode.hpp
  src/ModuloNode.cpp
  src/ModuloNode.hpp
  src/MultiplyNode.cpp
  src/MultiplyNode.hpp
  src/Node.hpp
  src/NotEqualExpressionNode.cpp
  src/NotEqualExpressionNode.hpp
  src/NotNode.cpp
  src/NotNode.hpp
  src/OrNode.cpp
  src/OrNode.hpp
  src/OrdinalExpressionNode.cpp
  src/OrdinalExpressionNode.hpp
  src/PredecessorExpressionNode.cpp
  src/PredecessorExpressionNode.hpp
  src/ProcedureCallNode.cpp
  src/ProcedureCallNode.hpp
  src/ProcedureDeclarationNode.cpp
  src/ProcedureDeclarationNode.hpp
  src/ProgramNode.cpp
  src/ProgramNode.hpp
  src/ReadStatementNode.cpp
  src/ReadStatementNode.hpp
  src/RegisterPool.cpp
  src/RegisterPool.hpp
  src/RepeatStatementNode.cpp
  src/RepeatStatementNode.hpp
  src/ReturnStatementNode.cpp
  src/ReturnStatementNode.hpp
  src/stacktrace.hpp
  src/StatementNode.hpp
  src/StopStatementNode.cpp
  src/StopStatementNode.hpp
  src/StringLiteralNode.cpp
  src/StringLiteralNode.hpp
  src/SubscriptOperatorNode.cpp
  src/SubscriptOperatorNode.hpp
  src/SubtractNode.cpp
  src/SubtractNode.hpp
  src/SuccessorExpressionNode.cpp
  src/SuccessorExpressionNode.hpp
  src/SymbolTable.cpp
  src/SymbolTable.hpp
  src/Type.cpp
  src/Type.hpp
  src/TypeDeclarationNode.cpp
  src/TypeDeclarationNode.hpp
  src/TypeNode.cpp
  src/TypeNode.hpp
  src/UnaryMinusNode.cpp
  src/UnaryMinusNode.hpp
  src/Value.hpp
  src/VariableDeclarationNode.cpp
  src/VariableDeclarationNode.hpp
  src/WhileStatementNode.cpp
  src/WhileStatementNode.hpp
  src/WriteStatementNode.cpp
  src/WriteStatementNode.hpp

  src/log/easylogging++.h
  src/log/easylogging++.cc

  ${BISON_Parser_OUTPUTS}
  ${FLEX_Scanner_OUTPUTS}
)
source_group("Cpsl" FILES ${cpsl_srcs})

add_subdirectory(fmt)

add_executable(cpsl ${cpsl_srcs})
target_link_libraries(cpsl ${FLEX_LIBRARIES} ${BISON_LIBRARIES} fmt)
target_include_directories(cpsl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
