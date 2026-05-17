#ifndef CODEGEN_H
#define CODEGEN_H

#include "ast.h"

// Generate fully optimized C source code from parsed PhoLang AST
// Returns 0 on success, -1 on failure
int codegen_generate(const ASTNode *program, const char *output_filepath);

#endif // CODEGEN_H
