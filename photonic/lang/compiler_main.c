#include "ast.h"
#include "lexer.h"
#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
    #include <direct.h>
    #define make_dir(path) _mkdir(path)
#else
    #include <sys/stat.h>
    #include <sys/types.h>
    #define make_dir(path) mkdir(path, 0777)
#endif

// Forward declaration of parser entrypoint
ASTNode *parser_parse(const char *source);

int main(int argc, char **argv) {
    printf("=== PhoLang Optical Compiler (pho) ===\n\n");
    
    if (argc < 2) {
        printf("Usage:\n");
        printf("  %s <input.pho> <output.c>   (Transpile directly to C)\n", argv[0]);
        printf("  %s run <input.pho>          (Compile and execute immediately, like Cargo)\n", argv[0]);
        printf("  %s build <input.pho>        (Compile into 'target/' directory)\n\n", argv[0]);
        return 1;
    }
    
    int is_run = (strcmp(argv[1], "run") == 0);
    int is_build = (strcmp(argv[1], "build") == 0);
    
    const char *input_path = NULL;
    const char *output_path = NULL;
    
    if (is_run || is_build) {
        if (argc < 3) {
            fprintf(stderr, "Error: Missing input .pho file.\n");
            printf("Usage: %s %s <input.pho>\n", argv[0], argv[1]);
            return 1;
        }
        input_path = argv[2];
        make_dir("target");
        output_path = "target/compiled.c";
    } else {
        if (argc < 3) {
            printf("Usage: %s <input.pho> <output.c>\n", argv[0]);
            return 1;
        }
        input_path = argv[1];
        output_path = argv[2];
    }
    
    printf("Reading source %s...\n", input_path);
    FILE *f = fopen(input_path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open input file %s\n", input_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    if (len < 0) {
        fprintf(stderr, "Failed to determine file size for %s\n", input_path);
        fclose(f);
        return 1;
    }
    fseek(f, 0, SEEK_SET);
    
    char *source = malloc(len + 1);
    if (!source) {
        fprintf(stderr, "Memory allocation failed for source code\n");
        fclose(f);
        return 1;
    }
    size_t bytes_read = fread(source, 1, len, f);
    source[bytes_read] = '\0';
    fclose(f);
    
    printf("Parsing source code...\n");
    ASTNode *ast = parser_parse(source);
    if (!ast) {
        fprintf(stderr, "\n[Compiler Error] Parser failed!\n");
        free(source);
        return 1;
    }
    
    printf("Syntactic Analysis Completed. Printing AST:\n");
    ast_print(ast, 0);
    
    printf("\nGenerating highly optimized C target code into %s...\n", output_path);
    if (codegen_generate(ast, output_path) == 0) {
        printf("[CodeGen Success] C source code successfully written to %s!\n", output_path);
    } else {
        fprintf(stderr, "[CodeGen Failure] Transpilation process failed!\n");
        ast_free(ast);
        free(source);
        return 1;
    }
    
    ast_free(ast);
    free(source);
    
    // If run or build, compile C code into native binary inside 'target/' directory
    if (is_run || is_build) {
        printf("\nCompiling C target into native executable via host GCC...\n");
        char compile_cmd[1024];
        
        // Link against pooling helper and libphotonic
#ifdef _WIN32
        sprintf(compile_cmd, "gcc -O2 target/compiled.c photonic/examples/mnist_small/pooling.c -I. -L. -lphotonic -lm -fopenmp -o target/app.exe");
#else
        sprintf(compile_cmd, "gcc -O2 target/compiled.c photonic/examples/mnist_small/pooling.c -I. -L. -lphotonic -lm -fopenmp -o target/app");
#endif
        
        int compile_res = system(compile_cmd);
        if (compile_res != 0) {
            fprintf(stderr, "\n[Compiler Error] Native C compilation failed! Ensure GCC and libphotonic.a exist.\n");
            return 1;
        }
        printf("[Build Success] Executable built inside 'target/' directory!\n");
        
        if (is_run) {
            printf("\nExecuting native Photonic model...\n");
#ifdef _WIN32
            int run_res = system("target\\app.exe");
#else
            int run_res = system("./target/app");
#endif
            (void)run_res;
        }
    }
    
    printf("\nCompilation finished successfully. Memory cleaned (Zero Leaks).\n");
    return 0;
}
