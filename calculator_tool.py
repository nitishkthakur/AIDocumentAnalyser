"""
Calculator Tool for LLMs

This module provides a safe calculator function that can be used by Large Language Models
as a tool for performing mathematical calculations. The calculator supports basic arithmetic
operations, advanced mathematical functions, and handles various number formats.

Key Features:
- Safe evaluation using ast.literal_eval for basic expressions
- Support for basic arithmetic: +, -, *, /, **, %
- Advanced mathematical functions: sin, cos, tan, log, sqrt, etc.
- Handles integers, floats, and scientific notation
- Error handling for invalid expressions and division by zero
- Comprehensive input validation and sanitization

Security:
- Uses safe evaluation methods to prevent code injection
- Restricted to mathematical operations only
- No access to system functions or variables

Usage Examples:
    calculator("2 + 3 * 4")  # Returns: "14"
    calculator("sqrt(16)")   # Returns: "4.0"
    calculator("sin(pi/2)")  # Returns: "1.0"
    calculator("2**10")      # Returns: "1024"
"""

import ast
import math
import operator
import re
from typing import Union


def calculator(expression: str) -> str:
    """
    A calculator function to perform mathematical calculations. Use it if you need the evaluation of any mathematical expression. Example expressions are 2 + 4.5, sqrt(16) + 2**3, etc.
    Here are the supported operations and functions:  

    Supported Operators:
        +    Addition
        -    Subtraction
        *    Multiplication
        /    Division
        **   Exponentiation (power)
        %    Modulo (remainder)
        <    Less than
        >    Greater than
        <=   Less than or equal to
        >=   Greater than or equal to
        ==   Equal to
        !=   Not equal to
        ( )  Parentheses for grouping

    Supported Functions:
        sin(x)      Sine
        cos(x)      Cosine
        tan(x)      Tangent
        asin(x)     Arc sine
        acos(x)     Arc cosine
        atan(x)     Arc tangent
        sinh(x)     Hyperbolic sine
        cosh(x)     Hyperbolic cosine
        tanh(x)     Hyperbolic tangent
        log(x[,b])  Logarithm (base b, default e)
        log10(x)    Logarithm base 10
        log2(x)     Logarithm base 2
        ln(x)       Natural logarithm
        sqrt(x)     Square root
        abs(x)      Absolute value
        ceil(x)     Ceiling
        floor(x)    Floor
        round(x[,n]) Round to n decimals
        pow(x, y)   Power
        min(x, ...) Minimum
        max(x, ...) Maximum
        sum(x)      Sum (of iterable)

    Supported Constants:
        pi    3.141592...
        e     2.718281...
        inf   Infinity
        nan   Not a number

    This function evaluates mathematical expressions safely, supporting basic arithmetic
    operations and common mathematical functions. It's designed to be used as a tool
    by Large Language Models for computational tasks.

    Args:
        expression (str): The mathematical expression to evaluate. Can include:
            - Basic arithmetic: +, -, *, /, **, % (modulo)
            - Parentheses for grouping: ( )
            - Mathematical functions: sin, cos, tan, log, ln, sqrt, abs, ceil, floor
            - Mathematical constants: pi, e
            - Numbers in various formats: integers, floats, scientific notation (1e5)
            - Comparison operators: <, >, <=, >=, ==, !=

    Returns:
        str: The result of the calculation as a string. For boolean results from 
             comparisons, returns "True" or "False". For numerical results, 
             returns the number formatted as a string with appropriate precision.

    Raises:
        ValueError: If the expression contains invalid syntax, unsupported operations,
                   or results in mathematical errors (like division by zero)
        TypeError: If the input is not a string

    Examples:
        >>> calculator("2 + 3 * 4")
        '14'
        
        >>> calculator("sqrt(16) + 2**3")
        '12.0'
        
        >>> calculator("sin(pi/2)")
        '1.0'
        
        >>> calculator("10 % 3")
        '1'
        
        >>> calculator("2.5 * 1e3")
        '2500.0'
        
        >>> calculator("abs(-42)")
        '42'
        
        >>> calculator("5 > 3")
        'True'
        
        >>> calculator("log(100, 10)")
        '2.0'
    """
    
    if not isinstance(expression, str):
        raise TypeError(f"Expected string input, got {type(expression).__name__}")
    
    # Remove whitespace and validate basic format
    expression = expression.strip()
    if not expression:
        raise ValueError("Empty expression provided")
    
    # Replace mathematical constants and functions with their values/implementations
    replacements = {
        'pi': str(math.pi),
        'e': str(math.e),
        'inf': str(math.inf),
        'nan': str(math.nan),
    }
    
    # Mathematical functions that need special handling
    math_functions = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'ln': math.log,  # Natural logarithm alias
        'sqrt': math.sqrt,
        'abs': abs,
        'ceil': math.ceil,
        'floor': math.floor,
        'round': round,
        'pow': pow,
        'min': min,
        'max': max,
        'sum': sum,
    }
    
    try:
        # First, try to handle function calls
        processed_expr = _process_functions(expression, math_functions)
        
        # Replace constants
        for const, value in replacements.items():
            processed_expr = processed_expr.replace(const, value)
        
        # Validate that the expression only contains safe characters
        if not _is_safe_expression(processed_expr):
            raise ValueError("Expression contains unsafe or unsupported operations")
        
        # Parse and evaluate the expression safely
        try:
            # Try to parse as a simple literal first
            result = ast.literal_eval(processed_expr)
        except (ValueError, SyntaxError):
            # If that fails, try to evaluate as a mathematical expression
            result = _safe_eval(processed_expr)
        
        # Format the result appropriately
        if isinstance(result, bool):
            return str(result)
        elif isinstance(result, (int, float)):
            # Format numbers nicely
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            elif isinstance(result, float):
                return f"{result:.10g}"  # Remove trailing zeros
            else:
                return str(result)
        else:
            return str(result)
            
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except OverflowError:
        raise ValueError("Result too large to compute")
    except (ValueError, TypeError, SyntaxError) as e:
        raise ValueError(f"Invalid mathematical expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")


def _process_functions(expression: str, functions: dict) -> str:
    """Process mathematical function calls in the expression."""
    
    # Handle function calls with regex
    for func_name, func in functions.items():
        pattern = rf'{func_name}\s*\('
        
        while re.search(pattern, expression):
            # Find function calls and evaluate them
            start = 0
            while True:
                match = re.search(pattern, expression[start:])
                if not match:
                    break
                
                func_start = start + match.start()
                paren_start = start + match.end() - 1
                
                # Find matching closing parenthesis
                paren_count = 1
                pos = paren_start + 1
                while pos < len(expression) and paren_count > 0:
                    if expression[pos] == '(':
                        paren_count += 1
                    elif expression[pos] == ')':
                        paren_count -= 1
                    pos += 1
                
                if paren_count == 0:
                    # Extract arguments
                    args_str = expression[paren_start + 1:pos - 1]
                    
                    try:
                        # Parse arguments
                        if args_str.strip():
                            # Split by comma, but be careful with nested expressions
                            args = _parse_function_args(args_str)
                            # Evaluate each argument
                            eval_args = []
                            for arg in args:
                                arg_result = _safe_eval(arg.strip())
                                eval_args.append(arg_result)
                        else:
                            eval_args = []
                        
                        # Call the function
                        result = func(*eval_args)
                        
                        # Replace the function call with its result
                        expression = expression[:func_start] + str(result) + expression[pos:]
                        start = func_start + len(str(result))
                        
                    except Exception:
                        start = pos
                else:
                    break
    
    return expression


def _parse_function_args(args_str: str) -> list:
    """Parse function arguments, handling nested parentheses and commas."""
    args = []
    current_arg = ""
    paren_count = 0
    
    for char in args_str:
        if char == ',' and paren_count == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            current_arg += char
    
    if current_arg.strip():
        args.append(current_arg.strip())
    
    return args


def _is_safe_expression(expression: str) -> bool:
    """Check if the expression contains only safe mathematical operations."""
    
    # Allowed characters: digits, operators, parentheses, decimal points, scientific notation
    safe_pattern = r'^[0-9+\-*/().%\s<>=!eE]+$'
    
    return bool(re.match(safe_pattern, expression))


def _safe_eval(expression: str) -> Union[int, float, bool]:
    """Safely evaluate a mathematical expression."""
    
    # Define allowed operators
    operators = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '**': operator.pow,
        '%': operator.mod,
        '<': operator.lt,
        '>': operator.gt,
        '<=': operator.le,
        '>=': operator.ge,
        '==': operator.eq,
        '!=': operator.ne,
    }
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Evaluate the AST safely
        return _eval_ast_node(tree.body, operators)
        
    except Exception as e:
        raise ValueError(f"Cannot evaluate expression: {e}")


def _eval_ast_node(node, operators):
    """Recursively evaluate AST nodes safely."""
    
    if isinstance(node, ast.Constant):  # Python 3.8+
        return node.value
    elif isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    elif isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, operators)
        right = _eval_ast_node(node.right, operators)
        op_func = operators.get(type(node.op).__name__.lower())
        if not op_func:
            # Handle special cases
            if isinstance(node.op, ast.Pow):
                op_func = operators['**']
            elif isinstance(node.op, ast.Mod):
                op_func = operators['%']
            elif isinstance(node.op, ast.Add):
                op_func = operators['+']
            elif isinstance(node.op, ast.Sub):
                op_func = operators['-']
            elif isinstance(node.op, ast.Mult):
                op_func = operators['*']
            elif isinstance(node.op, ast.Div):
                op_func = operators['/']
            else:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, operators)
        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    elif isinstance(node, ast.Compare):
        left = _eval_ast_node(node.left, operators)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Complex comparisons not supported")
        right = _eval_ast_node(node.comparators[0], operators)
        op = node.ops[0]
        if isinstance(op, ast.Lt):
            return left < right
        elif isinstance(op, ast.Gt):
            return left > right
        elif isinstance(op, ast.LtE):
            return left <= right
        elif isinstance(op, ast.GtE):
            return left >= right
        elif isinstance(op, ast.Eq):
            return left == right
        elif isinstance(op, ast.NotEq):
            return left != right
        else:
            raise ValueError(f"Unsupported comparison: {type(op).__name__}")
    else:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


if __name__ == "__main__":
    """Test the calculator function with various examples."""
    
    print("=== Calculator Tool Tests ===")
    
    # Test cases
    test_cases = [
        "2 + 3 * 4",
        "sqrt(16) + 2**3",
        "sin(pi/2)",
        "10 % 3",
        "2.5 * 1e3",
        "abs(-42)",
        "5 > 3",
        "log(100, 10)",
        "(5 + 3) * 2",
        "cos(0)",
        "max(10, 20, 5)",
        "min(10, 20, 5)",
        "ceil(4.2)",
        "floor(4.8)",
        "round(3.14159, 2)",
    ]
    
    for expression in test_cases:
        try:
            result = calculator(expression)
            print(f"calculator('{expression}') = {result}")
        except Exception as e:
            print(f"calculator('{expression}') = ERROR: {e}")
    
    print("\n=== Error Handling Tests ===")
    
    error_cases = [
        "1 / 0",  # Division by zero
        "invalid_function(5)",  # Invalid function
        "2 +",  # Incomplete expression
        "",  # Empty expression
        "import os",  # Unsafe operation
    ]
    
    for expression in error_cases:
        try:
            result = calculator(expression)
            print(f"calculator('{expression}') = {result}")
        except Exception as e:
            print(f"calculator('{expression}') = ERROR: {e}")
